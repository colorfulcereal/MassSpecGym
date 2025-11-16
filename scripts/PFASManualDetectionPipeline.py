import pandas as pd
from rdkit import Chem
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid
from IPython.display import display
import csv

# Example: Load your dataframe
pfas_labeled_df = pd.read_csv('/teamspace/studios/this_studio/files/merged_massspec_nist20_nist_new_with_fold.tsv', sep='\t')

print(pfas_labeled_df.iloc[0])

# pick only the validation set
pfas_labeled_df = pfas_labeled_df[pfas_labeled_df['fold'] == 'val']
print(f"Validation set size: {len(pfas_labeled_df)}")

import numpy as np
import pandas as pd

# -------------------
# PARAMETERS (tune these)
# -------------------
CF2_MONO = 49.9968          # monoisotopic CF2 mass
CF2_NOMINAL = 50.0          # nominal CF2 mass
KMD_TOL = 0.005             # Kendrick mass defect tolerance (normalized units)
PPM_TOL_FRAG = 10           # ppm tolerance for fragment matches
MZ_TOL_PPM = 5              # ppm tolerance for precursor differences
DELTA_CF2 = CF2_MONO        # CF2 spacing for homologous series
MIN_REL_INTENSITY = 0.01    # minimum relative intensity of 1% in the MS2 spectrum

DIAGNOSTIC_FRAGS = {
    "CF3": 68.9952,    # CF3+
    "C2F5": 118.9916,  # C2F5+
    "C3F5": 131.00,
    "C3F7-": 168.987,
    "C4F9-": 218.986,
    "SO3−": 79.95736,
    "HSO4−": 96.96010,
    "FSO3−":98.95576,
    # extend with PFAS-specific fragments if you have them
}

# -------------------
# HELPERS
# -------------------
def ppm_to_da(mz, ppm):
    return mz * ppm / 1e6

def str_to_float_list(s):
    """Convert comma-separated string to list of floats."""
    if pd.isna(s): 
        return []
    return [float(x) for x in s.split(",")]


# -------------------
# STEP 1: Parse MS2 columns into lists
# -------------------
pfas_labeled_df["ms2_fragments"] = pfas_labeled_df["mzs"].apply(str_to_float_list)
pfas_labeled_df["ms2_intensities"] = pfas_labeled_df["intensities"].apply(str_to_float_list)

# -------------------
# STEP 2: 
# a. Mass Defect
# b. Kendrick Mass Defect (KMD)
# -------------------

def compute_mass_defect(df, mz_col="precursor_mz"):
    df = df.copy()
    nominal = np.floor(df[mz_col]).astype(int)
    df["mass_defect"] = df[mz_col] - nominal
    return df

def compute_kendrick(df, mz_col="precursor_mz", ref_mass=CF2_MONO, ref_nominal=CF2_NOMINAL):
    df = df.copy()
    km = df[mz_col] * (ref_nominal / ref_mass)
    km_nominal = np.round(km)
    kmd = km_nominal - km
    df["kendrick_mass"] = km
    df["kendrick_nominal"] = km_nominal
    df["kendrick_md"] = kmd
    df["adjusted_mass_defect"] = df["kendrick_md"] * (ref_mass / ref_nominal)
    return df

def annotate_mass_defect_filter(df, md_col="mass_defect", low=-0.1, high=0.1):
    df = df.copy()
    df["md_within"] = (df[md_col] >= low) & (df[md_col] <= high)
    return df

# -------------------
# STEP 3: Diagnostic fragment filtering
# -------------------

def fragments_have_diagnostic(ms2_list, intensities, diag_frags=DIAGNOSTIC_FRAGS, ppm_tol=PPM_TOL_FRAG, min_rel_int=MIN_REL_INTENSITY):
    if ms2_list is None or len(ms2_list) == 0:
        return False
    max_int = max(intensities) if intensities else 1.0
    for mz, inten in zip(ms2_list, intensities):
        for ref_mz in diag_frags.values():
            delta_ppm = abs((mz - ref_mz) / ref_mz) * 1e6
            if delta_ppm <= ppm_tol and (inten / max_int) >= min_rel_int:
                return True
    return False

def annotate_diag_fragment_matches(df, ms2_col="ms2_fragments", inten_col="ms2_intensities",
                                   diag_frags=DIAGNOSTIC_FRAGS, ppm_tol=PPM_TOL_FRAG,
                                   min_rel_int=0.01):
    df = df.copy()
    df["has_diag_frag"] = df.apply(
        lambda row: fragments_have_diagnostic(
            row[ms2_col], row[inten_col], diag_frags, ppm_tol, min_rel_int
        ),
        axis=1
    )
    return df

# -------------------
# STEP 4: Homologous series detection (CF2 spacing)
# -------------------
def find_homologous_series(df, mz_col="precursor_mz", delta=DELTA_CF2, mz_tol_ppm=MZ_TOL_PPM):
    df = df.copy().reset_index(drop=True)
    mzs = df[mz_col].values
    N = len(mzs)
    hom_counts = np.ones(N, dtype=int)
    for i in range(N):
        mi = mzs[i]
        for n in range(1, 7):  # search up to +/-6 CF2 units
            target = mi + n * delta
            tol_da = ppm_to_da(target, mz_tol_ppm)
            matches = np.abs(mzs - target) <= tol_da
            hom_counts[i] += matches.sum()
    df["homologue_count"] = hom_counts
    return df

# -------------------
# STEP 5: Scoring and PFAS prediction
# -------------------
def score_and_select(df, w_md=1.0, w_kmd=1.0, w_frag=4.0, w_homo=1.0, 
                     kmd_tol=KMD_TOL, md_low=-0.1, md_high=0.1):
    df = df.copy()

    # Mass defect filter
    df["md_within"] = (df["mass_defect"] >= md_low) & (df["mass_defect"] <= md_high)

    # Kendrick filter
    df["kmd_within"] = np.abs(df["kendrick_md"]) <= kmd_tol

    # Fragment score
    df["frag_score"] = df["has_diag_frag"].astype(int)

    # Homologue score
    df["homo_score"] = (df["homologue_count"] >= 3).astype(int)

    # Weighted total score
    df["score"] = (
        df["md_within"].astype(float) * w_md +
        df["kmd_within"].astype(float) * w_kmd +
        df["frag_score"] * w_frag +
        df["homo_score"] * w_homo
    )

    # Prediction
    df["predicted_pfas"] = df["score"] > 0
    return df

# -------------------
# STEP 6: Precision / Recall
# -------------------
def precision_recall(df, truth_col="is_PFAS", pred_col="predicted_pfas"):
    tp = ((df[truth_col] == 1) & (df[pred_col] == True)).sum()
    fp = ((df[truth_col] == 0) & (df[pred_col] == True)).sum()
    fn = ((df[truth_col] == 1) & (df[pred_col] == False)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": precision, "recall": recall}

# -------------------
# RUN PIPELINE
# -------------------
df_kmd = compute_mass_defect(pfas_labeled_df, mz_col="precursor_mz") # MS1 
df_kmd = annotate_mass_defect_filter(df_kmd) # MS1 
df_kmd = compute_kendrick(df_kmd, mz_col="precursor_mz") # MS1 
df_kmd = annotate_diag_fragment_matches(df_kmd, ms2_col="ms2_fragments", inten_col="ms2_intensities") # MS2
df_kmd = find_homologous_series(df_kmd, mz_col="precursor_mz") # MS2
df_scored = score_and_select(df_kmd)

if "is_PFAS" in df_scored.columns:
    metrics = precision_recall(df_scored, truth_col="is_PFAS", pred_col="predicted_pfas")
    print("Precision / Recall:", metrics)

# Inspect top-scoring candidates
candidates = df_scored.sort_values("score", ascending=False).reset_index(drop=True)
# print(candidates[["precursor_mz","score","has_diag_frag","homologue_count","predicted_pfas"]].head(20))

print ("starting parameter search ...")

# Parameters Search

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid

param_grid = {
    "w_md": [0.5, 1.0],
    "w_kmd": [0.5, 1.0],
    "w_frag": [2.0, 4.0, 5.0],
    "w_homo": [0.5, 1.0],
    "homo_min": [2, 3],
    "score_cutoff": [2, 3, 4],
    "ppm_tol": [5, 10, 20],          # fragment mass tolerance
    "min_rel_int": [0.01, 0.02, 0.05] # fragment intensity threshold
}

# param_grid = {
#     "w_md": [0.5],
#     "w_kmd": [0.5],
#     "w_frag": [2.0],
#     "w_homo": [0.5],
#     "homo_min": [2],
#     "score_cutoff": [2],
#     "ppm_tol": [5],          # fragment mass tolerance
#     "min_rel_int": [0.01, 0.02] # fragment intensity threshold
# }


def score_and_select_grid_search(df, params):
    """
    Compute PFAS evidence score given parameter weights and thresholds.
    """
    w_md = params["w_md"]
    w_kmd = params["w_kmd"]
    w_frag = params["w_frag"]
    w_homo = params["w_homo"]
    cutoff = params["score_cutoff"]

    # build score
    df = df.copy()
    df["score"] = (
        w_md * df["md_within"].astype(int) +
        w_kmd * df["kmd_within"].astype(int) +
        w_frag * df["has_diag_frag"].astype(int) +
        w_homo * (df["homologue_count"] >= params["homo_min"]).astype(int)
    )
    df["predicted_pfas"] = df["score"] >= cutoff
    return df


results = []
for params in ParameterGrid(param_grid):
    # 1. Recompute fragment matches for current ppm and intensity thresholds
    df_tmp = annotate_diag_fragment_matches(
        df_scored,
        ms2_col="ms2_fragments",
        inten_col="ms2_intensities",
        diag_frags=DIAGNOSTIC_FRAGS,
        ppm_tol=params["ppm_tol"],
        min_rel_int=params["min_rel_int"]
    )
    
    # 2. Score and classify
    df_scored_grid_search = score_and_select_grid_search(df_tmp, params)
    
    # 3. Compute metrics
    y_true = df_scored_grid_search["is_PFAS"].astype(int)
    y_pred = df_scored_grid_search["predicted_pfas"].astype(int)
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    results.append({
        **params,
        "precision": prec * 100,
        "recall": rec * 100,
        "f1": f1
    })

results_df = pd.DataFrame(results)

top_results = results_df.sort_values(by="recall", ascending=False)

# Save all results to CSV
output_path = "./grid_search_results.csv"

top_results.to_csv(
    output_path,
    index=False,
    sep=",",                     # comma-separated
    quoting=csv.QUOTE_ALL,       # quote all fields to avoid issues
    lineterminator="\n",        # ensure one row per line
    escapechar="\\",             # escape special characters if needed
)

print(f"✅ Full grid search results written to {output_path}")