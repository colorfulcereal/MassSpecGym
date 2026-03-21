# python scripts/test_afff_pos.py
#
# Evaluates DreaMS-PFAS on Jonathan's AFFF positive-mode data.
# All compounds in the feature_list.csv are true PFAS (ground truth = 1).
# Runs the model on ALL high-quality spectra in afff/pos/ mzML files, labels
# each spectrum as PFAS if its precursor m/z matches a feature list entry
# within MZ_TOLERANCE, then reports precision and recall.

from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, classification_report

from massspecgym.models.pfas import HalogenDetectorDreamsTest
from dreams.utils.data import MSData
from dreams.api import dreams_predictions
from dreams.utils.dformats import DataFormatA
from dreams.utils.io import append_to_stem

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AFFF_DIR       = Path("~/Downloads/afff").expanduser()
POS_DIR        = AFFF_DIR / "pos"
FEATURE_LIST   = AFFF_DIR / "feature_list.csv"
OUTPUT_CSV     = Path("afff_pos_results.csv")
MZ_TOLERANCE   = 0.02   # Da — appropriate for high-res qTOF
N_PEAKS        = 60
THRESHOLD      = 0.2
POLARITY_KEY   = "pos"  # string to match in the polarity column

_LOCAL_CKPT  = Path.home() / "Downloads" / "HalogenDetection-FocalLoss-MergedMassSpecNIST20_NISTNew_NormalPFAS_ujmvyfxm_checkpoints_epoch=0-step=9285.ckpt"
_REMOTE_CKPT = "/teamspace/studios/this_studio/HalogenDetection-FocalLoss-MergedMassSpecNIST20_NISTNew_NormalPFAS/ujmvyfxm/checkpoints/epoch=0-step=9285.ckpt"

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
ckpt_path = str(_LOCAL_CKPT) if _LOCAL_CKPT.exists() else _REMOTE_CKPT
print(f"Loading checkpoint: {ckpt_path}")
model = HalogenDetectorDreamsTest.load_from_checkpoint(ckpt_path)
model.eval()

# ---------------------------------------------------------------------------
# Load feature list — all entries are PFAS; keep positive-mode rows
# ---------------------------------------------------------------------------
features = pd.read_csv(FEATURE_LIST)
print(f"Feature list columns: {features.columns.tolist()}")
print(f"Feature list polarity values: {features['polarity'].unique()}")

pos_features = features[features["polarity"].str.lower().str.contains(POLARITY_KEY)].copy()
pos_mzs = pos_features["mz"].values  # array of known PFAS precursor m/z values
print(f"\nPositive-mode PFAS features: {len(pos_features)}")

def match_feature(prec_mz, feature_mzs, tol=MZ_TOLERANCE):
    """Return the closest matching feature index and m/z error, or None if no match."""
    diffs = np.abs(feature_mzs - prec_mz)
    idx = np.argmin(diffs)
    if diffs[idx] <= tol:
        return idx, diffs[idx]
    return None, None

# ---------------------------------------------------------------------------
# Process each mzML in afff/pos/
# ---------------------------------------------------------------------------
all_rows = []

mzml_files = sorted(POS_DIR.glob("*.mzML")) + sorted(POS_DIR.glob("*.mzml"))
if not mzml_files:
    raise FileNotFoundError(f"No .mzML files found in {POS_DIR}")

print(f"\nFound {len(mzml_files)} mzML file(s) in {POS_DIR}\n")

for mzml_pth in mzml_files:
    print(f"Processing: {mzml_pth.name}")

    try:
        msdata = MSData.from_mzml(mzml_pth, verbose_parser=True)
    except ValueError as e:
        print(f"  Skipping: {e}")
        continue

    spectra  = msdata["spectrum"]
    prec_mzs = msdata["precursor_mz"]

    # Quality filter
    dformat = DataFormatA()
    quality_lvls = [dformat.val_spec(s, p, return_problems=True) for s, p in zip(spectra, prec_mzs)]
    hq_idx = np.where(np.array(quality_lvls) == "All checks passed")[0]
    print(f"  {len(hq_idx)}/{len(spectra)} spectra passed quality filter")

    if len(hq_idx) == 0:
        print("  No high-quality spectra — skipping file.")
        continue

    hq_pth = append_to_stem(mzml_pth, "high_quality").with_suffix(".hdf5")
    msdata.form_subset(idx=hq_idx, out_pth=hq_pth)
    msdata_hq = MSData.load(hq_pth)

    df_hq = msdata_hq.to_pandas()
    hq_prec_mzs = df_hq["precursor_mz"].values

    # Run model on all high-quality spectra
    raw_preds = dreams_predictions(spectra=msdata_hq, model_ckpt=model, n_highest_peaks=N_PEAKS)
    probs = torch.sigmoid(torch.from_numpy(raw_preds)).cpu().numpy()

    # Label each spectrum: 1 if precursor m/z matches a feature list entry
    for i, (prec_mz, prob) in enumerate(zip(hq_prec_mzs, probs)):
        feat_idx, mz_err = match_feature(prec_mz, pos_mzs)
        is_pfas = 1 if feat_idx is not None else 0

        row = {
            "file":                 mzml_pth.name,
            "spectrum_precursor_mz": prec_mz,
            "true_label":           is_pfas,
            "PFAS_pred":            float(prob),
            "predicted_PFAS":       int(prob >= THRESHOLD),
        }
        if feat_idx is not None:
            row["compound_name"] = pos_features.iloc[feat_idx]["name"]
            row["formula"]       = pos_features.iloc[feat_idx].get("formula", "")
            row["adduct"]        = pos_features.iloc[feat_idx].get("adduct", "")
            row["feature_mz"]    = pos_mzs[feat_idx]
            row["mz_error_da"]   = float(mz_err)
        else:
            row["compound_name"] = ""
            row["formula"]       = ""
            row["adduct"]        = ""
            row["feature_mz"]    = np.nan
            row["mz_error_da"]   = np.nan

        all_rows.append(row)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if not all_rows:
    print("\nNo spectra processed. Check that mzML files are present and readable.")
else:
    results = pd.DataFrame(all_rows)
    results.to_csv(OUTPUT_CSV, index=False)

    y_true = results["true_label"].values
    y_pred = results["predicted_PFAS"].values

    n_total    = len(results)
    n_pfas     = int(y_true.sum())
    n_non_pfas = n_total - n_pfas

    print(f"\n{'='*55}")
    print(f"  Total high-quality spectra processed : {n_total}")
    print(f"  Matched to feature list (true PFAS)  : {n_pfas}")
    print(f"  Unmatched (assumed non-PFAS)          : {n_non_pfas}")
    print(f"{'='*55}")

    if n_pfas == 0:
        print("\nWARNING: No spectra matched the feature list within the m/z tolerance.")
        print("Consider increasing MZ_TOLERANCE or checking the polarity filter.")
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        print(f"\n  Threshold : {THRESHOLD}")
        print(f"  Precision : {precision:.3f}")
        print(f"  Recall    : {recall:.3f}")
        print(f"  F1        : {f1:.3f}")
        print(f"\n{classification_report(y_true, y_pred, target_names=['non-PFAS', 'PFAS'], zero_division=0)}")
        print(f"Results saved to: {OUTPUT_CSV}")
