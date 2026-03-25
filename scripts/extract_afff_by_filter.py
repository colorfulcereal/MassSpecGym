# python scripts/extract_afff_by_filter.py
#
# Applies Jonathan's m/C and MD/C filters to all 199 annotated PFAS compounds
# in the AFFF feature_list.csv, then extracts matching MS2 spectra from all
# mzML files (pos/ and neg/) for use as annotated training data.
#
# Filters:
#   m/C   > 30          (high fluorine content)
#   MD/C  in (-0.01, 0.003)  (negative mass defect signature of fluorine)

import re
import os
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from dreams.utils.data import MSData
from dreams.api import dreams_predictions
from dreams.utils.dformats import DataFormatA
from dreams.utils.io import append_to_stem

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AFFF_DIR     = Path('/teamspace/studios/this_studio/MassSpecGym/afff/')
FEATURE_LIST = AFFF_DIR / 'feature_list.csv'
MZ_TOLERANCE = 0.02   # Da
MC_THRESHOLD = 30.0
MDC_MIN      = -0.01
MDC_MAX      = 0.003
OUTPUT_CSV   = Path('afff_filter_results.csv')
SUMMARY_CSV  = Path('afff_filter_summary.csv')

# ---------------------------------------------------------------------------
# Step 1 — Load feature list and compute m/C and MD/C
# ---------------------------------------------------------------------------
def parse_carbon_count(formula: str) -> int:
    """Extract carbon count from Hill-notation formula e.g. 'C8HF15O2' → 8."""
    match = re.search(r'C(\d+)', formula)
    if match:
        return int(match.group(1))
    # Handle edge case: formula has C but no number (single carbon)
    if re.search(r'C(?![a-z])', formula):
        return 1
    return 0

def compute_mc_mdc(smiles: str, formula: str):
    """Return (exact_mass, C, m_over_C, MD_over_C) or None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    exact_mass = ExactMolWt(mol)
    C = parse_carbon_count(formula)
    if C == 0:
        return None
    m_over_C  = exact_mass / C
    MD_over_C = (exact_mass - round(exact_mass)) / C
    return exact_mass, C, m_over_C, MD_over_C

features = pd.read_csv(FEATURE_LIST)
features = features[features['polarity'].str.lower().str.contains('pos')].reset_index(drop=True)
print(f"Loaded {len(features)} positive-mode compounds from feature list")
print(f"Columns: {features.columns.tolist()}\n")

# Compute filters
records = []
for _, row in features.iterrows():
    result = compute_mc_mdc(row['SMILES'], row['formula'])
    if result is None:
        print(f"  WARNING: could not parse '{row['name']}' (SMILES={row['SMILES']})")
        records.append({**row.to_dict(), 'exact_mass': None, 'C': None,
                        'm_over_C': None, 'MD_over_C': None, 'passes_filter': False})
        continue
    exact_mass, C, m_over_C, MD_over_C = result
    passes = (m_over_C > MC_THRESHOLD) and (MDC_MIN < MD_over_C < MDC_MAX)
    records.append({**row.to_dict(), 'exact_mass': exact_mass, 'C': C,
                    'm_over_C': m_over_C, 'MD_over_C': MD_over_C, 'passes_filter': passes})

features_ext = pd.DataFrame(records)
passing = features_ext[features_ext['passes_filter']]
failing = features_ext[~features_ext['passes_filter']]

print(f"Filter results (m/C > {MC_THRESHOLD}, {MDC_MIN} < MD/C < {MDC_MAX}):")
print(f"  Passing : {len(passing)} / {len(features_ext)}")
print(f"  Failing : {len(failing)}")

if len(failing) > 0:
    print("\nFailing compounds:")
    print(failing[['name', 'formula', 'm_over_C', 'MD_over_C']].to_string(index=False))

print()

# ---------------------------------------------------------------------------
# Step 2 — Extract matching MS2 spectra from all mzML files
# ---------------------------------------------------------------------------
passing_mzs    = passing['mz'].values
passing_index  = passing.reset_index(drop=True)

def match_feature(prec_mz, feature_mzs, tol=MZ_TOLERANCE):
    diffs = np.abs(feature_mzs - prec_mz)
    idx = np.argmin(diffs)
    if diffs[idx] <= tol:
        return idx, float(diffs[idx])
    return None, None

all_rows   = []
mzml_dirs  = [AFFF_DIR / 'pos']

for mzml_dir in mzml_dirs:
    mzml_dir = Path(mzml_dir)
    if not mzml_dir.exists():
        print(f"Directory not found, skipping: {mzml_dir}")
        continue

    mzml_files = sorted(mzml_dir.glob('*.mzML')) + sorted(mzml_dir.glob('*.mzml'))
    print(f"\n--- {mzml_dir.name}/ : {len(mzml_files)} file(s) ---")

    for mzml_pth in mzml_files:
        print(f"Processing: {mzml_pth.name}")

        try:
            msdata = MSData.from_mzml(mzml_pth, verbose_parser=True)
        except ValueError as e:
            print(f"  Skipping: {e}")
            continue

        spectra  = msdata['spectrum']
        prec_mzs = msdata['precursor_mz']

        dformat      = DataFormatA()
        quality_lvls = [dformat.val_spec(s, p, return_problems=True)
                        for s, p in zip(spectra, prec_mzs)]
        hq_idx = np.where(np.array(quality_lvls) == 'All checks passed')[0]
        print(f"  {len(hq_idx)}/{len(spectra)} spectra passed quality filter")

        if len(hq_idx) == 0:
            continue

        hq_pth = append_to_stem(mzml_pth, 'high_quality').with_suffix('.hdf5')
        msdata.form_subset(idx=hq_idx, out_pth=hq_pth)
        msdata_hq   = MSData.load(hq_pth)
        df_hq       = msdata_hq.to_pandas()
        hq_prec_mzs = df_hq['precursor_mz'].values

        n_matched = 0
        for i, prec_mz in enumerate(hq_prec_mzs):
            feat_idx, mz_err = match_feature(prec_mz, passing_mzs)
            if feat_idx is None:
                continue
            n_matched += 1
            feat_row = passing_index.iloc[feat_idx]
            all_rows.append({
                'compound_name' : feat_row['name'],
                'formula'       : feat_row['formula'],
                'SMILES'        : feat_row['SMILES'],
                'adduct'        : feat_row['adduct'],
                'polarity'      : feat_row['polarity'],
                'mz_feature'    : passing_mzs[feat_idx],
                'mz_spectrum'   : prec_mz,
                'mz_error_da'   : mz_err,
                'm_over_C'      : feat_row['m_over_C'],
                'MD_over_C'     : feat_row['MD_over_C'],
                'passes_filter' : True,
                'file'          : mzml_pth.name,
            })
        print(f"  Matched {n_matched} spectra to feature list")

# ---------------------------------------------------------------------------
# Step 3 — Save and summarise
# ---------------------------------------------------------------------------
if not all_rows:
    print("\nNo spectra matched. Check AFFF_DIR path and mzML files.")
else:
    results = pd.DataFrame(all_rows)
    results.to_csv(OUTPUT_CSV, index=False)

    summary = (results.groupby(['compound_name', 'formula', 'adduct', 'polarity',
                                'm_over_C', 'MD_over_C'])
                      .agg(n_spectra=('mz_spectrum', 'count'),
                           files=('file', lambda x: ', '.join(sorted(set(x)))))
                      .reset_index()
                      .sort_values('n_spectra', ascending=False))
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"  Compounds passing filter   : {len(passing)}")
    print(f"  Total spectra extracted    : {len(results)}")
    print(f"  Breakdown by polarity:")
    print(results['polarity'].value_counts().to_string())
    print(f"\n  Compounds with ≥1 spectrum matched:")
    print(f"    {results['compound_name'].nunique()} / {len(passing)}")
    print(f"\n  Top matched compounds:")
    print(summary[['compound_name', 'polarity', 'n_spectra']].head(15).to_string(index=False))
    print(f"\nResults saved to : {OUTPUT_CSV}")
    print(f"Summary saved to : {SUMMARY_CSV}")
