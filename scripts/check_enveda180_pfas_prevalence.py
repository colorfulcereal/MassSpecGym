"""
Gate check for the Enveda-180 dataset (https://zenodo.org/records/20436851):
computes true PFAS-chain prevalence (using the same labeling rule already
used throughout this repo) before investing in a full conversion pipeline.

Usage:
    python scripts/check_enveda180_pfas_prevalence.py --input enveda-180-filtered.jsonl.gz
    python scripts/check_enveda180_pfas_prevalence.py --input enveda-180-filtered.jsonl --limit 20000
"""
import argparse
import gzip
import json
from collections import Counter

from massspecgym.data.transforms import MolToPFASVector


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to enveda-180-filtered.jsonl(.gz)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of lines read (for quick dry runs)")
    args = parser.parse_args()

    pfas_checker = MolToPFASVector()

    # Pass 1: collect unique molecules (inchikey -> smiles) and per-spectrum (inchikey, ionmode)
    unique_mols = {}
    spectra = []
    n_lines = 0
    n_parse_errors = 0
    with open_maybe_gzip(args.input) as f:
        for line in f:
            if args.limit and n_lines >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_parse_errors += 1
                continue
            inchikey = rec.get("inchikey")
            smiles = rec.get("smiles")
            ionmode = rec.get("ionmode")
            if inchikey and inchikey not in unique_mols:
                unique_mols[inchikey] = smiles
            spectra.append((inchikey, ionmode))

    print(f"Parsed {n_lines} lines ({n_parse_errors} JSON parse errors -- expected if the "
          f"input was truncated, e.g. a byte-range sample)")
    print(f"Unique molecules (by inchikey): {len(unique_mols)}")

    # Compute PFAS-chain label per unique molecule, using the same rule
    # already used everywhere else in this repo (MolToFluorinatedTypeVector,
    # the ISEF paper's own labeling methodology).
    pfas_mols = set()
    n_fluorine_mols = 0
    for inchikey, smiles in unique_mols.items():
        if not smiles:
            continue
        if "F" in smiles.upper():
            n_fluorine_mols += 1
        try:
            if pfas_checker.has_pf_chain_ge_2(smiles):
                pfas_mols.add(inchikey)
        except Exception:
            continue

    n_mols = len(unique_mols)
    print(f"\nUnique molecules containing any fluorine: {n_fluorine_mols} "
          f"({100 * n_fluorine_mols / n_mols:.2f}%)" if n_mols else "")
    print(f"Unique molecules meeting PFAS-chain criterion (>=2 connected CF2/CF3): "
          f"{len(pfas_mols)} ({100 * len(pfas_mols) / n_mols:.2f}%)" if n_mols else "")

    # Spectrum-level breakdown by ion mode (mirrors the analysis done on the
    # existing merged dataset in dreams-pfas-workshop-paper.md section 10)
    ionmode_counts = Counter()
    ionmode_pfas_counts = Counter()
    for inchikey, ionmode in spectra:
        mode_key = ionmode if ionmode else "MISSING"
        ionmode_counts[mode_key] += 1
        if inchikey in pfas_mols:
            ionmode_pfas_counts[mode_key] += 1

    print(f"\n=== Spectrum-level ion mode distribution (n={len(spectra)}) ===")
    for mode, count in ionmode_counts.most_common():
        pfas_count = ionmode_pfas_counts.get(mode, 0)
        pct_pfas = 100 * pfas_count / count if count else 0.0
        print(f"  {mode:15s}: n={count:8d}  PFAS-chain-positive={pfas_count:6d} ({pct_pfas:.4f}%)")

    print(f"\nTotal spectra belonging to PFAS-chain molecules: {sum(ionmode_pfas_counts.values())}")
    print(f"Total unique PFAS-chain molecules: {len(pfas_mols)}")

    if len(pfas_mols) == 0:
        print("\n[GATE CHECK RESULT] No PFAS-chain molecules found -- before running "
              "prepare_enveda180_dataset.py, confirm this holds on the FULL dataset "
              "(not a truncated/partial sample) before concluding Enveda-180 doesn't "
              "help the chain-PFAS retraining goal.")


if __name__ == "__main__":
    main()
