"""
Converts the Enveda-180 dataset (https://zenodo.org/records/20436851,
filtered JSONL variant) into a TSV matching the column schema of the
existing merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv, so it can
be pointed at directly by the existing PFAS training scripts (pth=...)
with no dataset/model code changes.

Only run this after scripts/check_enveda180_pfas_prevalence.py has been
run on the FULL dataset and shown the PFAS-chain prevalence is worth
retraining on.

Usage:
    python scripts/prepare_enveda180_dataset.py \\
        --input enveda-180-filtered.jsonl.gz \\
        --output enveda180_prepared_with_fold.tsv \\
        --val-frac 0.2
"""
import argparse
import gzip
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from massspecgym.data.transforms import MolToPFASVector

TARGET_COLUMNS = [
    "Unnamed: 0.1", "Unnamed: 0", "identifier", "mzs", "intensities", "smiles",
    "inchikey", "formula", "precursor_formula", "parent_mass", "precursor_mz",
    "adduct", "instrument_type", "collision_energy", "fold",
    "simulation_challenge", "name", "is_PFAS",
    "ion_mode_true",  # bonus column, not in the original schema -- see plan
]


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def assign_folds(mol_info: dict, val_frac: float, seed: int) -> dict:
    """
    mol_info: inchikey -> {"is_pfas": bool, "has_pos": bool, "has_neg": bool}
    Returns: inchikey -> "train" | "val"

    Molecule-level split (every spectrum of a given molecule gets the same
    fold), stratified by (is_pfas, has_pos, has_neg) so PFAS status and ion
    mode coverage are proportionally represented in both folds. This is a
    stopgap for the full Murcko-histogram scaffold split used in the
    original ISEF dataset (not reimplemented here per plan decision).
    """
    inchikeys = list(mol_info.keys())
    strata = [
        (mol_info[ik]["is_pfas"], mol_info[ik]["has_pos"], mol_info[ik]["has_neg"])
        for ik in inchikeys
    ]
    strata_counts = pd.Series(strata).value_counts()
    # train_test_split requires >=2 members per stratum; fall back to no
    # stratification for singleton strata (rare, e.g. a lone PFAS molecule
    # combination) so the split doesn't crash.
    singleton_strata = set(strata_counts[strata_counts < 2].index)
    stratify = [s if s not in singleton_strata else None for s in strata]
    if any(s is None for s in stratify):
        print(f"[assign_folds] {sum(1 for s in stratify if s is None)} molecules in "
              f"singleton strata -- excluded from stratification (still split, just randomly)")

    train_idx, val_idx = train_test_split(
        np.arange(len(inchikeys)),
        test_size=val_frac,
        random_state=seed,
        stratify=strata if not any(s is None for s in stratify) else None,
    )
    fold_map = {}
    for i in train_idx:
        fold_map[inchikeys[i]] = "train"
    for i in val_idx:
        fold_map[inchikeys[i]] = "val"
    return fold_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to enveda-180-filtered.jsonl(.gz)")
    parser.add_argument("--output", required=True, help="Path to write the output TSV")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction (molecule-level split)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of lines read (for dry runs)")
    args = parser.parse_args()

    pfas_checker = MolToPFASVector()

    # ---- Pass 1: collect per-molecule info (is_pfas, has_pos, has_neg) ----
    mol_info = {}
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
            if not inchikey:
                continue
            if inchikey not in mol_info:
                smiles = rec.get("smiles")
                is_pfas = False
                if smiles:
                    try:
                        is_pfas = bool(pfas_checker.has_pf_chain_ge_2(smiles))
                    except Exception:
                        is_pfas = False
                mol_info[inchikey] = {"is_pfas": is_pfas, "has_pos": False, "has_neg": False}
            ionmode = (rec.get("ionmode") or "").lower()
            if "positive" in ionmode:
                mol_info[inchikey]["has_pos"] = True
            elif "negative" in ionmode:
                mol_info[inchikey]["has_neg"] = True

    print(f"Pass 1 done: {n_lines} lines read ({n_parse_errors} JSON parse errors)")
    print(f"Unique molecules: {len(mol_info)}")
    n_pfas_mols = sum(1 for v in mol_info.values() if v["is_pfas"])
    print(f"Unique PFAS-chain molecules: {n_pfas_mols} ({100 * n_pfas_mols / len(mol_info):.4f}%)"
          if mol_info else "")

    fold_map = assign_folds(mol_info, args.val_frac, args.seed)

    # ---- Pass 2: stream-write output rows ----
    n_written = 0
    ionmode_counts = {}
    ionmode_pfas_counts = {}
    with open_maybe_gzip(args.input) as f, open(args.output, "w") as out:
        out.write("\t".join(TARGET_COLUMNS) + "\n")
        n_lines2 = 0
        for line in f:
            if args.limit and n_lines2 >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            n_lines2 += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            inchikey = rec.get("inchikey", "")
            info = mol_info.get(inchikey, {"is_pfas": False})
            is_pfas = int(info["is_pfas"])
            fold = fold_map.get(inchikey, "train")
            ionmode_true = rec.get("ionmode", "")

            row = {
                "Unnamed: 0.1": "",
                "Unnamed: 0": "",
                "identifier": rec.get("title", ""),
                "mzs": ",".join(rec.get("mzs", [])),
                "intensities": ",".join(rec.get("intensities", [])),
                "smiles": rec.get("smiles", ""),
                "inchikey": inchikey,
                "formula": rec.get("formula", ""),
                "precursor_formula": rec.get("adduct_formula", ""),
                "parent_mass": "",
                "precursor_mz": rec.get("pepmass", ""),
                "adduct": rec.get("adduct", ""),
                "instrument_type": rec.get("instrument_type", ""),
                "collision_energy": rec.get("collision_energies", ""),
                "fold": fold,
                "simulation_challenge": "",
                "name": "",
                "is_PFAS": is_pfas,
                "ion_mode_true": ionmode_true,
            }
            out.write("\t".join(str(row[c]) for c in TARGET_COLUMNS) + "\n")
            n_written += 1

            mode_key = ionmode_true if ionmode_true else "MISSING"
            ionmode_counts[mode_key] = ionmode_counts.get(mode_key, 0) + 1
            if is_pfas:
                ionmode_pfas_counts[mode_key] = ionmode_pfas_counts.get(mode_key, 0) + 1

    print(f"\nWrote {n_written} spectra to {args.output}")
    print(f"\n=== Spectrum-level ion mode distribution in output ===")
    for mode, count in sorted(ionmode_counts.items(), key=lambda x: -x[1]):
        pfas_count = ionmode_pfas_counts.get(mode, 0)
        pct_pfas = 100 * pfas_count / count if count else 0.0
        print(f"  {mode:15s}: n={count:8d}  PFAS-chain-positive={pfas_count:6d} ({pct_pfas:.4f}%)")


if __name__ == "__main__":
    main()
