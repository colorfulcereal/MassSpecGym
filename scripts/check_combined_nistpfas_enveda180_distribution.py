"""
Combines NIST-PFAS (from the existing merged TSV, excluding NIST20 and
MassSpecGym) with the full Enveda-180 dataset, then reports the
distribution of PFAS types -- chain-PFAS (>=2 connected CF2/CF3),
isolated-CF2-only, isolated-CF3-only, and drug-like PFAS (either isolated
type) -- cut by ionization mode (positive/negative).

NIST-PFAS is identified as the negative-ion-mode subset of the merged TSV
(confirmed in dreams-pfas-workshop-paper.md section 10: the 24,391
negative-mode rows in that file match NIST-PFAS's exact total from the
ISEF paper's Table 2, with zero crossover from MassSpecGym/NIST20).

Usage:
    python scripts/check_combined_nistpfas_enveda180_distribution.py \\
        --merged-tsv ~/Downloads/merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv \\
        --enveda-jsonl ~/Downloads/enveda-180.jsonl.gz
"""
import argparse
import gzip
import json
from collections import Counter

import pandas as pd
from rdkit import Chem

from massspecgym.data.transforms import (
    MolToPFASVector,
    MolToIsolatedCF2Vector,
    MolToIsolatedCF3Vector,
    ion_mode_idx_from_adduct,
)
from massspecgym.utils import smiles_to_inchi_key

ION_MODE_NAMES = {0: "negative", 1: "positive", 2: "unknown"}


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def classify_molecule(smiles, pfas_chain_checker, cf2_checker, cf3_checker):
    if not smiles:
        return False, False, False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, False, False
    try:
        has_chain = bool(pfas_chain_checker.has_pf_chain_ge_2(smiles))
    except Exception:
        has_chain = False
    has_cf2_only = bool(cf2_checker._has_isolated_cf2_only(mol))
    has_cf3_only = bool(cf3_checker._has_isolated_cf3_only(mol))
    return has_chain, has_cf2_only, has_cf3_only


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-tsv", required=True, help="Path to merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv")
    parser.add_argument("--enveda-jsonl", required=True, help="Path to enveda-180(.filtered).jsonl(.gz)")
    parser.add_argument("--limit-enveda", type=int, default=None, help="Optional cap on Enveda-180 lines read (for dry runs)")
    args = parser.parse_args()

    pfas_chain_checker = MolToPFASVector()
    cf2_checker = MolToIsolatedCF2Vector()
    cf3_checker = MolToIsolatedCF3Vector()

    # ---- Load NIST-PFAS subset from the merged TSV (negative-ion-mode rows only) ----
    print("Loading merged TSV (adduct, smiles, inchikey columns only)...")
    df = pd.read_csv(args.merged_tsv, sep="\t", usecols=["adduct", "smiles", "inchikey"])
    df["ion_mode_idx"] = df["adduct"].apply(ion_mode_idx_from_adduct)
    nist_pfas = df[df["ion_mode_idx"] == 0].copy()  # negative mode == NIST-PFAS, confirmed in section 10
    print(f"NIST-PFAS spectra (negative-mode rows in merged TSV): {len(nist_pfas)}")

    # NOTE: the merged TSV's stored 'inchikey' column is entirely NaN for
    # this subset (checked directly) -- compute it from SMILES instead,
    # same helper the rest of this codebase uses (massspecgym.utils via
    # MassSpecDataset.compute_mol_freq) when inchikey isn't available.
    unique_smiles = nist_pfas["smiles"].dropna().unique()
    smiles_to_ik = {s: smiles_to_inchi_key(s) for s in unique_smiles}
    nist_pfas["inchikey"] = nist_pfas["smiles"].map(smiles_to_ik)

    nist_pfas_unique = {}
    for _, row in nist_pfas.drop_duplicates(subset=["inchikey"]).iterrows():
        nist_pfas_unique[row["inchikey"]] = row["smiles"]
    print(f"NIST-PFAS unique molecules (by computed inchikey): {len(nist_pfas_unique)}")

    # ---- Load Enveda-180 (unique molecules + per-spectrum ion mode) ----
    print("\nStreaming Enveda-180 JSONL...")
    enveda_unique = {}
    enveda_spectra = []  # (inchikey, ionmode_str)
    n_lines = 0
    with open_maybe_gzip(args.enveda_jsonl) as f:
        for line in f:
            if args.limit_enveda and n_lines >= args.limit_enveda:
                break
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            inchikey = rec.get("inchikey")
            smiles = rec.get("smiles")
            ionmode = rec.get("ionmode") or ""
            if inchikey and inchikey not in enveda_unique:
                enveda_unique[inchikey] = smiles
            enveda_spectra.append((inchikey, "positive" if "positive" in ionmode.lower()
                                    else ("negative" if "negative" in ionmode.lower() else "unknown")))
    print(f"Enveda-180 spectra read: {len(enveda_spectra)}, unique molecules: {len(enveda_unique)}")

    # ---- Classify all unique molecules (both sources) ----
    print("\nClassifying molecules (chain-PFAS / isolated-CF2 / isolated-CF3)...")
    mol_class = {}  # inchikey -> (has_chain, has_cf2_only, has_cf3_only)
    overlap = set(nist_pfas_unique) & set(enveda_unique)
    if overlap:
        print(f"NOTE: {len(overlap)} inchikeys appear in BOTH NIST-PFAS and Enveda-180 -- counted once.")

    all_unique = {**enveda_unique, **nist_pfas_unique}  # nist_pfas wins on overlap (rare/none expected)
    for inchikey, smiles in all_unique.items():
        mol_class[inchikey] = classify_molecule(smiles, pfas_chain_checker, cf2_checker, cf3_checker)

    n_mols = len(all_unique)
    n_chain = sum(1 for v in mol_class.values() if v[0])
    n_cf2 = sum(1 for v in mol_class.values() if v[1])
    n_cf3 = sum(1 for v in mol_class.values() if v[2])
    n_drug_like = sum(1 for v in mol_class.values() if v[1] or v[2])
    n_any_pfas = sum(1 for v in mol_class.values() if v[0] or v[1] or v[2])

    print(f"\n=== Combined unique-molecule counts (NIST-PFAS + Enveda-180, n={n_mols}) ===")
    print(f"  Chain-PFAS (>=2 connected CF2/CF3):     {n_chain:8d}  ({100*n_chain/n_mols:.4f}%)")
    print(f"  Isolated-CF2-only:                      {n_cf2:8d}  ({100*n_cf2/n_mols:.4f}%)")
    print(f"  Isolated-CF3-only:                      {n_cf3:8d}  ({100*n_cf3/n_mols:.4f}%)")
    print(f"  Drug-like PFAS (CF2-only OR CF3-only):   {n_drug_like:8d}  ({100*n_drug_like/n_mols:.4f}%)")
    print(f"  TOTAL any-PFAS (chain OR drug-like):     {n_any_pfas:8d}  ({100*n_any_pfas/n_mols:.4f}%)")

    # ---- Build combined per-spectrum table: (source, ion_mode, chain, cf2, cf3, drug_like) ----
    combined_spectra = []
    for _, row in nist_pfas.iterrows():
        combined_spectra.append(("NIST-PFAS", "negative", row["inchikey"]))
    for inchikey, mode in enveda_spectra:
        combined_spectra.append(("Enveda-180", mode, inchikey))

    print(f"\n=== Spectrum-level distribution, cut by ion mode (n={len(combined_spectra)} spectra total) ===")
    mode_counts = Counter()
    mode_chain = Counter()
    mode_cf2 = Counter()
    mode_cf3 = Counter()
    mode_drug_like = Counter()
    mode_any = Counter()
    mode_source_counts = Counter()

    for source, mode, inchikey in combined_spectra:
        mode_counts[mode] += 1
        mode_source_counts[(mode, source)] += 1
        has_chain, has_cf2, has_cf3 = mol_class.get(inchikey, (False, False, False))
        if has_chain:
            mode_chain[mode] += 1
        if has_cf2:
            mode_cf2[mode] += 1
        if has_cf3:
            mode_cf3[mode] += 1
        if has_cf2 or has_cf3:
            mode_drug_like[mode] += 1
        if has_chain or has_cf2 or has_cf3:
            mode_any[mode] += 1

    for mode in ["negative", "positive", "unknown"]:
        n = mode_counts.get(mode, 0)
        if n == 0:
            continue
        print(f"\n--- Ion mode: {mode} (n={n} spectra) ---")
        for src in ["NIST-PFAS", "Enveda-180"]:
            src_n = mode_source_counts.get((mode, src), 0)
            if src_n:
                print(f"    from {src}: {src_n}")
        chain_n = mode_chain.get(mode, 0)
        cf2_n = mode_cf2.get(mode, 0)
        cf3_n = mode_cf3.get(mode, 0)
        drug_n = mode_drug_like.get(mode, 0)
        any_n = mode_any.get(mode, 0)
        print(f"    Chain-PFAS:            {chain_n:8d}  ({100*chain_n/n:.4f}%)")
        print(f"    Isolated-CF2-only:     {cf2_n:8d}  ({100*cf2_n/n:.4f}%)")
        print(f"    Isolated-CF3-only:     {cf3_n:8d}  ({100*cf3_n/n:.4f}%)")
        print(f"    Drug-like (CF2 or CF3):{drug_n:8d}  ({100*drug_n/n:.4f}%)")
        print(f"    TOTAL any-PFAS:        {any_n:8d}  ({100*any_n/n:.4f}%)")


if __name__ == "__main__":
    main()
