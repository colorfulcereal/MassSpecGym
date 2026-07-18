"""
EDA: prevalence of "drug-like PFAS" (isolated CF2/CF3 groups, not connected
to another PF-carbon) in the Enveda-180 dataset
(https://zenodo.org/records/20436851), per the OECD PFAS definition
(Wang et al. 2021, "A New OECD Definition for Per- and Polyfluoroalkyl
Substances", Environmental Science & Technology,
DOI 10.1021/acs.est.1c06896): any chemical containing at least one
saturated CF2 or CF3 moiety counts as PFAS -- a broader definition than
the chain-PFAS rule (>=2 connected CF2/CF3) already checked in
check_enveda180_pfas_prevalence.py.

Reuses the existing isolated-CF2/isolated-CF3 transforms already in this
repo (massspecgym/data/transforms.py), which implement this same OECD
taxonomy (see documentation/DreaMS-PFAS_training_and_evaluation.md).

Usage:
    python scripts/check_enveda180_drug_like_pfas.py --input ~/Downloads/enveda-180.jsonl.gz
"""
import argparse
import gzip
import json
from collections import Counter

from rdkit import Chem

from massspecgym.data.transforms import (
    MolToPFASVector,
    MolToIsolatedCF2Vector,
    MolToIsolatedCF3Vector,
)


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to enveda-180(-filtered).jsonl(.gz)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of lines read (for quick dry runs)")
    args = parser.parse_args()

    pfas_chain_checker = MolToPFASVector()
    cf2_checker = MolToIsolatedCF2Vector()
    cf3_checker = MolToIsolatedCF3Vector()

    # ---- Pass 1: collect unique molecules + per-spectrum (inchikey, ionmode) ----
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

    print(f"Parsed {n_lines} lines ({n_parse_errors} JSON parse errors)")
    print(f"Unique molecules (by inchikey): {len(unique_mols)}")

    # ---- Classify each unique molecule ----
    has_chain = {}
    has_cf2_only = {}
    has_cf3_only = {}
    n_errors = 0
    for inchikey, smiles in unique_mols.items():
        if not smiles:
            has_chain[inchikey] = has_cf2_only[inchikey] = has_cf3_only[inchikey] = False
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_errors += 1
            has_chain[inchikey] = has_cf2_only[inchikey] = has_cf3_only[inchikey] = False
            continue
        try:
            has_chain[inchikey] = bool(pfas_chain_checker.has_pf_chain_ge_2(smiles))
        except Exception:
            has_chain[inchikey] = False
        has_cf2_only[inchikey] = bool(cf2_checker._has_isolated_cf2_only(mol))
        has_cf3_only[inchikey] = bool(cf3_checker._has_isolated_cf3_only(mol))

    n_mols = len(unique_mols)
    n_chain = sum(has_chain.values())
    n_cf2 = sum(has_cf2_only.values())
    n_cf3 = sum(has_cf3_only.values())
    n_cf2_and_cf3 = sum(1 for ik in unique_mols if has_cf2_only[ik] and has_cf3_only[ik])
    n_drug_like = sum(1 for ik in unique_mols if has_cf2_only[ik] or has_cf3_only[ik])
    n_oecd_pfas = sum(1 for ik in unique_mols if has_chain[ik] or has_cf2_only[ik] or has_cf3_only[ik])

    print(f"\nSMILES parse errors (RDKit): {n_errors}")
    print(f"\n=== Unique-molecule counts (OECD PFAS definition, Wang et al. 2021) ===")
    print(f"  Chain-PFAS (>=2 connected CF2/CF3):        {n_chain:8d}  ({100*n_chain/n_mols:.4f}%)")
    print(f"  Isolated-CF2-only ('drug-like'):           {n_cf2:8d}  ({100*n_cf2/n_mols:.4f}%)")
    print(f"  Isolated-CF3-only ('drug-like'):           {n_cf3:8d}  ({100*n_cf3/n_mols:.4f}%)")
    print(f"  Both isolated-CF2-only AND CF3-only:       {n_cf2_and_cf3:8d}  ({100*n_cf2_and_cf3/n_mols:.4f}%)")
    print(f"  Drug-like PFAS (CF2-only OR CF3-only):     {n_drug_like:8d}  ({100*n_drug_like/n_mols:.4f}%)")
    print(f"  TOTAL OECD PFAS (chain OR drug-like):      {n_oecd_pfas:8d}  ({100*n_oecd_pfas/n_mols:.4f}%)")

    # ---- Spectrum-level breakdown by ion mode ----
    def spectrum_breakdown(label, membership_dict):
        counts = Counter()
        member_counts = Counter()
        for inchikey, ionmode in spectra:
            mode_key = ionmode if ionmode else "MISSING"
            counts[mode_key] += 1
            if membership_dict.get(inchikey, False):
                member_counts[mode_key] += 1
        print(f"\n=== Spectrum-level breakdown: {label} (n={len(spectra)} spectra) ===")
        for mode, count in counts.most_common():
            m = member_counts.get(mode, 0)
            pct = 100 * m / count if count else 0.0
            print(f"  {mode:15s}: n={count:8d}  {label}={m:6d} ({pct:.4f}%)")

    drug_like_dict = {ik: (has_cf2_only[ik] or has_cf3_only[ik]) for ik in unique_mols}
    spectrum_breakdown("drug-like-PFAS-positive", drug_like_dict)

    oecd_pfas_dict = {ik: (has_chain[ik] or has_cf2_only[ik] or has_cf3_only[ik]) for ik in unique_mols}
    spectrum_breakdown("OECD-PFAS-positive (any type)", oecd_pfas_dict)


if __name__ == "__main__":
    main()
