"""
Combines NIST-PFAS (negative-ion-mode subset of the existing merged TSV,
excluding NIST20 and MassSpecGym) with the full Enveda-180 dataset
(https://zenodo.org/records/20436851) into a single TSV, in the same
column schema as merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv
plus a few precomputed bonus label columns, so it can be pointed at
directly by the existing DreaMS training scripts (pth=...) with no
dataset/model code changes.

NIST-PFAS is identified as the negative-ion-mode subset of the merged TSV
(confirmed in dreams-pfas-workshop-paper.md section 10/14: the 24,391
negative-mode rows match NIST-PFAS's exact total from the ISEF paper's
Table 2, with zero crossover from MassSpecGym/NIST20). Its stored
'inchikey' column is entirely NaN for this subset -- recomputed from
SMILES here (same fix applied in
scripts/check_combined_nistpfas_enveda180_distribution.py).

Fold assignment: NIST-PFAS's ORIGINAL train/val fold (from the source
merged TSV) is preserved as-is (100 train / 3 val unique molecules,
23,471 / 920 spectra) -- an earlier attempt to run a fresh split across
the whole combined pool (both stratified-random and Murcko Histogram
Split) caused chain-PFAS to nearly vanish from train (down to 0.02%
train / 99.98% val at the spectrum level), since NIST-PFAS's mostly-
acyclic chain-PFAS molecules collapse into a single Murcko histogram
group that lands entirely in one fold. Only Enveda-180's ~184,330
molecules (no prior split to preserve) get a fresh **Murcko Histogram
Split** (https://dreams-docs.readthedocs.io/en/latest/tutorials/murcko_hist_split.html),
reusing dreams.algorithms.murcko_hist.murcko_hist directly (already
installed as a dependency of the `dreams` package -- not reimplemented).

Usage:
    python scripts/prepare_combined_nistpfas_enveda180_dataset.py \\
        --merged-tsv ~/Downloads/merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv \\
        --enveda-jsonl ~/Downloads/enveda-180.jsonl.gz \\
        --output combined_nistpfas_enveda180_with_fold.tsv \\
        --val-mols-frac 0.15
"""
import argparse
import gzip
import json
import re
from collections import defaultdict

import pandas as pd
from rdkit import Chem

from dreams.algorithms.murcko_hist.murcko_hist import murcko_hist, are_sub_hists

from massspecgym.data.transforms import (
    MolToPFASVector,
    MolToIsolatedCF2Vector,
    MolToIsolatedCF3Vector,
    ion_mode_idx_from_adduct,
)
from massspecgym.utils import smiles_to_inchi_key

TARGET_COLUMNS = [
    "Unnamed: 0.1", "Unnamed: 0", "identifier", "mzs", "intensities", "smiles",
    "inchikey", "formula", "precursor_formula", "parent_mass", "precursor_mz",
    "adduct", "instrument_type", "collision_energy", "fold",
    "simulation_challenge", "name", "is_PFAS",
    # bonus columns, not in the original schema (see plan/workshop doc for precedent)
    "ion_mode_true", "has_chain_pfas", "has_isolated_cf2", "has_isolated_cf3",
]


def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def has_F(smiles):
    """Same convention already used in TestMassSpecDataset/IonModeMassSpecDataset's item['F']."""
    return bool(re.search("f", str(smiles), re.IGNORECASE))


def classify_smiles(smiles, pfas_chain_checker, cf2_checker, cf3_checker):
    if not smiles:
        return False, False, False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, False, False
    try:
        has_chain = bool(pfas_chain_checker.has_pf_chain_ge_2(smiles))
    except Exception:
        has_chain = False
    has_cf2 = bool(cf2_checker._has_isolated_cf2_only(mol))
    has_cf3 = bool(cf3_checker._has_isolated_cf3_only(mol))
    return has_chain, has_cf2, has_cf3


def category(info):
    """Mutually-exclusive priority bucket: chain > drug-like isolated > other-F > non-F."""
    if info["has_chain"]:
        return "chain_pfas"
    if info["has_cf2"] or info["has_cf3"]:
        return "drug_like_isolated"
    if info["has_F"]:
        return "other_fluorinated"
    return "non_fluorinated"


def assign_folds(mol_info: dict, val_mols_frac: float = 0.15) -> dict:
    """
    Real Murcko Histogram Split, transcribed from
    https://dreams-docs.readthedocs.io/en/latest/tutorials/murcko_hist_split.html
    (algorithm only -- the underlying murcko_hist/are_sub_hists functions
    are imported directly from the installed `dreams` package, not
    reimplemented).

    mol_info: inchikey -> {"smiles": str, ...}. Returns inchikey -> "train" | "val".

    Steps: compute each unique molecule's Murcko histogram; group molecules
    by histogram, sorted descending by group size; starting from the
    median-frequency group and walking toward the most-frequent group,
    assign each group to val unless it's structurally close (are_sub_hists,
    k=3, d=4) to an already-assigned val group, until cumulative val
    molecule count exceeds val_mols_frac; everything less frequent than the
    median is bulk-assigned to train (not eligible for val under this
    algorithm). NOTE: this targets ~val_mols_frac of MOLECULE count via
    whole-histogram-group increments, not an exact ratio -- the tutorial's
    own MassSpecGym example landed at 80.55/19.45 for a 0.15 target.
    """
    inchikeys = list(mol_info.keys())
    n = len(inchikeys)
    print(f"[assign_folds] Computing Murcko histograms for {n} molecules...")

    hist_by_ik = {}
    n_errors = 0
    for i, ik in enumerate(inchikeys):
        if i % 20000 == 0:
            print(f"  ... {i}/{n}")
        smiles = mol_info[ik]["smiles"]
        h = {}
        try:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is not None:
                h = murcko_hist(mol)
        except Exception:
            n_errors += 1
        hist_by_ik[ik] = h
    if n_errors:
        print(f"[assign_folds] {n_errors} molecules failed Murcko histogram "
              f"computation (treated as empty histogram)")

    groups = defaultdict(list)
    for ik in inchikeys:
        groups[str(hist_by_ik[ik])].append(ik)
    group_list = sorted(groups.items(), key=lambda kv: -len(kv[1]))  # (hist_str, [inchikeys])
    group_hist = {hs: hist_by_ik[iks[0]] for hs, iks in group_list}
    n_groups = len(group_list)
    print(f"[assign_folds] {n} molecules -> {n_groups} distinct Murcko histograms")

    median_i = n_groups // 2
    cum_val_mols = 0
    val_idx, train_idx = [], []

    for i in range(median_i, -1, -1):
        current_hist = group_hist[group_list[i][0]]
        is_val_subhist = any(
            are_sub_hists(current_hist, group_hist[group_list[j][0]], k=3, d=4)
            for j in val_idx
        )
        if is_val_subhist:
            train_idx.append(i)
        else:
            if cum_val_mols / n <= val_mols_frac:
                cum_val_mols += len(group_list[i][1])
                val_idx.append(i)
            else:
                train_idx.append(i)

    train_idx.extend(range(median_i + 1, n_groups))
    assert len(train_idx) + len(val_idx) == n_groups

    fold_map = {}
    for i in val_idx:
        for ik in group_list[i][1]:
            fold_map[ik] = "val"
    for i in train_idx:
        for ik in group_list[i][1]:
            fold_map[ik] = "train"
    return fold_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-tsv", required=True)
    parser.add_argument("--enveda-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-mols-frac", type=float, default=0.15,
                         help="Target fraction of unique molecules in val for the Murcko Histogram Split (tutorial default 0.15; actual ratio will differ slightly, see script docstring)")
    parser.add_argument("--limit-enveda", type=int, default=None, help="Optional cap on Enveda-180 lines read (for dry runs)")
    args = parser.parse_args()

    pfas_chain_checker = MolToPFASVector()
    cf2_checker = MolToIsolatedCF2Vector()
    cf3_checker = MolToIsolatedCF3Vector()

    # ---- Load NIST-PFAS subset from the merged TSV ----
    print("Loading NIST-PFAS subset from merged TSV...")
    merged_cols = ["identifier", "mzs", "intensities", "smiles", "adduct",
                   "precursor_mz", "formula", "precursor_formula",
                   "instrument_type", "collision_energy", "fold"]
    df = pd.read_csv(args.merged_tsv, sep="\t", usecols=merged_cols)
    df["ion_mode_idx"] = df["adduct"].apply(ion_mode_idx_from_adduct)
    nist_pfas = df[df["ion_mode_idx"] == 0].copy()  # negative mode == NIST-PFAS, confirmed in section 10/14
    print(f"NIST-PFAS spectra: {len(nist_pfas)}")

    unique_smiles = nist_pfas["smiles"].dropna().unique()
    smiles_to_ik = {s: smiles_to_inchi_key(s) for s in unique_smiles}
    nist_pfas["inchikey"] = nist_pfas["smiles"].map(smiles_to_ik)
    print(f"NIST-PFAS unique molecules: {nist_pfas['inchikey'].nunique()}")

    # Preserve NIST-PFAS's ORIGINAL fold as-is (see docstring: a fresh split
    # here collapses chain-PFAS almost entirely into one fold).
    nist_pfas_fold_map = dict(zip(nist_pfas["inchikey"], nist_pfas["fold"]))
    print(f"NIST-PFAS original fold (preserved): "
          f"{sum(1 for f in nist_pfas_fold_map.values() if f == 'train')} train / "
          f"{sum(1 for f in nist_pfas_fold_map.values() if f == 'val')} val unique molecules")

    # ---- Pass 1: stream Enveda-180 for unique-molecule info ----
    print("\nPass 1: streaming Enveda-180 for unique molecule info...")
    enveda_mol_smiles = {}
    enveda_mol_modes = {}  # inchikey -> {"has_pos": bool, "has_neg": bool}
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
            if not inchikey:
                continue
            if inchikey not in enveda_mol_smiles:
                enveda_mol_smiles[inchikey] = rec.get("smiles")
                enveda_mol_modes[inchikey] = {"has_pos": False, "has_neg": False}
            ionmode = (rec.get("ionmode") or "").lower()
            if "positive" in ionmode:
                enveda_mol_modes[inchikey]["has_pos"] = True
            elif "negative" in ionmode:
                enveda_mol_modes[inchikey]["has_neg"] = True
    print(f"Enveda-180 spectra: {n_lines}, unique molecules: {len(enveda_mol_smiles)}")

    # ---- Classify all unique molecules across both sources ----
    print("\nClassifying all unique molecules (chain-PFAS / isolated-CF2 / isolated-CF3)...")
    mol_info = {}
    for inchikey, smiles in enveda_mol_smiles.items():
        has_chain, has_cf2, has_cf3 = classify_smiles(smiles, pfas_chain_checker, cf2_checker, cf3_checker)
        modes = enveda_mol_modes[inchikey]
        mol_info[inchikey] = {
            "smiles": smiles, "has_F": has_F(smiles),
            "has_chain": has_chain, "has_cf2": has_cf2, "has_cf3": has_cf3,
            "has_pos": modes["has_pos"], "has_neg": modes["has_neg"],
        }

    nist_pfas_unique_smiles = dict(zip(nist_pfas["inchikey"], nist_pfas["smiles"]))
    n_overlap = 0
    for inchikey, smiles in nist_pfas_unique_smiles.items():
        if inchikey in mol_info:
            n_overlap += 1
            mol_info[inchikey]["has_neg"] = True  # merge mode coverage on overlap
            continue
        has_chain, has_cf2, has_cf3 = classify_smiles(smiles, pfas_chain_checker, cf2_checker, cf3_checker)
        mol_info[inchikey] = {
            "smiles": smiles, "has_F": has_F(smiles),
            "has_chain": has_chain, "has_cf2": has_cf2, "has_cf3": has_cf3,
            "has_pos": False, "has_neg": True,  # NIST-PFAS is negative-mode by construction
        }
    if n_overlap:
        print(f"NOTE: {n_overlap} inchikeys appear in BOTH NIST-PFAS and Enveda-180.")

    n_mols = len(mol_info)
    print(f"Combined unique molecules: {n_mols}")

    # Only run the Murcko Histogram Split on molecules NOT covered by
    # NIST-PFAS's preserved original fold (see docstring).
    mol_info_for_murcko = {ik: info for ik, info in mol_info.items() if ik not in nist_pfas_fold_map}
    print(f"\nRunning Murcko Histogram Split on {len(mol_info_for_murcko)} Enveda-180-only "
          f"molecules ({len(nist_pfas_fold_map)} NIST-PFAS molecules keep their preserved fold)...")
    fold_map = assign_folds(mol_info_for_murcko, args.val_mols_frac)
    fold_map.update(nist_pfas_fold_map)  # NIST-PFAS's preserved fold takes precedence

    # ---- Pass 2: write output TSV ----
    print(f"\nWriting combined TSV to {args.output} ...")
    n_nist_written = 0
    n_enveda_written = 0
    spectrum_counts = defaultdict(int)  # (category, fold) -> spectrum count
    fallback_info = {"has_chain": False, "has_cf2": False, "has_cf3": False, "has_F": False}
    with open(args.output, "w") as out:
        out.write("\t".join(TARGET_COLUMNS) + "\n")

        # NIST-PFAS rows (already fully loaded in memory)
        for _, row in nist_pfas.iterrows():
            inchikey = row["inchikey"]
            info = mol_info.get(inchikey, fallback_info)
            fold = fold_map.get(inchikey, "train")
            spectrum_counts[(category(info), fold)] += 1
            out_row = {
                "Unnamed: 0.1": "", "Unnamed: 0": "",
                "identifier": row["identifier"],
                "mzs": row["mzs"], "intensities": row["intensities"],
                "smiles": row["smiles"], "inchikey": inchikey,
                "formula": row.get("formula", ""), "precursor_formula": row.get("precursor_formula", ""),
                "parent_mass": "", "precursor_mz": row["precursor_mz"], "adduct": row["adduct"],
                "instrument_type": row.get("instrument_type", ""), "collision_energy": row.get("collision_energy", ""),
                "fold": fold, "simulation_challenge": "", "name": "",
                "is_PFAS": int(info["has_chain"]),
                "ion_mode_true": "ESI Negative",
                "has_chain_pfas": int(info["has_chain"]),
                "has_isolated_cf2": int(info["has_cf2"]),
                "has_isolated_cf3": int(info["has_cf3"]),
            }
            out.write("\t".join(str(out_row[c]) for c in TARGET_COLUMNS) + "\n")
            n_nist_written += 1

        # Enveda-180 rows (second streaming pass over the JSONL)
        n_lines2 = 0
        with open_maybe_gzip(args.enveda_jsonl) as f:
            for line in f:
                if args.limit_enveda and n_lines2 >= args.limit_enveda:
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
                info = mol_info.get(inchikey, fallback_info)
                fold = fold_map.get(inchikey, "train")
                spectrum_counts[(category(info), fold)] += 1
                out_row = {
                    "Unnamed: 0.1": "", "Unnamed: 0": "",
                    "identifier": rec.get("title", ""),
                    "mzs": ",".join(rec.get("mzs", [])),
                    "intensities": ",".join(rec.get("intensities", [])),
                    "smiles": rec.get("smiles", ""), "inchikey": inchikey,
                    "formula": rec.get("formula", ""), "precursor_formula": rec.get("adduct_formula", ""),
                    "parent_mass": "", "precursor_mz": rec.get("pepmass", ""), "adduct": rec.get("adduct", ""),
                    "instrument_type": rec.get("instrument_type", ""), "collision_energy": rec.get("collision_energies", ""),
                    "fold": fold, "simulation_challenge": "", "name": "",
                    "is_PFAS": int(info["has_chain"]),
                    "ion_mode_true": rec.get("ionmode", ""),
                    "has_chain_pfas": int(info["has_chain"]),
                    "has_isolated_cf2": int(info["has_cf2"]),
                    "has_isolated_cf3": int(info["has_cf3"]),
                }
                out.write("\t".join(str(out_row[c]) for c in TARGET_COLUMNS) + "\n")
                n_enveda_written += 1

    n_written = n_nist_written + n_enveda_written
    print(f"\nWrote {n_written} total spectra ({n_nist_written} NIST-PFAS + {n_enveda_written} Enveda-180) to {args.output}")

    n_chain = sum(1 for v in mol_info.values() if v["has_chain"])
    n_cf2 = sum(1 for v in mol_info.values() if v["has_cf2"])
    n_cf3 = sum(1 for v in mol_info.values() if v["has_cf3"])
    n_drug_like = sum(1 for v in mol_info.values() if v["has_cf2"] or v["has_cf3"])
    print(f"Combined unique molecules: {n_mols}")
    print(f"  chain-PFAS: {n_chain}  isolated-CF2: {n_cf2}  isolated-CF3: {n_cf3}  "
          f"drug-like(either): {n_drug_like}")

    n_val_mols = sum(1 for f in fold_map.values() if f == "val")
    print(f"\nMolecule-level fold split: {n_mols - n_val_mols} train ({100*(n_mols-n_val_mols)/n_mols:.2f}%) / "
          f"{n_val_mols} val ({100*n_val_mols/n_mols:.2f}%)  [tutorial reference: 80.55% / 19.45%]")

    print("\n=== Spectrum-level category representation, train vs val ===")
    any_zero_val = False
    for cat in ["chain_pfas", "drug_like_isolated", "other_fluorinated", "non_fluorinated"]:
        n_train = spectrum_counts[(cat, "train")]
        n_val = spectrum_counts[(cat, "val")]
        total = n_train + n_val
        if total == 0:
            print(f"  {cat:20s}: no spectra in this dataset")
            continue
        pct_train = 100 * n_train / total
        pct_val = 100 * n_val / total
        flag = ""
        if n_val == 0:
            any_zero_val = True
            flag = "  <-- ZERO VAL REPRESENTATION"
        print(f"  {cat:20s}: train={n_train:8d} ({pct_train:5.2f}%)  val={n_val:7d} ({pct_val:5.2f}%){flag}")
    if any_zero_val:
        print("\n[WARNING] At least one category has zero validation-set spectra -- "
              "see the acyclic-Murcko-scaffold risk noted in the script docstring/plan.")


if __name__ == "__main__":
    main()
