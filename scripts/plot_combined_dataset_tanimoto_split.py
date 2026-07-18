"""
Validates the combined dataset's train/val split by plotting the
distribution of maximum Tanimoto similarity between each validation-set
molecule and the training set -- same check as the DreaMS Murcko
Histogram Split tutorial
(https://dreams-docs.readthedocs.io/en/latest/tutorials/murcko_hist_split.html,
Code Cell 10) and the ISEF paper's Figure 4. A low-similarity peak
confirms the split doesn't leak near-duplicate scaffolds between train
and val; a peak near 1.0 would indicate leakage.

NOTE: does NOT use dreams.utils.data.evaluate_split. Two problems found
with it at this dataset's scale: (1) a real bug -- its internal loop only
computes similarities for whichever fold name happens to be LAST in
df[fold_col].unique() (row order dependent), so depending on row order it
can silently return train-vs-train self-similarity (always 1.0) instead
of val-vs-train; (2) it computes each pairwise Tanimoto via a Python-level
loop (O(V*T) individual DataStructs.FingerprintSimilarity calls), which is
impractically slow at this dataset's scale (44,415 val x 140,018 train
unique molecules = ~6.2 billion comparisons -- did not finish in several
minutes even with 4 workers). This script reimplements the same
statistic using RDKit's vectorized DataStructs.BulkTanimotoSimilarity
(one call per val molecule against the whole train reference set, C++
vectorized instead of a Python loop), with optional train-side
subsampling for speed.

Usage:
    python scripts/plot_combined_dataset_tanimoto_split.py \\
        --input ~/Downloads/DreaMS-PFAS-Paper/combined_nistpfas_enveda180_with_fold.tsv \\
        --output tanimoto_split_quality.png \\
        --max-train-ref 20000
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from tqdm import tqdm

from dreams.utils.mols import morgan_fp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the combined TSV (needs smiles + fold columns)")
    parser.add_argument("--output", default="tanimoto_split_quality.png")
    parser.add_argument("--max-train-ref", type=int, default=20000,
                         help="Subsample the training reference set to this many unique molecules for speed "
                              "(0 = use all, but this can be extremely slow at large scale)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading {args.input} (smiles, fold columns only)...")
    df = pd.read_csv(args.input, sep="\t", usecols=["smiles", "fold"])
    df = df[df["fold"].isin(["train", "val"])].drop_duplicates(subset=["smiles"])
    train_smiles = df.loc[df["fold"] == "train", "smiles"].tolist()
    val_smiles = df.loc[df["fold"] == "val", "smiles"].tolist()
    print(f"Unique molecules: {len(train_smiles)} train / {len(val_smiles)} val")

    rng = np.random.default_rng(args.seed)
    if args.max_train_ref and len(train_smiles) > args.max_train_ref:
        idx = rng.choice(len(train_smiles), size=args.max_train_ref, replace=False)
        train_smiles = [train_smiles[i] for i in idx]
        print(f"Subsampled training reference set to {len(train_smiles)} molecules for speed.")

    print("\nComputing Morgan fingerprints...")
    train_fps = [morgan_fp(Chem.MolFromSmiles(s), as_numpy=False) for s in tqdm(train_smiles, desc="train")]
    val_fps = [morgan_fp(Chem.MolFromSmiles(s), as_numpy=False) for s in tqdm(val_smiles, desc="val")]

    print("\nComputing max Tanimoto similarity to train set for each val molecule "
          "(RDKit BulkTanimotoSimilarity, vectorized)...")
    max_sims = np.array([
        max(DataStructs.BulkTanimotoSimilarity(vfp, train_fps))
        for vfp in tqdm(val_fps, desc="val-vs-train")
    ])

    print(f"\nValidation molecules evaluated: {len(max_sims)}")
    print(pd.Series(max_sims).describe())

    plt.figure(figsize=(5, 4))
    plt.hist(max_sims, bins=100)
    plt.xlabel("Max Tanimoto similarity to training set")
    plt.ylabel("Num. validation set molecules")
    plt.title("Combined NIST-PFAS + Enveda-180: split quality")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved plot to {args.output}")

    n_high_sim = (max_sims >= 0.99).sum()
    print(f"\nValidation molecules with near-duplicate (>=0.99 Tanimoto) match in "
          f"train: {n_high_sim} ({100*n_high_sim/len(max_sims):.2f}%)")


if __name__ == "__main__":
    main()
