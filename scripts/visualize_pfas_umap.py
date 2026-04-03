"""
Visualize fine-tuned DreaMS embeddings with UMAP.

One plot panel (or two if --finetuned_ckpt supplied) shows:
  • Non-PFAS spectra as a light gray background cloud
  • PFAS spectra colored by high-level chemical subtype:
      PFCAs            — perfluoroalkyl carboxylic acids
      PFSAs            — perfluoroalkyl sulfonic acids
      Fluorotelomers   — mixed H/F chain compounds
      PFAS Sulfonamides— sulfonamide-containing PFAS
      Other PFAS       — remaining PFAS

Usage:
    # Pretrained only:
    python scripts/visualize_pfas_umap.py \
        --data ~/Downloads/merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv \
        --out results/plots/pfas_umap.png

    # Pretrained + fine-tuned side-by-side:
    python scripts/visualize_pfas_umap.py \
        --data ~/Downloads/merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv \
        --finetuned_ckpt ~/Downloads/HalogenDetection-FocalLoss-...ckpt \
        --out results/plots/pfas_umap_before_after.png
"""

import argparse
from collections import Counter
from pathlib import Path

import matchms
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset

from massspecgym.data.transforms import SpecTokenizer
from dreams.api import PreTrainedModel
from dreams.models.dreams.dreams import DreaMS as DreaMSModel


# ── High-level PFAS subtype classification ────────────────────────────────────
# Priority-ordered: first match wins.
# Each tuple: (label, SMARTS)

_SUBTYPE_SMARTS = [
    # PFSAs: sulfonic acid + at least one CF2
    ("PFSAs",              "C(F)(F)S(=O)(=O)[OH,O-,OCC]"),
    # PFCAs: carboxylic acid + at least one CF2
    ("PFCAs",              "C(F)(F)C(=O)[OH,O-,OCC,N]"),
    # Sulfonamides: S(=O)(=O)N — covers FOSAs, FOSEs
    ("PFAS Sulfonamides",  "S(=O)(=O)N"),
    # Fluorotelomers: CH2 directly bonded to CF2 (mixed H/F chain)
    ("Fluorotelomers",     "[CH2][CX4](F)F"),
]
_COMPILED = [(name, Chem.MolFromSmarts(s)) for name, s in _SUBTYPE_SMARTS]

NON_PFAS_LABEL = "Non-PFAS"
OTHER_PFAS_LABEL = "Other PFAS"

def classify_pfas_subtype(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
    if mol is None:
        return OTHER_PFAS_LABEL
    for name, pat in _COMPILED:
        if pat is not None and mol.HasSubstructMatch(pat):
            return name
    return OTHER_PFAS_LABEL


# ── Color / style config ──────────────────────────────────────────────────────

LABEL_COLORS = {
    NON_PFAS_LABEL:       "#CCCCCC",   # light gray background
    "PFCAs":              "#C0392B",   # brick red
    "PFSAs":              "#2166AC",   # steel blue
    "Fluorotelomers":     "#27AE60",   # green
    "PFAS Sulfonamides":  "#E67E22",   # orange
    OTHER_PFAS_LABEL:     "#8E44AD",   # purple
}

LABEL_SIZES = {
    NON_PFAS_LABEL:       3,
    "PFCAs":              25,
    "PFSAs":              25,
    "Fluorotelomers":     25,
    "PFAS Sulfonamides":  25,
    OTHER_PFAS_LABEL:     20,
}

LABEL_ALPHA = {
    NON_PFAS_LABEL:       0.15,
    "PFCAs":              0.90,
    "PFSAs":              0.90,
    "Fluorotelomers":     0.90,
    "PFAS Sulfonamides":  0.90,
    OTHER_PFAS_LABEL:     0.80,
}

# Non-PFAS drawn first (background), then PFAS subtypes on top
LABEL_ORDER = [NON_PFAS_LABEL, "PFCAs", "PFSAs", "Fluorotelomers", "PFAS Sulfonamides"]


# ── Dataset ───────────────────────────────────────────────────────────────────

class RawSpecDataset(Dataset):
    def __init__(self, df: pd.DataFrame, n_peaks: int = 60):
        self.df = df.reset_index(drop=True)
        self.tokenizer = SpecTokenizer(n_peaks=n_peaks)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        mzs  = np.array([float(x) for x in str(row["mzs"]).split(",")])
        ints = np.array([float(x) for x in str(row["intensities"]).split(",")])
        spec = matchms.Spectrum(
            mz=mzs,
            intensities=ints,
            metadata={"precursor_mz": float(row["precursor_mz"])},
        )
        try:
            token = self.tokenizer(spec)
        except Exception:
            token = torch.zeros(61, 2)
        return token.float()


# ── Embedding extraction ──────────────────────────────────────────────────────

PRETRAINED_URL = "https://zenodo.org/records/10997887/files/ssl_model.ckpt?download=1"


def load_pretrained_backbone() -> nn.Module:
    return PreTrainedModel.from_ckpt(
        ckpt_path=PRETRAINED_URL,
        ckpt_cls=DreaMSModel,
        n_highest_peaks=60,
    ).model.eval()


def load_finetuned_backbone(ckpt_path: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone_sd = {
        k[len("main_model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("main_model.")
    }
    backbone = load_pretrained_backbone()
    backbone.load_state_dict(backbone_sd, strict=False)
    return backbone.eval()


@torch.no_grad()
def extract_embeddings(backbone, dataset, batch_size=128,
                       device=torch.device("cpu")) -> np.ndarray:
    backbone = backbone.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    parts = []
    for batch in loader:
        emb = backbone(batch.to(device))[:, 0, :]
        parts.append(emb.cpu().float().numpy())
    return np.vstack(parts)


# ── Dimensionality reduction ──────────────────────────────────────────────────

def reduce_umap(embeddings: np.ndarray) -> np.ndarray:
    import umap
    return umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.1,
        metric="cosine", random_state=42, verbose=True,
    ).fit_transform(embeddings)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_panel(ax, coords: np.ndarray, labels: list, title: str, method: str = "UMAP"):
    import matplotlib.patches as mpatches

    label_arr = np.array(labels)
    present = set(labels)

    for lab in LABEL_ORDER:
        if lab not in present:
            continue
        mask = label_arr == lab
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=LABEL_COLORS[lab],
            s=LABEL_SIZES[lab],
            alpha=LABEL_ALPHA[lab],
            linewidths=0,
            rasterized=True,
            zorder=(1 if lab == NON_PFAS_LABEL else 2),
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel(f"{method} 1", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{method} 2", fontsize=11, fontweight="bold")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Legend: Non-PFAS first, then subtypes
    handles = [
        mpatches.Patch(color=LABEL_COLORS[lab], label=lab)
        for lab in LABEL_ORDER if lab in present
    ]
    ax.legend(handles=handles, fontsize=10, loc="lower left",
              framealpha=0.9, edgecolor="#cccccc")


def make_plot(coords_pre, labels, out_path, coords_ft=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 2 if coords_ft is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 7), dpi=150)
    if n_panels == 1:
        axes = [axes]

    _plot_panel(axes[0], coords_pre, labels, "Pretrained DreaMS")
    if coords_ft is not None:
        _plot_panel(axes[1], coords_ft, labels, "DreaMS-PFAS (Fine-Tuned)")

    n_pfas    = sum(1 for l in labels if l != NON_PFAS_LABEL)
    n_nonpfas = sum(1 for l in labels if l == NON_PFAS_LABEL)
    fig.suptitle(
        f"DreaMS Embeddings — UMAP  |  PFAS: {n_pfas:,}  |  Non-PFAS: {n_nonpfas:,}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="Path to merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv")
    p.add_argument("--finetuned_ckpt", default=None,
                   help="Fine-tuned checkpoint (.ckpt). If given, produces a two-panel plot.")
    p.add_argument("--out", default="results/plots/pfas_umap.png")
    p.add_argument("--n_pfas",    type=int, default=5000)
    p.add_argument("--n_nonpfas", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_peaks",    type=int, default=60)
    p.add_argument("--fold", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps")  if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    cols = ["identifier", "mzs", "intensities", "precursor_mz", "smiles", "is_PFAS", "fold"]
    df = pd.read_csv(args.data, sep="\t", usecols=cols, low_memory=False)
    df["is_PFAS"] = df["is_PFAS"].astype(str).str.lower() == "true"

    if args.fold != "all":
        df = df[df["fold"] == args.fold]

    # Sample Non-PFAS
    nonpfas_df = df[~df["is_PFAS"]].sample(
        n=min(args.n_nonpfas, (~df["is_PFAS"]).sum()), random_state=args.seed
    ).copy()
    nonpfas_df["subtype"] = NON_PFAS_LABEL

    # Classify PFAS by high-level subtype, sample proportionally
    pfas_all = df[df["is_PFAS"]].copy()
    pfas_all["subtype"] = pfas_all["smiles"].apply(classify_pfas_subtype)

    pfas_all = pfas_all[pfas_all["subtype"] != OTHER_PFAS_LABEL]

    counts = pfas_all["subtype"].value_counts()
    total  = len(pfas_all)
    pfas_parts = []
    for subtype, count in counts.items():
        n_take = max(1, round(args.n_pfas * count / total))
        n_take = min(n_take, count)
        pfas_parts.append(
            pfas_all[pfas_all["subtype"] == subtype].sample(n=n_take, random_state=args.seed)
        )
    pfas_df = pd.concat(pfas_parts, ignore_index=True)

    sample_df = pd.concat([pfas_df, nonpfas_df], ignore_index=True)
    labels = sample_df["subtype"].tolist()
    print(f"Sample: {len(pfas_df):,} PFAS + {len(nonpfas_df):,} Non-PFAS = {len(sample_df):,}")
    print("Subtype distribution:", Counter(labels))

    # Embeddings
    dataset = RawSpecDataset(sample_df, n_peaks=args.n_peaks)

    print("\nLoading pretrained backbone...")
    backbone_pre = load_pretrained_backbone()
    print("Extracting pretrained embeddings...")
    emb_pre = extract_embeddings(backbone_pre, dataset, args.batch_size, device)

    emb_ft = None
    if args.finetuned_ckpt:
        print(f"\nLoading fine-tuned checkpoint: {args.finetuned_ckpt}")
        backbone_ft = load_finetuned_backbone(args.finetuned_ckpt)
        print("Extracting fine-tuned embeddings...")
        emb_ft = extract_embeddings(backbone_ft, dataset, args.batch_size, device)

    # UMAP
    print("\nRunning UMAP on pretrained embeddings...")
    coords_pre = reduce_umap(emb_pre)

    coords_ft = None
    if emb_ft is not None:
        print("Running UMAP on fine-tuned embeddings...")
        coords_ft = reduce_umap(emb_ft)

    # Plot
    make_plot(coords_pre, labels, args.out, coords_ft=coords_ft)

    # Save embeddings
    emb_out = Path(args.out).with_suffix(".npz")
    save_dict = {
        "emb_pretrained":   emb_pre,
        "coords_pretrained": coords_pre,
        "labels":           np.array(labels),
        "identifiers":      sample_df["identifier"].values,
    }
    if emb_ft is not None:
        save_dict["emb_finetuned"]   = emb_ft
        save_dict["coords_finetuned"] = coords_ft
    np.savez_compressed(emb_out, **save_dict)
    print(f"Saved embeddings to {emb_out}")


if __name__ == "__main__":
    main()
