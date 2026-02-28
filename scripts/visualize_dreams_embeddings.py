"""
Visualize DreaMS embeddings with UMAP / t-SNE, colored by PFAS chemical subclass.

PFAS subclasses are assigned by priority-ordered SMARTS matching:
  Fluorotelomer phosphate   — [#8]P(=O)
  PFAS sulfonamide          — S(=O)(=O)N
  FT thioacrylate           — [CH2]=[C]C(=O)S
  FT acrylate/methacrylate  — [CH2]=[C]C(=O)O
  Nitroaromatic PFAS        — [N+](=O)[O-]
  PFAS betaine              — [NX4+]
  Other PFAS                — catch-all

Usage (on GPU environment):
    # Pretrained DreaMS only:
    python scripts/visualize_dreams_embeddings.py \
        --data /path/to/merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv \
        --out results/plots/umap_embeddings.png

    # Compare pretrained vs fine-tuned (two-panel):
    python scripts/visualize_dreams_embeddings.py \
        --data /path/to/merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv \
        --finetuned_ckpt /path/to/checkpoint.ckpt \
        --out results/plots/umap_before_after.png

    # Use t-SNE instead of UMAP:
    python scripts/visualize_dreams_embeddings.py --method tsne ...
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


# ── PFAS subclass classification ──────────────────────────────────────────────

# Priority-ordered: first match wins
_SUBCLASS_SMARTS = [
    ("Fluorotelomer phosphate",  "[#8]P(=O)"),
    ("PFAS sulfonamide",         "S(=O)(=O)N"),
    ("FT thioacrylate",          "[CH2]=[C]C(=O)S"),
    ("FT acrylate/methacrylate", "[CH2]=[C]C(=O)O"),
    ("Nitroaromatic PFAS",       "[N+](=O)[O-]"),
    ("PFAS betaine",             "[NX4+]"),
]
_COMPILED = [(name, Chem.MolFromSmarts(s)) for name, s in _SUBCLASS_SMARTS]

NON_PFAS_LABEL = "Non-PFAS"

def classify_pfas_subclass(smiles: str) -> str:
    """Assign a PFAS chemical subclass via priority-ordered SMARTS matching."""
    mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
    if mol is None:
        return "Other PFAS"
    for name, pat in _COMPILED:
        if mol.HasSubstructMatch(pat):
            return name
    return "Other PFAS"


# ── Plotting config ───────────────────────────────────────────────────────────

LABEL_COLORS = {
    NON_PFAS_LABEL:              "#cccccc",
    "FT acrylate/methacrylate":  "#e63946",
    "Nitroaromatic PFAS":        "#f4a261",
    "PFAS sulfonamide":          "#2a9d8f",
    "Fluorotelomer phosphate":   "#457b9d",
    "FT thioacrylate":           "#a8dadc",
    "PFAS betaine":              "#6a4c93",
    "Other PFAS":                "#f1c40f",
}
LABEL_SIZES = {k: (4 if k == NON_PFAS_LABEL else 20) for k in LABEL_COLORS}
LABEL_ALPHA = {k: (0.20 if k == NON_PFAS_LABEL else 0.88) for k in LABEL_COLORS}

# Non-PFAS drawn first so PFAS points sit on top
LABEL_ORDER = [NON_PFAS_LABEL] + [k for k in LABEL_COLORS if k != NON_PFAS_LABEL]


# ── Dataset ───────────────────────────────────────────────────────────────────

class RawSpecDataset(Dataset):
    """
    Reconstructs matchms Spectrum objects from raw comma-separated mzs/intensities
    strings in the TSV and applies SpecTokenizer on the fly.
    """

    def __init__(self, df: pd.DataFrame, n_peaks: int = 60):
        self.df = df.reset_index(drop=True)
        self.tokenizer = SpecTokenizer(n_peaks=n_peaks)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        mzs = np.array([float(x) for x in str(row["mzs"]).split(",")])
        ints = np.array([float(x) for x in str(row["intensities"]).split(",")])
        precursor_mz = float(row["precursor_mz"])

        spec = matchms.Spectrum(
            mz=mzs,
            intensities=ints,
            metadata={"precursor_mz": precursor_mz},
        )
        try:
            token = self.tokenizer(spec)  # [n_peaks+1, 2]
        except Exception:
            token = torch.zeros(61, 2)

        return token.float()


# ── Embedding extraction ──────────────────────────────────────────────────────

PRETRAINED_URL = "https://zenodo.org/records/10997887/files/ssl_model.ckpt?download=1"


def load_pretrained_backbone() -> nn.Module:
    model = PreTrainedModel.from_ckpt(
        ckpt_path=PRETRAINED_URL,
        ckpt_cls=DreaMSModel,
        n_highest_peaks=60,
    ).model.eval()
    return model


def load_finetuned_backbone(ckpt_path: str) -> nn.Module:
    """
    Load fine-tuned DreaMS backbone from a Lightning checkpoint without
    importing the training script (which runs training at module level).

    Checkpoint keys:
      main_model.*  — DreaMS backbone (kept)
      lin_out.*     — classification head (discarded)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone_sd = {
        k[len("main_model."):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("main_model.")
    }
    backbone = load_pretrained_backbone()
    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:5]}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}")
    return backbone.eval()


@torch.no_grad()
def extract_embeddings(
    backbone: nn.Module,
    dataset: RawSpecDataset,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Forward pass → CLS token embeddings [N, 1024]."""
    backbone = backbone.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_embs = []
    for batch in loader:
        emb = backbone(batch.to(device))[:, 0, :]   # [B, 1024]
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)


# ── Dimensionality reduction ──────────────────────────────────────────────────

def reduce_embeddings(embeddings: np.ndarray, method: str = "umap") -> np.ndarray:
    if method == "umap":
        import umap
        return umap.UMAP(
            n_components=2, n_neighbors=30, min_dist=0.1,
            metric="cosine", random_state=42, verbose=True,
        ).fit_transform(embeddings)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(
            n_components=2, perplexity=30, n_iter=1000,
            metric="cosine", random_state=42, verbose=1,
        ).fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'umap' or 'tsne'.")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_single(ax, coords: np.ndarray, labels: list, title: str, method: str):
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
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(f"{method.upper()} 1", fontsize=10)
    ax.set_ylabel(f"{method.upper()} 2", fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    handles = [
        mpatches.Patch(color=LABEL_COLORS[lab], label=lab)
        for lab in LABEL_ORDER if lab in present
    ]
    ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.85)


def make_plot(
    coords_pretrained: np.ndarray,
    labels: list,
    method: str,
    out_path: str,
    coords_finetuned: np.ndarray = None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 2 if coords_finetuned is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7), dpi=150)
    if n_panels == 1:
        axes = [axes]

    _plot_single(axes[0], coords_pretrained, labels, "Pretrained DreaMS", method)
    if coords_finetuned is not None:
        _plot_single(axes[1], coords_finetuned, labels, "Fine-tuned DreaMS", method)

    n_pfas = sum(1 for l in labels if l != NON_PFAS_LABEL)
    fig.suptitle(
        f"DreaMS Embeddings — {method.upper()}  |  "
        f"n={len(labels):,}  ({n_pfas:,} PFAS by chemical subclass)",
        fontsize=14, y=1.01,
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
                   help="Optional path to fine-tuned checkpoint (.ckpt). "
                        "If provided, a two-panel before/after plot is produced.")
    p.add_argument("--out", default="results/plots/umap_embeddings.png")
    p.add_argument("--method", choices=["umap", "tsne"], default="umap")
    p.add_argument("--n_pfas", type=int, default=5000,
                   help="Total PFAS spectra to sample, drawn proportionally per subclass "
                        "(default 5000, matching n_nonpfas for a balanced plot)")
    p.add_argument("--n_nonpfas", type=int, default=5000,
                   help="Number of non-PFAS spectra to sample (default 5000)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_peaks", type=int, default=60)
    p.add_argument("--fold", choices=["train", "val", "all"], default="all",
                   help="Which data fold to use. Defaults to 'all' because most PFAS "
                        "subclasses exist only in the train fold.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # ── Load & sample ─────────────────────────────────────────────────────────
    print("Loading data...")
    cols = ["identifier", "mzs", "intensities", "precursor_mz", "smiles", "is_PFAS", "fold"]
    df = pd.read_csv(args.data, sep="\t", usecols=cols, low_memory=False)
    df["is_PFAS"] = df["is_PFAS"].astype(str).str.lower() == "true"

    if args.fold != "all":
        df = df[df["fold"] == args.fold]

    # ── Label all PFAS rows first (needed for proportional sampling) ──────────
    pfas_all = df[df["is_PFAS"]].copy()
    pfas_all["subclass"] = pfas_all["smiles"].apply(classify_pfas_subclass)

    # Sample PFAS proportionally per subclass so rare subclasses aren't
    # swamped by the dominant FT acrylate/methacrylate group
    subclass_counts = pfas_all["subclass"].value_counts()
    total_pfas = len(pfas_all)
    pfas_parts = []
    for subclass, count in subclass_counts.items():
        n_take = max(1, round(args.n_pfas * count / total_pfas))
        n_take = min(n_take, count)
        pfas_parts.append(
            pfas_all[pfas_all["subclass"] == subclass].sample(n=n_take, random_state=args.seed)
        )
    pfas_df = pd.concat(pfas_parts, ignore_index=True)

    nonpfas_df = df[~df["is_PFAS"]].sample(
        n=min(args.n_nonpfas, (~df["is_PFAS"]).sum()), random_state=args.seed
    )
    nonpfas_df = nonpfas_df.copy()
    nonpfas_df["subclass"] = NON_PFAS_LABEL

    sample_df = pd.concat([pfas_df, nonpfas_df], ignore_index=True)
    print(f"Sample: {len(pfas_df):,} PFAS + {len(nonpfas_df):,} non-PFAS = {len(sample_df):,} total")

    # ── Build labels ──────────────────────────────────────────────────────────
    labels = sample_df["subclass"].tolist()
    print("Label distribution:", Counter(labels))

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = RawSpecDataset(sample_df, n_peaks=args.n_peaks)

    # ── Embeddings ────────────────────────────────────────────────────────────
    print("\nLoading pretrained DreaMS backbone...")
    backbone_pre = load_pretrained_backbone()
    print("Extracting pretrained embeddings...")
    emb_pre = extract_embeddings(backbone_pre, dataset, args.batch_size, device)
    print(f"Embeddings shape: {emb_pre.shape}")

    emb_ft = None
    if args.finetuned_ckpt:
        print(f"\nLoading fine-tuned checkpoint: {args.finetuned_ckpt}")
        backbone_ft = load_finetuned_backbone(args.finetuned_ckpt)
        print("Extracting fine-tuned embeddings...")
        emb_ft = extract_embeddings(backbone_ft, dataset, args.batch_size, device)

    # ── Dimensionality reduction ──────────────────────────────────────────────
    print(f"\nRunning {args.method.upper()} on pretrained embeddings...")
    coords_pre = reduce_embeddings(emb_pre, method=args.method)

    coords_ft = None
    if emb_ft is not None:
        print(f"Running {args.method.upper()} on fine-tuned embeddings...")
        coords_ft = reduce_embeddings(emb_ft, method=args.method)

    # ── Plot ──────────────────────────────────────────────────────────────────
    make_plot(coords_pre, labels, args.method, args.out, coords_finetuned=coords_ft)

    # Save raw embeddings + coords for further analysis
    emb_out = Path(args.out).with_suffix(".npz")
    save_dict = {
        "emb_pretrained": emb_pre,
        "coords_pretrained": coords_pre,
        "labels": np.array(labels),
        "identifiers": sample_df["identifier"].values,
    }
    if emb_ft is not None:
        save_dict["emb_finetuned"] = emb_ft
        save_dict["coords_finetuned"] = coords_ft
    np.savez_compressed(emb_out, **save_dict)
    print(f"Saved embeddings + coords to {emb_out}")


if __name__ == "__main__":
    main()
