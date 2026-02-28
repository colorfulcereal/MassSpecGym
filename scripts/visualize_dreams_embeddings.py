"""
Visualize DreaMS embeddings with UMAP / t-SNE, colored by PFAS class.

Usage (on GPU environment):
    # Pretrained DreaMS only:
    python scripts/visualize_dreams_embeddings.py \
        --data /path/to/merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv \
        --out results/plots/umap_embeddings.png

    # Compare pretrained vs fine-tuned (two-panel):
    python scripts/visualize_dreams_embeddings.py \
        --data /path/to/merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv \
        --finetuned_ckpt /path/to/checkpoint.ckpt \
        --out results/plots/umap_embeddings.png

    # Use t-SNE instead of UMAP:
    python scripts/visualize_dreams_embeddings.py --method tsne ...
"""

import argparse
import sys
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


# ── PFAS subclass labels ──────────────────────────────────────────────────────

def classify_pfas_subclass(smiles: str) -> str:
    """Derive a coarse PFAS subclass from SMILES."""
    import re
    if not smiles or smiles == "nan":
        return "Other PFAS"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Other PFAS"
    f_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    if f_count < 4:
        return "Other PFAS"
    if re.search(r'C\(=O\)O|C\(=O\)\[O', smiles):
        return "PFCA (e.g. PFOA)"
    if re.search(r'S\(=O\)', smiles):
        return "PFSA (e.g. PFOS)"
    if re.search(r'P\(=O\)', smiles):
        return "Fluorotelomer phosphate"
    return "Other PFAS"


# ── Dataset ───────────────────────────────────────────────────────────────────

class RawSpecDataset(Dataset):
    """
    Builds matchms Spectrum objects from raw mzs/intensities strings in the TSV
    and applies SpecTokenizer on the fly.
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
    """Load the fine-tuned FluorineDetectorDreamsTest and return its DreaMS backbone."""
    # Import here to avoid polluting the namespace
    sys.path.insert(0, str(Path(__file__).parent))
    from train_fluorinated_molecules_model import FluorineDetectorDreamsTest

    model = FluorineDetectorDreamsTest.load_from_checkpoint(
        ckpt_path, map_location="cpu"
    )
    backbone = model.main_model.eval()
    return backbone


@torch.no_grad()
def extract_embeddings(
    backbone: nn.Module,
    dataset: RawSpecDataset,
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Run forward pass and collect CLS token embeddings [N, 1024]."""
    backbone = backbone.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_embs = []
    for batch in loader:
        batch = batch.to(device)               # [B, n_peaks+1, 2]
        emb = backbone(batch)[:, 0, :]         # CLS token [B, 1024]
        all_embs.append(emb.cpu().float().numpy())
    return np.vstack(all_embs)


# ── Dimensionality reduction ──────────────────────────────────────────────────

def reduce_embeddings(embeddings: np.ndarray, method: str = "umap") -> np.ndarray:
    if method == "umap":
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
            verbose=True,
        )
        return reducer.fit_transform(embeddings)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            n_iter=1000,
            metric="cosine",
            random_state=42,
            verbose=1,
        )
        return reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'umap' or 'tsne'.")


# ── Plotting ──────────────────────────────────────────────────────────────────

# Color palette — non-PFAS as neutral gray, PFAS subclasses as vivid colors
LABEL_COLORS = {
    "Non-PFAS":                 "#cccccc",
    "PFCA (e.g. PFOA)":         "#e63946",
    "PFSA (e.g. PFOS)":         "#f4a261",
    "Fluorotelomer phosphate":  "#2a9d8f",
    "Other PFAS":               "#457b9d",
}
LABEL_SIZES = {
    "Non-PFAS": 4,
    "PFCA (e.g. PFOA)": 18,
    "PFSA (e.g. PFOS)": 18,
    "Fluorotelomer phosphate": 18,
    "Other PFAS": 18,
}
LABEL_ALPHA = {
    "Non-PFAS": 0.25,
    "PFCA (e.g. PFOA)": 0.9,
    "PFSA (e.g. PFOS)": 0.9,
    "Fluorotelomer phosphate": 0.9,
    "Other PFAS": 0.9,
}
LABEL_ORDER = [
    "Non-PFAS",
    "Other PFAS",
    "Fluorotelomer phosphate",
    "PFSA (e.g. PFOS)",
    "PFCA (e.g. PFOA)",
]


def _plot_single(ax, coords: np.ndarray, labels: list, title: str, method: str):
    import matplotlib.patches as mpatches

    label_arr = np.array(labels)
    for lab in LABEL_ORDER:
        mask = label_arr == lab
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=LABEL_COLORS[lab],
            s=LABEL_SIZES[lab],
            alpha=LABEL_ALPHA[lab],
            linewidths=0,
            label=lab,
            rasterized=True,
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(f"{method.upper()} 1", fontsize=10)
    ax.set_ylabel(f"{method.upper()} 2", fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Legend (only PFAS labels)
    handles = [
        mpatches.Patch(color=LABEL_COLORS[lab], label=lab)
        for lab in LABEL_ORDER
        if lab in set(labels)
    ]
    ax.legend(handles=handles, fontsize=8, loc="best", markerscale=1.5, framealpha=0.8)


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

    n_pfas = sum(1 for l in labels if l != "Non-PFAS")
    n_total = len(labels)
    fig.suptitle(
        f"DreaMS Embeddings — {method.upper()}  |  n={n_total:,} ({n_pfas:,} PFAS)",
        fontsize=14,
        y=1.01,
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
    p.add_argument("--n_nonpfas", type=int, default=3000,
                   help="Number of non-PFAS spectra to sample (default 3000)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_peaks", type=int, default=60)
    p.add_argument("--fold", choices=["train", "val", "all"], default="val",
                   help="Which data fold to use. 'val' is faster and less biased.")
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

    # ── Load & sample data ────────────────────────────────────────────────────
    print("Loading data...")
    cols = ["identifier", "mzs", "intensities", "precursor_mz", "smiles", "is_PFAS", "fold"]
    df = pd.read_csv(args.data, sep="\t", usecols=cols, low_memory=False)
    df["is_PFAS"] = df["is_PFAS"].astype(str).str.lower() == "true"

    if args.fold != "all":
        df = df[df["fold"] == args.fold]

    pfas_df = df[df["is_PFAS"]].copy()
    nonpfas_df = df[~df["is_PFAS"]].sample(
        n=min(args.n_nonpfas, len(df[~df["is_PFAS"]])), random_state=args.seed
    )
    sample_df = pd.concat([pfas_df, nonpfas_df], ignore_index=True)

    print(f"Sample: {len(pfas_df):,} PFAS + {len(nonpfas_df):,} non-PFAS = {len(sample_df):,} total")

    # ── Build labels ──────────────────────────────────────────────────────────
    labels = []
    for _, row in sample_df.iterrows():
        if not row["is_PFAS"]:
            labels.append("Non-PFAS")
        else:
            labels.append(classify_pfas_subclass(str(row["smiles"])))

    from collections import Counter
    print("Label distribution:", Counter(labels))

    # ── Build dataset ─────────────────────────────────────────────────────────
    dataset = RawSpecDataset(sample_df, n_peaks=args.n_peaks)

    # ── Extract embeddings (pretrained) ───────────────────────────────────────
    print("\nLoading pretrained DreaMS backbone...")
    backbone_pre = load_pretrained_backbone()
    print("Extracting pretrained embeddings...")
    emb_pre = extract_embeddings(backbone_pre, dataset, args.batch_size, device)
    print(f"Embeddings shape: {emb_pre.shape}")

    # ── Extract embeddings (fine-tuned, optional) ─────────────────────────────
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

    # Also save embeddings + labels for further analysis
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
