"""
Side-by-side UMAP: pretrained vs fine-tuned DreaMS embeddings,
colored by ionization mode (positive / negative), derived from the adduct column.

Usage:
    # Compute embeddings from scratch:
    python scripts/visualize_pfas_umap_ionization.py \
        --data ~/Downloads/merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv \
        --finetuned_ckpt ~/Downloads/HalogenDetection-...ckpt \
        --out results/plots/pfas_umap_ionization.png

    # Re-use pre-computed .npz (must have coords_pretrained + coords_finetuned):
    python scripts/visualize_pfas_umap_ionization.py \
        --data ~/Downloads/merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv \
        --embeddings results/plots/pfas_umap_ionization.npz \
        --out results/plots/pfas_umap_ionization.png
"""

import argparse
from collections import Counter
from pathlib import Path

import matchms
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from massspecgym.data.transforms import SpecTokenizer
from dreams.api import PreTrainedModel
from dreams.models.dreams.dreams import DreaMS as DreaMSModel


# ── Ionization mode from adduct ───────────────────────────────────────────────

def ionization_mode(adduct: str) -> str:
    a = str(adduct).strip()
    if a.endswith("+"):
        return "Positive"
    if a.endswith("-") or a.endswith("–"):
        return "Negative"
    return "Unknown"


# ── Color / style config ──────────────────────────────────────────────────────

ION_CONFIG = {
    "Positive": dict(color="#C0392B", size=8, alpha=0.55),   # brick red
    "Negative": dict(color="#2166AC", size=8, alpha=0.55),   # steel blue
    "Unknown":  dict(color="#AAAAAA", size=5, alpha=0.30),
}
ION_ORDER = ["Positive", "Negative", "Unknown"]


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
            tok = self.tokenizer(spec)
        except Exception:
            tok = torch.zeros(61, 2)
        return tok.float()


# ── Backbone loading ──────────────────────────────────────────────────────────

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


# ── UMAP ──────────────────────────────────────────────────────────────────────

def reduce_umap(embeddings: np.ndarray) -> np.ndarray:
    import umap
    return umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.1,
        metric="cosine", random_state=42, verbose=True,
    ).fit_transform(embeddings)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_panel(ax, coords: np.ndarray, labels: list, title: str):
    import matplotlib.patches as mpatches

    label_arr = np.array(labels)
    present   = set(labels)

    for lab in ION_ORDER:
        if lab not in present:
            continue
        mask = label_arr == lab
        s = ION_CONFIG[lab]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=s["color"], s=s["size"], alpha=s["alpha"],
            linewidths=0, rasterized=True, zorder=2,
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP 1", fontsize=11, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=11, fontweight="bold")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    handles = [
        mpatches.Patch(color=ION_CONFIG[lab]["color"], label=lab)
        for lab in ION_ORDER if lab in present
    ]
    ax.legend(handles=handles, fontsize=10, loc="lower left",
              framealpha=0.9, edgecolor="#cccccc")


def make_plot(coords_pre, coords_ft, labels, out_path, n_total):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

    _plot_panel(axes[0], coords_pre, labels, "Pretrained DreaMS")
    _plot_panel(axes[1], coords_ft,  labels, "DreaMS-PFAS (Fine-Tuned)")

    n_pos = sum(1 for l in labels if l == "Positive")
    n_neg = sum(1 for l in labels if l == "Negative")
    fig.suptitle(
        f"Ionization Mode — UMAP  |  Positive: {n_pos:,}  |  Negative: {n_neg:,}  |  Total: {n_total:,}",
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
                   help="Path to merged TSV (needs mzs, intensities, precursor_mz, adduct columns)")
    p.add_argument("--embeddings", default=None,
                   help="Pre-computed .npz with coords_pretrained + coords_finetuned (skips inference)")
    p.add_argument("--finetuned_ckpt", default=None,
                   help="Fine-tuned checkpoint (.ckpt). Required if --embeddings not supplied.")
    p.add_argument("--out", default="results/plots/pfas_umap_ionization.png")
    p.add_argument("--n_samples",  type=int, default=10000,
                   help="Total spectra to sample (drawn proportionally by ionization mode)")
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

    # ── Load & sample ─────────────────────────────────────────────────────────
    print("Loading data...")
    cols = ["identifier", "mzs", "intensities", "precursor_mz", "adduct", "fold"]
    df = pd.read_csv(args.data, sep="\t", usecols=cols, low_memory=False)

    if args.fold != "all":
        df = df[df["fold"] == args.fold]

    df["ion_mode"] = df["adduct"].apply(ionization_mode)

    # Sample proportionally by ionization mode
    mode_counts = df["ion_mode"].value_counts()
    total = len(df)
    parts = []
    for mode, count in mode_counts.items():
        n_take = max(1, round(args.n_samples * count / total))
        n_take = min(n_take, count)
        parts.append(df[df["ion_mode"] == mode].sample(n=n_take, random_state=args.seed))
    sample_df = pd.concat(parts, ignore_index=True)

    labels = sample_df["ion_mode"].tolist()
    print(f"Sample: {len(sample_df):,} spectra")
    print("Ionization mode distribution:", Counter(labels))

    # ── Embeddings ────────────────────────────────────────────────────────────
    if args.embeddings:
        print(f"\nLoading pre-computed embeddings from {args.embeddings}")
        npz = np.load(args.embeddings, allow_pickle=True)
        assert "coords_pretrained" in npz and "coords_finetuned" in npz, (
            "NPZ must contain both 'coords_pretrained' and 'coords_finetuned'."
        )
        coords_pre = npz["coords_pretrained"]
        coords_ft  = npz["coords_finetuned"]
        assert len(coords_pre) == len(sample_df), (
            f"Embedding rows ({len(coords_pre)}) != sample rows ({len(sample_df)}). "
            "Re-generate with the same --n_samples / --seed."
        )
    else:
        assert args.finetuned_ckpt, "Provide --finetuned_ckpt (or --embeddings)."
        dataset = RawSpecDataset(sample_df, n_peaks=args.n_peaks)

        print("\nLoading pretrained backbone...")
        backbone_pre = load_pretrained_backbone()
        print("Extracting pretrained embeddings...")
        emb_pre = extract_embeddings(backbone_pre, dataset, args.batch_size, device)

        print(f"\nLoading fine-tuned checkpoint: {args.finetuned_ckpt}")
        backbone_ft = load_finetuned_backbone(args.finetuned_ckpt)
        print("Extracting fine-tuned embeddings...")
        emb_ft = extract_embeddings(backbone_ft, dataset, args.batch_size, device)

        print("\nRunning UMAP on pretrained embeddings...")
        coords_pre = reduce_umap(emb_pre)
        print("Running UMAP on fine-tuned embeddings...")
        coords_ft  = reduce_umap(emb_ft)

        emb_out = Path(args.out).with_suffix(".npz")
        np.savez_compressed(emb_out,
                            coords_pretrained=coords_pre,
                            coords_finetuned=coords_ft,
                            labels=np.array(labels))
        print(f"Saved embeddings to {emb_out}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    make_plot(coords_pre, coords_ft, labels, args.out, len(sample_df))


if __name__ == "__main__":
    main()
