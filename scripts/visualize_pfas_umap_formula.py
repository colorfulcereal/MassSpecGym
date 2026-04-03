"""
UMAP visualization of DreaMS embeddings colored by number of fluorine atoms
(1–3, 4–7, 8–12, 13+), derived from precursor_formula — no SMARTS needed.
Non-PFAS shown as a gray background.

Usage:
    # From pre-computed embeddings (.npz from visualize_pfas_umap.py):
    python scripts/visualize_pfas_umap_formula.py \
        --data ~/Downloads/merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv \
        --embeddings results/plots/pfas_umap.npz \
        --out results/plots/pfas_umap_fluorine.png

    # Recompute embeddings from scratch:
    python scripts/visualize_pfas_umap_formula.py \
        --data ~/Downloads/merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv \
        --finetuned_ckpt ~/Downloads/HalogenDetection-...ckpt \
        --out results/plots/pfas_umap_fluorine.png
"""

import argparse
import re
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


# ── Formula parsing ───────────────────────────────────────────────────────────

def _count_element(formula: str, element: str) -> int:
    """Extract count of an element from a molecular formula string."""
    pattern = rf"{element}(\d*)"
    match = re.search(pattern, str(formula))
    if match is None:
        return 0
    n = match.group(1)
    return int(n) if n else 1


def fluorine_bin(formula: str) -> str:
    n = _count_element(formula, "F")
    if n == 0:
        return "Unknown"
    if n <= 3:
        return "1–3 F"
    if n <= 7:
        return "4–7 F"
    if n <= 12:
        return "8–12 F"
    return "13+ F"


# ── Color / style config ──────────────────────────────────────────────────────

NON_PFAS_LABEL = "Non-PFAS"
NON_PFAS_STYLE = dict(color="#CCCCCC", size=3, alpha=0.15)

# Fluorine count — cool sequential blue
FLUOR_CONFIG = {
    "1–3 F":  dict(color="#C6DBEF", size=25, alpha=0.90),
    "4–7 F":  dict(color="#6BAED6", size=25, alpha=0.90),
    "8–12 F": dict(color="#2171B5", size=25, alpha=0.90),
    "13+ F":  dict(color="#08306B", size=25, alpha=0.90),
}
FLUOR_ORDER = [NON_PFAS_LABEL, "1–3 F", "4–7 F", "8–12 F", "13+ F"]


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


# ── UMAP ──────────────────────────────────────────────────────────────────────

def reduce_umap(embeddings: np.ndarray) -> np.ndarray:
    import umap
    return umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=0.1,
        metric="cosine", random_state=42, verbose=True,
    ).fit_transform(embeddings)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_panel(ax, coords, labels, config, order, title):
    import matplotlib.patches as mpatches

    label_arr = np.array(labels)
    present   = set(labels)

    for lab in order:
        if lab not in present:
            continue
        mask = label_arr == lab
        if lab == NON_PFAS_LABEL:
            s = NON_PFAS_STYLE
        else:
            s = config[lab]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=s["color"], s=s["size"], alpha=s["alpha"],
            linewidths=0, rasterized=True,
            zorder=(1 if lab == NON_PFAS_LABEL else 2),
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP 1", fontsize=11, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=11, fontweight="bold")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    handles = [
        mpatches.Patch(
            color=(NON_PFAS_STYLE["color"] if lab == NON_PFAS_LABEL else config[lab]["color"]),
            label=lab,
        )
        for lab in order if lab in present
    ]
    ax.legend(handles=handles, fontsize=10, loc="lower left",
              framealpha=0.9, edgecolor="#cccccc")


def make_plot(coords_pre, coords_ft, fluor_labels, out_path, n_pfas, n_nonpfas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

    _plot_panel(axes[0], coords_pre, fluor_labels, FLUOR_CONFIG, FLUOR_ORDER,
                "Pretrained DreaMS")
    _plot_panel(axes[1], coords_ft,  fluor_labels, FLUOR_CONFIG, FLUOR_ORDER,
                "DreaMS-PFAS (Fine-Tuned)")

    fig.suptitle(
        f"Fluorine Atom Segmentation — UMAP  |  PFAS: {n_pfas:,}  |  Non-PFAS: {n_nonpfas:,}",
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
                   help="Path to merged TSV with mzs/intensities/precursor_formula columns")
    p.add_argument("--embeddings", default=None,
                   help="Pre-computed .npz with 'coords_pretrained' and 'coords_finetuned' keys (skips inference)")
    p.add_argument("--finetuned_ckpt", required=False, default=None,
                   help="Fine-tuned checkpoint (.ckpt). Required if --embeddings not supplied.")
    p.add_argument("--out", default="results/plots/pfas_umap_formula.png")
    p.add_argument("--n_pfas",     type=int, default=5000)
    p.add_argument("--n_nonpfas",  type=int, default=5000)
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

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    cols = ["identifier", "mzs", "intensities", "precursor_mz",
            "precursor_formula", "is_PFAS", "fold"]
    df = pd.read_csv(args.data, sep="\t", usecols=cols, low_memory=False)
    df["is_PFAS"] = df["is_PFAS"].astype(str).str.lower() == "true"

    if args.fold != "all":
        df = df[df["fold"] == args.fold]

    pfas_df = df[df["is_PFAS"]].sample(
        n=min(args.n_pfas, df["is_PFAS"].sum()), random_state=args.seed
    ).copy()
    nonpfas_df = df[~df["is_PFAS"]].sample(
        n=min(args.n_nonpfas, (~df["is_PFAS"]).sum()), random_state=args.seed
    ).copy()

    sample_df = pd.concat([pfas_df, nonpfas_df], ignore_index=True)
    n_pfas    = len(pfas_df)
    n_nonpfas = len(nonpfas_df)
    print(f"Sample: {n_pfas:,} PFAS + {n_nonpfas:,} Non-PFAS = {len(sample_df):,}")

    # ── Build fluorine labels from formula ───────────────────────────────────
    fluor_labels = []
    for _, row in sample_df.iterrows():
        if not row["is_PFAS"]:
            fluor_labels.append(NON_PFAS_LABEL)
        else:
            fluor_labels.append(fluorine_bin(row.get("precursor_formula", "")))

    print("Fluorine count distribution:", Counter(fluor_labels))

    # ── Embeddings ────────────────────────────────────────────────────────────
    if args.embeddings:
        print(f"\nLoading pre-computed embeddings from {args.embeddings}")
        npz = np.load(args.embeddings, allow_pickle=True)
        assert "coords_pretrained" in npz and "coords_finetuned" in npz, (
            "NPZ must contain both 'coords_pretrained' and 'coords_finetuned' keys. "
            "Re-run with --finetuned_ckpt to generate them."
        )
        coords_pre = npz["coords_pretrained"]
        coords_ft  = npz["coords_finetuned"]
        assert len(coords_pre) == len(sample_df), (
            f"Embedding rows ({len(coords_pre)}) != sample rows ({len(sample_df)}). "
            "Re-generate with the same --n_pfas / --n_nonpfas / --seed."
        )
    else:
        assert args.finetuned_ckpt, "Provide --finetuned_ckpt (or --embeddings) to run."
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
        coords_ft = reduce_umap(emb_ft)

        emb_out = Path(args.out).with_suffix(".npz")
        np.savez_compressed(emb_out,
                            coords_pretrained=coords_pre,
                            coords_finetuned=coords_ft,
                            labels_fluor=np.array(fluor_labels))
        print(f"Saved embeddings to {emb_out}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    make_plot(coords_pre, coords_ft, fluor_labels, args.out, n_pfas, n_nonpfas)


if __name__ == "__main__":
    main()
