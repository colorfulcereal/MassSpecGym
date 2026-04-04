# python scripts/test_PFAS_golden_dataset.py
#
# Evaluates DreaMS-PFAS on the pre-processed test set TSV.
# All spectra are true PFAS (is_PFAS=1). Primary metric: Recall at threshold=0.2.

from pathlib import Path
import numpy as np
import pandas as pd
import matchms
import torch
from sklearn.metrics import recall_score, classification_report

from massspecgym.data.transforms import SpecTokenizer
from massspecgym.models.pfas import HalogenDetectorDreamsTest
from dreams.api import dreams_predictions

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_LOCAL_TSV  = Path.home() / "Downloads" / "afff_pos_matched_spectra_test_set.tsv"
_REMOTE_TSV = Path("/teamspace/studios/this_studio/MassSpecGym/afff/afff_pos_matched_spectra_test_set.tsv")
INPUT_TSV   = _LOCAL_TSV if _LOCAL_TSV.exists() else _REMOTE_TSV
OUTPUT_CSV = Path("afff_pos_tsv_results.csv")
N_PEAKS    = 60
THRESHOLD  = 0.2

_LOCAL_CKPT  = Path.home() / "Downloads" / "HalogenDetection-FocalLoss-MergedMassSpecNIST20_NISTNew_NormalPFAS_ujmvyfxm_checkpoints_epoch=0-step=9285.ckpt"
_REMOTE_CKPT = "/teamspace/studios/this_studio/HalogenDetection-FocalLoss-MergedMassSpecNIST20_NISTNew_NormalPFAS/ujmvyfxm/checkpoints/epoch=0-step=9285.ckpt"

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
ckpt_path = str(_LOCAL_CKPT) if _LOCAL_CKPT.exists() else _REMOTE_CKPT
print(f"Loading checkpoint: {ckpt_path}")
model = HalogenDetectorDreamsTest.load_from_checkpoint(ckpt_path)
model.eval()

# ---------------------------------------------------------------------------
# Load TSV
# ---------------------------------------------------------------------------
print(f"\nLoading TSV: {INPUT_TSV}")
df = pd.read_csv(INPUT_TSV, sep="\t")
print(f"Loaded {len(df)} spectra  (is_PFAS=1: {df['is_PFAS'].sum()})")

# ---------------------------------------------------------------------------
# Build spectrum list for dreams_predictions
# ---------------------------------------------------------------------------
tokenizer = SpecTokenizer(n_peaks=N_PEAKS)
tokens = []

for _, row in df.iterrows():
    mzs  = np.array([float(x) for x in str(row["mzs"]).split(",")])
    ints = np.array([float(x) for x in str(row["intensities"]).split(",")])
    spec = matchms.Spectrum(
        mz=mzs,
        intensities=ints,
        metadata={"precursor_mz": float(row["precursor_mz"])},
    )
    try:
        tok = tokenizer(spec)
    except Exception:
        tok = torch.zeros(N_PEAKS + 1, 2)
    tokens.append(tok.float())

tokens_tensor = torch.stack(tokens)   # [N, n_peaks+1, 2]

# ---------------------------------------------------------------------------
# Run model
# ---------------------------------------------------------------------------
print("Running DreaMS-PFAS predictions...")
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

model = model.to(device)
batch_size = 128
all_probs = []

with torch.no_grad():
    for i in range(0, len(tokens_tensor), batch_size):
        batch = tokens_tensor[i : i + batch_size].to(device)
        logits = model(batch)
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs.tolist())

probs_arr = np.array(all_probs)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
y_true = df["is_PFAS"].values.astype(int)
y_pred = (probs_arr >= THRESHOLD).astype(int)

results = df[["identifier", "precursor_mz", "adduct", "SMILES"]].copy()
results["PFAS_prob"]      = probs_arr
results["predicted_PFAS"] = y_pred
results["true_label"]     = y_true
results.to_csv(OUTPUT_CSV, index=False)

n_total   = len(results)
n_flagged = int(y_pred.sum())
recall    = recall_score(y_true, y_pred, zero_division=0)

print(f"\n{'='*55}")
print(f"  Total spectra          : {n_total}")
print(f"  Flagged as PFAS        : {n_flagged}  ({100*n_flagged/n_total:.1f}%)")
print(f"  Threshold              : {THRESHOLD}")
print(f"  Recall                 : {recall:.3f}  ({recall*100:.1f}%)")
print(f"{'='*55}")
print(f"\n{classification_report(y_true, y_pred, target_names=['non-PFAS', 'PFAS'], zero_division=0)}")
print(f"Results saved to: {OUTPUT_CSV}")
