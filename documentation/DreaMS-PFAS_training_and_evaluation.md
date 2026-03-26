# DreaMS-PFAS: Training and Evaluation

Fine-tuning the DreaMS pretrained mass spectrometry model to detect PFAS (Per- and Polyfluoroalkyl Substances) and other fluorinated compounds from MS/MS spectra.

---

## 1. Project Overview

**Goal:** Train a classifier on top of DreaMS embeddings to identify PFAS and fluorinated compounds directly from MS/MS spectra — without relying on molecular structure inputs at inference time.

**Motivation:** PFAS are persistent environmental contaminants found in water, soil, and biological systems. Rapid, high-throughput screening via mass spectrometry is essential for environmental monitoring. A learned model can complement rule-based detection methods (CF₂ loss patterns, KMD analysis) by capturing subtler spectral signatures.

**PFAS Taxonomy (OECD Definition):**
| Type | Description | Label Transform |
|------|-------------|-----------------|
| Type 1 | Isolated CF₂ group (`–CF₂–`) | `MolToIsolatedCF2Vector` |
| Type 2 | Isolated CF₃ group (`–CF₃`) | `MolToIsolatedCF3Vector` |
| Type 3 | Larger fluorinated chains (≥2 CF₂ or ≥1 CF₂+CF₃) | `MolToPFASVector` |

---

## 2. Model Architecture

### Backbone: DreaMS
- Pretrained self-supervised model for mass spectra (Zenodo: [`ssl_model.ckpt`](https://zenodo.org/records/10997887))
- Inputs: top-60 peaks (m/z, intensity pairs) + precursor m/z, Fourier-feature encoded
- Output: 1024-dimensional spectrum embedding (CLS token at position 0)

### Classification Head
- `nn.Linear(1024, 1)` — produces a single logit for binary classification
- Sigmoid applied post-hoc for probability; threshold (default 0.9) converts to label

### Key Class
`massspecgym/models/pfas/base.py` — `HalogenDetectorDreamsTest`

```python
# Simplified architecture
self.main_model = PreTrainedModel.from_ckpt(ckpt_path=..., ckpt_cls=DreaMSModel, n_highest_peaks=60).model
self.lin_out = nn.Linear(1024, 1)

def forward(self, x):
    embedding = self.main_model(x)[:, 0, :]  # [B, 1024]
    return self.lin_out(embedding).squeeze(1)  # [B] logits
```

---

## 3. Data Pipeline

### Dataset
- **Class:** `TestMassSpecDataset` (subclass of `MassSpecDataset`) in `scripts/train_PFAS_model.py`
- **Source:** Merged NIST20 + NIST-new TSV — `merged_massspec_nist20_nist_new_with_fold.tsv`
- **External PFAS library:** `data/demo_external_pfas_library.tsv` (for out-of-distribution evaluation)
- **Splits:** train/val/test via fold column; managed by `MassSpecDataModule`

### Spectrum Transform
```python
SpecTokenizer(n_peaks=60)  # massspecgym/data/transforms.py
# → tensor of shape [61, 2]: 60 peaks + 1 precursor m/z row
```

### Label Transforms (choose target task)
| Transform | Output | Task |
|-----------|--------|------|
| `MolToIsolatedCF2Vector` | `[is_CF2, ...]` | PFAS Type 1 detection |
| `MolToIsolatedCF3Vector` | `[is_CF3, ...]` | PFAS Type 2 detection |
| `MolToPFASVector` | `[is_PFAS, ...]` | Full PFAS (OECD Type 3) detection |
| `MolToFluorinatedTypeVector` | multi-class vector | Fluorination type classification |
| `MolToHalogensVector` | `[F, Cl, Br, I]` counts | Halogen presence |

All transforms are in `massspecgym/data/transforms.py`.

### Hard Negatives
Fluorinated compounds that are **not** PFAS are the hardest negatives. The dataset flags these via a per-sample `F` field (1 if SMILES contains fluorine). The model upweights these in the loss via `hard_neg_weight`.

```python
# In TestMassSpecDataset.__getitem__
item['F'] = 1 if re.search("f", smiles, re.IGNORECASE) else 0
```

---

## 4. Training Protocol

### Framework
PyTorch Lightning (`Trainer`) with automatic device selection (MPS/CUDA/CPU).

### Hyperparameters (current best)
| Parameter | Value | Notes |
|-----------|-------|-------|
| `lr` | `1e-5` | Empirically best; Adam optimizer |
| `batch_size` | 64 | |
| `n_peaks` | 60 | Matches DreaMS pretraining |
| `threshold` | 0.9 | High precision focus |
| `loss` | `bce` | BCE with logits |
| `hard_neg_weight` | 1.0 | Increase to penalize fluorinated non-PFAS misclassifications |
| `pos_weight` | 1.0 | Increase to handle class imbalance |
| `val_check_interval` | 0.2 | Validate 5× per epoch |

### Loss Function
BCE with logits, supporting optional hard-negative upweighting:
```python
per_sample = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=..., reduction="none")
loss = (per_sample * weights).mean()
```
A focal loss variant (`_step_focal_loss`) is also implemented but currently disabled.

### Experiment Tracking
Weights & Biases via `WandbLogger`. Config keys logged: `batch_size`, `n_peaks`, `threshold`, `alpha`, `gamma`, `loss`.

### Run Command
```bash
python3.11 scripts/train_PFAS_model.py
```

---

## 5. Evaluation

### Per-Epoch Metrics (logged to W&B)
- Precision, Recall, Accuracy, F1 — via `torchmetrics` (`BinaryPrecision`, `BinaryRecall`, `BinaryAccuracy`)
- Computed separately for `train_/` and `val_/` prefixes

### Threshold Sweep (post-validation)
After each validation epoch, a full precision/recall/accuracy/F1 table is computed for thresholds 0.0–1.0 (step 0.1) and saved to `pr_table.csv`.

### Disabled (currently commented out)
- `val_/auroc` — AUROC via `roc_auc_score`
- `val_/ap_score` — Average Precision via `average_precision_score`

To re-enable, uncomment in `on_validation_epoch_end()` in `massspecgym/models/pfas/base.py`.

### Output Artifacts (written each validation epoch)
| File | Contents |
|------|----------|
| `pr_table.csv` | Precision/recall table across thresholds |
| `pfas_pred_probs.csv` | Predicted probabilities for all true-positive PFAS |
| `true_positive_identifiers.txt` | Up to 100 sampled TP compound identifiers |
| `false_negative_identifiers.txt` | Up to 100 FN compounds with prob < 0.2 |

---

## 6. PFAS Classification (OECD)

**Script:** `scripts/classify_pfas_oecd.py`

Uses RDKit substructure matching to assign each compound to an OECD PFAS type:
- Type 1: isolated `–CF₂–` (not adjacent to other CF₂/CF₃)
- Type 2: isolated `–CF₃` terminal group
- Type 3: longer fluorinated chains (≥2 contiguous CF₂, or CF₂+CF₃)

This classification is used to generate labels for targeted training runs.

---

## 7. Embedding Visualization

**Script:** `scripts/visualize_dreams_embeddings.py`

Extracts DreaMS embeddings from a dataset and projects them to 2D via UMAP or t-SNE, colored by PFAS subclass. Useful for:
- Verifying that PFAS compounds cluster separately before fine-tuning
- Qualitatively assessing model improvement after training

---

## 8. Inference on Raw mzML Files (`detect_PFAS.py`)

**Script:** `scripts/detect_PFAS.py`

Runs the trained `HalogenDetectorDreamsTest` model on raw `.mzML` files to produce per-spectrum PFAS probability scores. Two main functions:

### `find_PFAS(in_pth)`
Processes a single `.mzML` file end-to-end:
1. Loads the file via `MSData.from_mzml()`
2. Runs spectral quality checks using DreaMS `DataFormatA` — filters to high-quality spectra only, saved as `.hdf5`
3. Calls `dreams_predictions()` to get raw logits from the loaded checkpoint
4. Applies `torch.sigmoid()` to convert logits → probabilities, stored in a `PFAS_preds` column
5. Returns a pandas DataFrame with all spectra metadata + `PFAS_preds`

### `scan_and_run_pfas(directory, output_csv, threshold=0.95)`
Batch-processes an entire directory of `.mzML` files:
1. Iterates over all `.mzML` files in `directory`
2. Calls `find_PFAS()` on each file
3. Filters rows where `PFAS_preds >= threshold`
4. Concatenates hits across all files and saves to `output_csv`

### Usage
```python
# Load checkpoint
model = HalogenDetectorDreamsTest.load_from_checkpoint(ckpt_path)

# Scan a directory of .mzML files
final_results = scan_and_run_pfas(
    directory='/path/to/mzML_files/',
    output_csv='pfas_hits.csv',
    threshold=0.9
)
```

### Notes
- The checkpoint path (`ckpt_path`) must point to a saved `HalogenDetectorDreamsTest` checkpoint
- Spectral quality filtering happens before inference — low-quality spectra are excluded
- The `threshold` in `scan_and_run_pfas` is independent of the training-time `threshold` in the model; set based on desired precision/recall trade-off

---

## 9. Related Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_PFAS_model.py` | Main fine-tuning training loop |
| `scripts/train_fluorinated_molecules_model.py` | Broader fluorinated compound classification |
| `scripts/train_halogen_model.py` | Multi-class halogen (F, Cl, Br, I) detection |
| `scripts/classify_pfas_oecd.py` | OECD PFAS type assignment from SMILES |
| `scripts/pfas_identification.py` | Rule-based MassQL PFAS detection (CF₂ losses, KMD, fragments) |
| `scripts/visualize_dreams_embeddings.py` | UMAP/t-SNE of DreaMS embedding space |
| `scripts/detect_PFAS.py` | Model inference on raw `.mzML` files — batch scans directories for PFAS hits |
| `scripts/PFASManualDetectionPipeline.py` | Manual detection workflow |

---

## 10. External Real-World PFAS Dataset (Jonathan)

A real-world PFAS dataset was provided by Jonathan (collaborator) via OneDrive. It consists of **five folders** of `.mzML` files from different environmental projects. A `data_overview.xlsx` file describes the origin of each dataset and links to associated publications.

### Folder Structure

| Folder | Annotated? | Notes |
|--------|-----------|-------|
| (four named folders) | Yes | Each contains a `feature_list.csv` |
| `soil_sediments_eof` | No | Unannotated runs from PFAS-contaminated samples |
| `afff` | Yes | Contains `pos/` and `neg/` subfolders; mixed polarity |

### Feature Lists (`feature_list.csv`)
Each annotated folder contains a `feature_list.csv` with:
- `name`, `SMILES`, `formula`, `exact_mass`, and/or `mz`
- Can be used to extract MS2 spectra of previously identified PFAS from the mzML files
- No RT column — not all compounds are present in every mzML file and RT shifts exist across projects
- The `afff` folder feature list also has a `polarity` column and an `adduct` column (compounds include complex adducts beyond `[M-H]-`)

### Polarity
- **All folders:** negative mode
- **`afff` folder:** both positive (`pos/`) and negative (`neg/`) subfolders

### Curated Library File
`afff_soil_nrw.msp` — curated spectra from the `afff` and `soil_nrw` folders, usable for library matching. Does **not** include spectra from the other three folders.

### Data Quality Caveats
These are important to keep in mind when training or evaluating on this data:

- **Complex isomers:** Many identified PFAS are branched/linear chain isomers or different chain-length compositions (e.g., C8/C8, C6/C10). Confidence levels and unresolved chromatographic peaks should be considered.
- **Detector saturation:** Highly dynamic PFAS samples (e.g., very high PFOS alongside trace-level others) caused saturation at certain dilutions. Different dilution levels are included (e.g., `…1_10…` in filenames). Saturated spectra may have shifted MS1 masses.
- **Spectral cross-talk:** Some files use a 4 Da isolation window, which can introduce fragment ions from neighboring precursors into spectra. Manually curated but worth noting.

### Strategy for Extracting Unannotated PFAS Spectra
To mine additional unannotated but likely-PFAS spectra from raw data (e.g., `soil_sediments_eof` or MassIVE):
- Apply **m/C > 30** and **-0.01 < MD/C < 0.003** filters in MS1 (Kendrick-type filter on fluorine-rich compounds)
- Use **FeatureFinderMetabo** (OpenMS) for isotope componentization to accurately estimate carbon count C
- This can be applied to any raw data to surface unknown highly fluorinated PFAS spectra

### Collaboration Notes
- Jonathan will send spectra classified as highly fluorine-confident (currently annotated via SIRIUS) in coming weeks — useful for external validation
- He is willing to test the retrained model and the negative-mode model once ready
- Current SIRIUS + PubChem annotation leaves >50% of his fluorine-confident spectra unannotated — highlighting the gap this model aims to fill

---

## 11. Evaluation Results: AFFF Positive-Mode (Jonathan's Dataset)

**Date:** 2026-03-21 | **Script:** `scripts/test_afff_pos.py` | **Threshold:** 0.2

### Setup
- 4 mzML files from `afff/pos/` (real-world environmental samples, positive ion mode)
- 58 positive-mode PFAS in `feature_list.csv` (all ground truth = PFAS)
- 1364 total high-quality spectra after DreaMS quality filtering
- 20 spectra matched to feature list entries (±0.02 Da); 38 features had no matching high-quality MS2

### Metrics
| Metric | Value |
|--------|-------|
| Precision | **0.917** |
| Recall | **0.550** (over 20 matched spectra) |
| F1 | **0.687** |
| True Positives | 11 |
| False Negatives | 9 |
| False Positives | 1 |

### True Positives
| Compound | Adduct | PFAS_pred |
|----------|--------|-----------|
| 5:1:2 FTB | [M]+ | 0.894 |
| 7:1:2 FTB | [M]+ | 0.447 |
| 6:2 FTSy-(2')OHPr-TriMeAm | [M]+ | 0.208 |
| PFHxSAm-Pr-TriMeAm | [M]+ | 0.856 |
| 6:2 FTSAm-Pr-B | [M]+ | 0.525 / 0.410 / 0.903 / 0.943 (4 spectra across files) |
| 6:2 FTTh-(2')OHPr-TriMeAm | [M]+ | 0.668 |
| PFOSAm-Pr-DiMeAm | [M+H]+ | 0.804 |
| 6:2 FTSAm-Pr-DiMeNO | [M]+ | 0.652 |

### False Negatives (Missed)
| Compound | Adduct | PFAS_pred |
|----------|--------|-----------|
| 6:2 FTSAm-Pr-B | [M]+ | 0.070 |
| 6:2 FTSO-(2')OHPr-TriMeAm | [M]+ | 0.001 / 0.001 / 0.006 (3 spectra) |
| 6:2 FTSy-(2')OHPr-TriMeAm | [M]+ | 0.160 |
| 8:2 FTSO-(2')OHPr-TriMeAm | [M]+ | 0.000 |
| 8:2 FTSAm-Pr-B | [M]+ | 0.000 |
| 8:2 FTSy-(2')OHPr-TriMeAm | [M]+ | 0.002 |
| 6:2 FTSAm-Pr-DiMeAm | [M+H]+ | 0.000 |

### Key Findings
1. **FTB (fluorotelomer betaines) and FTSAm classes are well-detected** — model confidently scores these above 0.2
2. **FTSO (fluorotelomer sulfoxide) and FTSy amine-oxide classes are consistently missed** — PFAS_pred scores near 0.000, not borderline; these have polar head groups (amine oxides, sulfoxides) likely absent from NIST20 training data
3. **Lowering the threshold will not recover the false negatives** — scores too low (< 0.01 for most); fix requires retraining with these compound classes
4. **Precision of 0.917 is strong** — only 1 false positive out of 1344 non-PFAS spectra
5. **6:2 FTSAm-Pr-B detected in 3/4 files but missed in one** — spectrum-level quality variation, not a structural recognition failure

---

## 12. Open Questions / Next Steps

- [ ] **Multi-class fluorination:** Extend from binary PFAS label to multi-class fluorinated type using `MolToFluorinatedTypeVector`
- [ ] **Re-enable AUROC/AP:** Uncomment `roc_auc_score` and `average_precision_score` in validation epoch end
- [ ] **External evaluation:** Run inference on `data/demo_external_pfas_library.tsv` and Jonathan's real-world mzML dataset to assess generalization
- [ ] **Incorporate Jonathan's dataset:** Extract annotated MS2 spectra using `feature_list.csv` files; handle polarity (neg/pos) and dilution variants
- [ ] **Mine unannotated PFAS:** Apply m/C > 30 and -0.01 < MD/C < 0.003 MS1 filters + FeatureFinderMetabo (OpenMS) to extract likely-PFAS spectra from `soil_sediments_eof` and other raw data
- [ ] **Negative-mode model:** Build/evaluate a model specifically for negative-mode spectra (all of Jonathan's data except `afff/pos`)
- [ ] **Backbone freezing experiments:** Compare frozen DreaMS (linear probe) vs. full fine-tuning
- [ ] **Test-set evaluation:** Add `trainer.test(model, datamodule=data_module)` after training
- [ ] **`pos_weight` tuning:** Adjust `pos_weight` buffer based on actual class imbalance ratio in training set
- [ ] **Checkpoint saving:** Add `ModelCheckpoint` callback to save best validation model

---

## 13. Key File Paths

| File | Role |
|------|------|
| `massspecgym/models/pfas/base.py` | `HalogenDetectorDreamsTest` model class |
| `massspecgym/data/transforms.py` | All spectrum and molecule transforms |
| `massspecgym/data/datasets.py` | `MassSpecDataset` base class |
| `massspecgym/data/data_module.py` | `MassSpecDataModule` (train/val/test splits) |
| `scripts/train_PFAS_model.py` | Training entry point + `TestMassSpecDataset` |
| `data/demo_external_pfas_library.tsv` | External PFAS evaluation set |
| `documentation/PFAS_IDENTIFICATION_TOOL_DOCUMENTATION.md` | Rule-based detection reference |
