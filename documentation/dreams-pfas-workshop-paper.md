# DreaMS-PFAS Workshop Paper — Running Context

Living scratchpad for the follow-up workshop paper extending the ISEF report
`[AJAS-Hamsini Ramanathan] DreaMS-PFAS Report.pdf`. Update this file as the
plan solidifies — treat it as the shared memory between sessions, not a
finished spec.

---

## 1. Where this picks up: the ISEF paper

**DreaMS-PFAS** fine-tunes the pretrained DreaMS transformer (116M params,
self-supervised on 24M MS/MS spectra) with a linear classification head to
detect PFAS from tandem mass spectra, without hand-crafted rules.

- Training data: 737,876 spectra merged from MassSpecGym + NIST20 +
  NIST-PFAS (57,039 unique molecules; 117 unique PFAS molecules / 20,194
  PFAS spectra). PFAS label = SMILES contains a perfluoroalkyl substructure
  with ≥2 connected fully-fluorinated aliphatic carbons (CF2/CF3).
- Split: molecule-level Murcko histogram split (80/20) to avoid scaffold
  leakage.
- Result: 93.8% precision / 89% recall / F1 0.91 at threshold 0.2 (AUC-PR
  0.9517, AUC-ROC 0.9733) — vs. rule-based baselines (PFΔScreen: 87.7%
  precision / 22.1% recall / F1 0.35). >4x recall improvement.
- Real-world test: 327,130 spectra from a Moorea, French Polynesia MassIVE
  marine dataset (MSV000099559) → 12 putative hits, 9 not caught by
  PFΔScreen, all with characteristic negative mass defects, 4 with clear
  PFAS fragmentation (C2F4 ladder etc.).
- Binary classifier only. Explicitly flagged in the paper's own Limitations
  (§4.3) and Future Work (§4.4):
  - Isolated CF2/CF3 ("drug-like PFAS") compounds are underrepresented in
    training data and were excluded as positives.
  - Planned: 4-class taxonomy (environmental PFAS chain / isolated
    CF2-CF3 / other fluorinated / non-fluorinated), extension to other
    halogens (Cl, Br), more experimental validation.

## 2. Problems the mentor/professor want addressed (workshop paper scope)

Given verbally by the user (Hamsini) — dictated, so phrasing is
reconstructed; confirm against the mentor's written summary once shared.

1. **Ion-mode (polarity) skew in the training data.** The PFAS-labeled
   training data is heavily skewed toward *negative* ion mode spectra,
   while DreaMS's own pretraining corpus (GNPS) is predominantly
   *positive* ion mode. This is a distribution mismatch between what the
   backbone learned to represent well and what the PFAS fine-tuning data
   looks like — likely a real driver of generalization gaps, not just a
   dataset artifact. Need a plan to characterize and correct for it
   (rebalancing, stratified sampling, explicit polarity conditioning,
   evaluating positive- vs negative-mode subsets separately, etc.).
2. **Build a larger, richer dataset.** Go beyond MassSpecGym + NIST20 +
   NIST-PFAS — need more PFAS-positive spectra, and specifically more
   positive-ion-mode PFAS spectra to correct problem #1.
3. **Move from binary to a finer-grained taxonomy.** Classify:
   - isolated CF2 PFAS
   - isolated CF3 PFAS
   - recurring/chain CF2 PFAS (the existing "environmental PFAS" class,
     ≥2 connected CF2/CF3)

   This is the same direction as the ISEF paper's own §4.4 future work.

## 3. Relevant existing codebase assets (already in this repo)

Found while orienting — the multi-class direction is not starting from
scratch:

- `documentation/DreaMS-PFAS_training_and_evaluation.md` — existing internal
  writeup of the training/eval pipeline; includes an OECD-style PFAS
  taxonomy table (Type 1 isolated CF2, Type 2 isolated CF3, Type 3 chain
  PFAS) with transforms already named.
- `massspecgym/data/transforms.py`:
  - `MolToPFASVector` — chain PFAS (≥2 connected PF-carbons) label
  - `MolToIsolatedCF2Vector`, `MolToIsolatedCF3Vector` — isolated-group
    labels
  - `MolToFluorinatedTypeVector` (line ~602) — **already implements a
    4-class mutually-exclusive scheme**: 0 = isolated CF2/CF3, 1 = PFAS
    chain, 2 = other fluorine, 3 = non-fluorinated. This looks like the
    natural starting point for the multi-class head, may just need
    wiring into a training script + eval.
- `massspecgym/models/pfas/base.py` — `HalogenDetectorDreamsTest`, the
  DreaMS-backbone + linear-head binary classifier used in the ISEF paper.
- Ion-mode / polarity is already a known area of exploration in-repo:
  - `scripts/visualize_pfas_umap_ionization.py` — buckets spectra into
    Positive/Negative/Unknown by adduct and visualizes UMAP by ionization
    mode. Worth running/reviewing before designing the fix for problem #1.
  - `scripts/test_afff_pos.py`, `afff_pos_pfascreen_results/` — positive-mode
    AFFF (aqueous film-forming foam) PFAS results already exist.
- Other related scripts: `scripts/train_PFAS_model.py`,
  `scripts/train_halogen_model.py`,
  `scripts/train_fluorinated_molecules_model.py`.

## 4. Open questions (need mentor's full written summary / user input)

- What's the target venue/format for the workshop paper (page limit, camera-ready
  deadline)? Affects how much new data collection + retraining is
  realistic.
- For the ion-mode fix: are we (a) sourcing more positive-mode PFAS
  reference spectra, (b) reweighting/resampling existing data, (c) adding
  polarity as an explicit model input/covariate, or (d) training
  polarity-specific heads? Need the mentor's intended direction.
- For the richer dataset: which new sources are in scope (additional NIST
  releases, MassBank, GNPS public PFAS libraries, MoNA, in-house AFFF data
  in `afff_pos_pfascreen_results/`)?
- Multi-class taxonomy: does the mentor want exactly the existing 4-class
  `MolToFluorinatedTypeVector` scheme, or a different split (e.g. separate
  isolated-CF2 vs isolated-CF3 rather than merged)?
- Should this be a new model/paper artifact ("DreaMS-PFAS v2") or an
  extension/ablation section added to the existing work?

## 5. Mentor's (Roman's) full proposed roadmap

Given verbally, six steps, meant to be worked through sequentially with
checkpoints — not all decided up front:

1. Extend the model with an ionization-mode (polarity) feature.
2. Retrain on the new Enveda-180 dataset
   (https://zenodo.org/records/20436851 — large, ~1-in-4 molecules
   fluorinated). Also retrain a plain fluorine detector on this data, not
   just PFAS — both are independently useful/requested by others.
3. Check whether the enlarged data supports a more refined PFAS class
   taxonomy (PubChem-hierarchy-based, see §2/§3 above) instead of just
   binary PFAS/non-PFAS — previously concluded there wasn't enough data for
   this; revisit now that dataset is bigger.
4. Benchmark on "Jonathan's data" (i.e. PFΔScreen author's dataset/lab —
   Jonathan Zweigle, cited throughout the ISEF paper as the PFΔScreen
   rule-based baseline).
5. Based on benchmark results, decide whether more experiments/refinement
   are needed.
6. If the model performs well, apply it to MassIVE data to find putative
   PFAS. A first version of **DreaMS-Mol** now exists (structure prediction
   from spectra) that could help interpret hits — currently positive-mode
   only.

## 6. Current focus (this session): Step 1 only

Explicitly scoped down — not touching the dataset yet (no Enveda-180 in
this pass), just the architecture:

- **Goal:** add an ionization-mode signal into `HalogenDetectorDreamsTest`
  without modifying DreaMS itself (no negative-mode DreaMS backbone exists
  yet — noted as a real gap, "replace later").
- **Data:** keep training/evaluating on the existing merged
  MassSpecGym + NIST20 + NIST-PFAS dataset used in the ISEF paper — this
  step is purely an architecture ablation vs. the existing binary baseline.
- **Approach:** write a *new* model class + *new* dataset subclass rather
  than editing `HalogenDetectorDreamsTest` / `TestMassSpecDataset` in
  place, so the original binary/spectral-only baseline stays runnable for
  comparison.
- **Known engineering blocker:** `TestMassSpecDataset.__getitem__` in
  `scripts/train_PFAS_model.py` currently drops `adduct` entirely
  ("removed adduct due to a str error") — the generic
  `torch.as_tensor(v, dtype=self.dtype)` cast loop chokes on the raw
  adduct string. Need to derive a numeric/categorical ion-mode feature from
  `metadata["adduct"]` *before* that cast loop (same pattern already used
  for the `item['F']` fluorine-presence flag). Reuse the existing
  `ionization_mode(adduct)` helper in
  `scripts/visualize_pfas_umap_ionization.py` (buckets into
  Positive/Negative/Unknown) for consistency, rather than re-deriving the
  regex logic.
- **Architecture options discussed** (fusing a categorical ion-mode signal
  with the 1024-d DreaMS embedding token):
  1. Concatenate raw feature + linear head (`Linear(1025,1)`) — cheapest,
     but a single linear layer can only add a constant shift from
     ion-mode; can't let polarity change how spectral features are
     weighted.
  2. Concatenate + small FFN (`Linear(1025,H) → ReLU → Linear(H,1)`) —
     user's original suggestion; nonlinearity lets ion-mode gate/interact
     with individual embedding dimensions, still minimal new parameters.
  3. FiLM-style conditioning — `nn.Embedding(k, 1024)` (x2 for scale
     `gamma` and shift `beta`) indexed by ion-mode, applied as
     `gamma[mode] ⊙ embedding + beta[mode]` before the head. Lets each
     polarity re-read the whole embedding, standard technique for
     conditioning a shared backbone on a discrete domain/style variable.
  4. Per-mode gated/expert heads — separate `Linear(1024,1)` per polarity,
     selected by mode. Equivalent to a `Linear(2048,1)` over
     `[h·mode, h·(1-mode)]`. Most flexible (fully decouples per-mode
     decision function) but each expert only trains on its polarity's
     data — risks starving the already-underrepresented positive-mode
     PFAS examples, which is the exact problem we're trying to fix, so
     this needs care (e.g. shared trunk + small per-mode residual, not a
     fully separate head).
  - Style note: use a small `nn.Embedding` lookup for the categorical
    ion-mode (supports a 3rd "Unknown" bucket cleanly) rather than a raw
    0/1 float, consistent with how `StandardMeta` in
    `massspecgym/data/transforms.py` already embeds `adduct`/instrument
    type elsewhere in this codebase.
- **Decision (updated 2026-07-11):** full writeup of the four options
  (concat+linear, concat+FFN, FiLM, gated heads) with diagrams, code
  sketches, parameter counts, and a recommendation was prepared for
  mentor review: `documentation/dreams-pfas-ion-mode-fusion-options.md`.
  Mentor confirmed wanting options B (concat+FFN) and C (FiLM) built
  overall; user asked to implement the simplest variant, **Option A
  (concat + single linear layer)**, first as a fast floor/sanity-check
  benchmark before investing in B/C. Implemented in this session — see
  §7.1 below.

## 7. Status

**Option A (concat + single linear layer) is implemented, locally
verified, and benchmarked** (2026-07-11/12) — see §7.1.1 for results:
recall improved over the ISEF baseline at the paper's chosen threshold,
precision roughly unchanged. Now adding per-ion-mode (pos/neg) breakdown
reporting per mentor request (§8). Options B (concat+FFN) and C (FiLM)
are the next planned architecture passes, reusing the same dataset/helper
code.

### 7.1 Option A implementation (done)

Plan used: `/Users/ramsindhu/.claude/plans/sprightly-painting-comet.md`
(local Claude Code plan file, not in the repo).

New/changed files:
- `massspecgym/data/transforms.py` — added
  `ion_mode_idx_from_adduct(adduct) -> int` (0=negative, 1=positive,
  2=unknown), mirroring the suffix logic already in
  `scripts/visualize_pfas_umap_ionization.py:ionization_mode()`.
- `massspecgym/models/pfas/ion_mode_linear.py` (new) —
  `HalogenDetectorDreamsIonModeLinear(HalogenDetectorDreamsTest)`. Replaces
  the parent's `Linear(1024,1)` head with `Linear(1024+3,1)` over
  `concat([embedding, one_hot(ion_mode, 3)])`. Overrides `forward`,
  `_step_bce_loss`, `_step_focal_loss`, `on_batch_end` to thread
  `batch["ion_mode"]` through; everything else (metrics, epoch-end
  logging, calibration curve/PR table saving) inherited unchanged.
- `massspecgym/models/pfas/__init__.py` — now exports
  `HalogenDetectorDreamsIonModeLinear` alongside the original
  `HalogenDetectorDreamsTest`.
- `scripts/train_PFAS_ion_mode_linear_model.py` (new) — near-duplicate of
  `scripts/train_PFAS_model.py`. New `IonModeMassSpecDataset` (same body as
  `TestMassSpecDataset` plus `item['ion_mode'] = ion_mode_idx_from_adduct(...)`
  computed before the tensor-cast loop). Same hyperparameters as the
  original script (threshold=0.2, alpha=0.25, gamma=0.75, lr=1e-5,
  batch_size=64, loss='focal'). New W&B project name
  (`HalogenDetection-FocalLoss-IonModeLinear-MergedMassSpecNIST20_NISTNew`).
  Real training data path (`.../merged_massspec_nist20_nist_new_with_fold.tsv`)
  is Lightning.ai-Studio-only and doesn't exist in this local checkout.

**Local verification performed** (env: conda `dreams_gradio`, which has
torch/matchms/dreams/massspecgym installed — the base env does not):
1. Unit-checked `ion_mode_idx_from_adduct` on `"[M+H]+"` (→1), `"[M-H]-"`
   (→0), `"weird"`/`None` (→2).
2. Unit-tested `HalogenDetectorDreamsIonModeLinear.forward()`/`backward()`
   with the real DreaMS backbone mocked out (avoids downloading the
   ~500MB Zenodo checkpoint for a pure architecture check) — confirmed
   output shape `[B]`, gradients populate on `lin_out` weight/bias, and
   `lin_out.in_features == 1027`.
3. Ran `IonModeMassSpecDataset` against the real local debug MGF
   (`data/debug/example_5_spectra.mgf`, 5 positive-mode spectra) — confirmed
   `ion_mode` derived correctly (all 1.0) and spec tensor shape intact.
4. Ran the real `MassSpecDataModule` + `collate_fn` (default_collate) over
   that debug set — confirmed the new `"ion_mode"` key collates into a
   proper `[B]` float tensor alongside existing keys, closing the
   collation-risk concern flagged during planning.
5. All 4 touched/added files (`transforms.py`, `ion_mode_linear.py`,
   `pfas/__init__.py`, `train_PFAS_ion_mode_linear_model.py`) compile
   cleanly (`py_compile`).

**Update (2026-07-12): the remote benchmark run is done** — see §7.1.1
below for results. (Originally this section noted the remote run as
outstanding; superseded now that results are back.)

### 7.1.1 Option A benchmark results (n=3 runs)

Training was run on the user's Lightning.ai Studio environment (real
DreaMS checkpoint + full merged MassSpecGym+NIST20+NIST-PFAS TSV). 3 runs'
`pr_table.csv` outputs were collected in
`~/Downloads/DreaMS-PFAS-Paper/OptionA-Results/` (outside the repo, not
committed to git) and aggregated (mean ± 2SE per threshold, thresholds 0.0
and 1.0 dropped) into
`~/Downloads/DreaMS-PFAS-Paper/OptionA-Results/pr_table_aggregated_mean_2se.csv`.

**Aggregated Option A results (n=3 runs):**

| Threshold | Precision % (mean ± 2SE) | Recall % (mean ± 2SE) | F1 (mean ± 2SE) |
|---|---|---|---|
| 0.1 | 90.7 ± 5.9 | 91.7 ± 0.4 | 0.911 ± 0.028 |
| 0.2 | 93.4 ± 4.0 | 91.5 ± 0.5 | 0.924 ± 0.018 |
| 0.3 | 94.9 ± 2.9 | 91.5 ± 0.5 | 0.931 ± 0.012 |
| 0.4 | 95.8 ± 2.3 | 89.0 ± 4.8 | 0.922 ± 0.023 |
| 0.5 | 96.7 ± 1.7 | 84.8 ± 8.8 | 0.902 ± 0.050 |
| 0.6 | 97.2 ± 2.2 | 72.5 ± 8.0 | 0.830 ± 0.057 |
| 0.7 | 97.5 ± 2.2 | 62.5 ± 0.1 | 0.762 ± 0.007 |
| 0.8 | 98.0 ± 2.1 | 59.2 ± 1.5 | 0.738 ± 0.010 |
| 0.9 | 98.8 ± 1.4 | 56.3 ± 1.3 | 0.717 ± 0.007 |

**Comparison to the ISEF baseline (binary/spectral-only model, Table 3,
n=5 runs)** at the paper's chosen threshold 0.2:

| Metric | Baseline (n=5) | Option A (n=3) | Delta |
|---|---|---|---|
| Precision | 93.8 ± 2.7 | 93.4 ± 4.0 | ~unchanged, well within error bars |
| Recall | 89.0 ± 1.6 | 91.5 ± 0.5 | **+2.5 pts, non-overlapping ±2SE bands, tighter spread** |
| F1 | 0.91 ± 0.02 | 0.924 ± 0.018 | +0.014 |

Precision is essentially flat vs. the baseline, but recall is meaningfully
higher — the ±2SE bands don't overlap (baseline [87.4, 90.6] vs. Option A
[91.1, 92.0]) — and more consistent across runs (SE shrank from ±1.6 to
±0.5). The gap is even larger at stricter thresholds, where the baseline's
recall falls off a cliff but Option A degrades more gradually:

| Threshold | Baseline recall | Option A recall |
|---|---|---|
| 0.3 | 86.6 ± 5.8 | 91.5 ± 0.5 |
| 0.4 | 81.3 ± 9.5 | 89.0 ± 4.8 |
| 0.5 | 73.3 ± 7.3 | 84.8 ± 8.8 |
| 0.6 | 62.2 ± 0.6 | 72.5 ± 8.0 |

Option A's peak F1 (0.931 at threshold 0.3) also exceeds the baseline's
peak (0.91).

**Caveat:** Option A is only n=3 runs vs. the baseline's n=5 — SE
estimates are noisier with fewer samples. Treat this as an encouraging
signal (consistent with the mentor's hypothesis that giving the model
explicit polarity information should help), not yet a conclusive result.
Two more runs would bring Option A to parity (n=5) for a cleaner
comparison — not yet done.

Given even the simplest fusion (concat + single linear layer, expected to
be a "weak floor" per the fusion-options doc) already shows a plausible
recall gain, this is a reasonable basis to proceed to Option B (FFN)
regardless of exact statistical significance here.

## 8. Per-ion-mode (pos/neg) breakdown reporting (done, 2026-07-12)

Mentor request: *"it would be good to see the performance split by ion
mode, separately for pos and neg after training."* Until this, all
validation reporting (`pr_table.csv`, calibration curve, TP/FN dumps) was
aggregated across all validation spectra regardless of polarity — there
was no way to see whether recall/precision differ between positive- and
negative-mode spectra, which is the central axis of this whole effort.

**What changed:**
- `massspecgym/models/pfas/base.py` (shared parent class,
  `HalogenDetectorDreamsTest`): `_reset_metrics_val` now also resets
  `self.all_ion_modes = []`; `on_batch_end` collects `batch.get("ion_mode")`
  per validation sample (or `None` placeholders when absent, keeping the
  list index-aligned with `all_true_labels`/`all_predicted_probs`). The
  threshold-sweep loop in `save_precision_recall_table` was factored into
  `_compute_pr_table(y_true, y_prob) -> pd.DataFrame` (no behavior change
  to the existing `pr_table.csv` output). `save_precision_recall_table`
  now additionally writes `pr_table_ion_mode_{negative,positive,unknown}.csv`
  for any mode with at least one sample, using the same threshold-sweep
  logic — silently skipped entirely when no ion_mode data was collected
  (fully backward compatible with datasets/models that don't have it).
- `massspecgym/models/pfas/ion_mode_linear.py`
  (`HalogenDetectorDreamsIonModeLinear`): its `on_batch_end` override
  (which already has `ion_mode` in scope) now also appends it to
  `self.all_ion_modes`.
- `scripts/train_PFAS_model.py` (the original ISEF baseline script):
  `TestMassSpecDataset` now also derives
  `item['ion_mode'] = ion_mode_idx_from_adduct(metadata["adduct"])`,
  mirroring the fix already in `IonModeMassSpecDataset`. This is purely
  additive — the baseline model's `forward()`/loss/predictions don't
  consume `ion_mode` at all, so training behavior is unchanged — but it
  means the **original baseline can now also produce a per-mode
  breakdown**, which is the strongest comparison for the paper: whether
  the baseline's recall is worse on positive-mode spectra, and whether the
  ion-mode-aware models close that gap.

**Local verification:** unit-tested `save_precision_recall_table` with
mocked DreaMS backbone across 3 scenarios — mixed 0/1/2 ion modes (writes
all 3 per-mode files + aggregate), all-`None` ion modes (writes only the
aggregate, no per-mode files, confirming backward compatibility), and
positive-mode-only data (writes aggregate + positive file only, no
negative/unknown). Also re-ran the real dataset → dataloader → model →
`save_precision_recall_table` pipeline end-to-end against the local debug
MGF (`data/debug/example_5_spectra.mgf`, all positive-mode) and confirmed
the same positive-only file pattern through the real (non-mocked) data
path. All touched files compile cleanly.

**Not done locally:** an actual per-mode breakdown on the real merged
training data (to see the real positive- vs. negative-mode
precision/recall gap for both the baseline and Option A) — needs a
Lightning.ai Studio run, same limitation as before. Next step for the
user: rerun training (baseline and/or Option A) remotely and check for
`pr_table_ion_mode_positive.csv` / `pr_table_ion_mode_negative.csv` in the
output alongside the existing `pr_table.csv`.

**Naming note (bug found + fixed 2026-07-12):** `save_precision_recall_table`
now writes the aggregate table as `pr_table_{self.seed}.csv` (a change
made independently to support the multi-seed run loop, distinguishing
output across runs like `pr_table_0.csv`, `pr_table_1783816586.csv`, etc. —
matches the files aggregated in §7.1.1). The per-ion-mode files added
above initially did NOT include this seed suffix, which would have caused
every multi-run per-mode CSV to silently overwrite the previous run's —
defeating the purpose for the same reason the aggregate table needed the
seed suffix. Fixed while verifying Option B: per-mode filenames are now
`pr_table_ion_mode_{mode_name}_{self.seed}.csv`, consistent with the
aggregate table's naming.

## 9. Option B (concat + FFN) implementation (done, 2026-07-12)

Built as a **fully separate model class + training script**, per explicit
request, so Option A (running remotely) and Option B can be launched
independently without either disturbing the other. No existing
Option-A file was touched except the one-line bug fix above.

- `massspecgym/models/pfas/ion_mode_ffn.py` (new) —
  `HalogenDetectorDreamsIonModeFFN(HalogenDetectorDreamsTest)`. Same
  concat-with-one-hot-ion-mode input as Option A, but replaces the single
  `Linear(1024+3,1)` head with `Linear(1024+3,128) → ReLU → Linear(128,1)`
  (`hidden_dim=128`, configurable). `_step_bce_loss`/`_step_focal_loss`/
  `on_batch_end` mirror `ion_mode_linear.py`'s bodies exactly (same
  ion_mode plumbing, same per-mode PR-table reporting inherited from the
  shared base class — no changes needed there for Option B to get it).
- `massspecgym/models/pfas/__init__.py` — additive export of the new
  class alongside the existing two.
- `scripts/train_PFAS_ion_mode_ffn_model.py` (new) — mirrors
  `train_PFAS_ion_mode_linear_model.py`'s current state (seed loop,
  `loss='bce'`) exactly, swapping in the FFN model
  (`hidden_dim=128` passed explicitly) and a distinct W&B project name
  (`PFASDetection-IonModeFFN-MergedMassSpecNIST20_NISTNew`) so run
  history doesn't collide with Option A's.

**Local verification:** unit-tested `forward()`/`backward()` with a mocked
DreaMS backbone (`fc1.in_features==1027`, `fc1.out_features==128`,
`fc2.in_features==128`, gradients populate all 4 fc1/fc2 params). Reran
the real dataset→dataloader→model→`save_precision_recall_table` pipeline
against the local debug MGF and confirmed `pr_table_0.csv` +
`pr_table_ion_mode_positive_0.csv` only (no negative/unknown, all-positive
debug set) — same pattern as Option A's equivalent check. Confirmed via
`git status`/`git diff --stat` that only the intended files changed
(2 new files, 1 additive `__init__.py` edit, 1 one-line bug fix in
`base.py` shared by both options).

**Not done locally:** full-scale benchmark training against the real
merged dataset — needs the user's remote environment, run independently
of the in-progress Option A run.

## 10. Real per-ion-mode results from Option A's first full run (2026-07-12)

Real training on `merged_massspec_nist20_nist_new_env_pfas_with_fold.tsv`
(the actual 737,876-row merged file, obtained from the user's Downloads,
confirmed to match the ISEF paper's totals) produced the first genuine
per-ion-mode breakdown
(`pr_table_ion_mode_{negative,positive,unknown}_1783882568.csv`). Findings:

- **Negative mode (val n=920):** precision 1.0 at every threshold, high
  recall. Looks "perfect," but this bucket is **100% PFAS-positive** — it
  is entirely NIST-PFAS's small, homogeneous reference set, so this is a
  trivial classification task, not evidence of genuine generalization.
- **Positive mode (val n=45,479, prevalence 0.12%):** precision and
  recall **collapse to exactly 0** at any threshold ≥ 0.1 — the
  ion-mode-aware Option A model is not detecting real positive-mode PFAS
  at all.
- **"Unknown" mode (val n=97,089, prevalence 0.026%):** same collapse to
  0. Investigated why this bucket is so large (67.7% of validation) and
  found it is **not** genuinely ambiguous-polarity spectra — it is
  **482,381 rows (65% of the whole dataset) where the `adduct` column is
  literally `NaN`, and every one of those NaN rows is from NIST20**. The
  only non-null adduct values in the entire file are `[M+H]+`/`[M+Na]+`
  (MassSpecGym) and `[M–H]–` (NIST-PFAS, note: en-dash, not ASCII
  hyphen — `ion_mode_idx_from_adduct` already handles both). NIST20's
  real polarity metadata simply isn't present in this merged TSV's
  `adduct` column.
- **Attempted to recover NIST20's real polarity via mass-defect
  arithmetic** (compute each molecule's monoisotopic mass from its
  `formula` column, diff against `precursor_mz`, match against known
  adduct mass shifts). Result: only ~47% of a 20k sample cleanly matched
  a known shift (41.7% `[M+H]+`, 4.0% `[M+Na]+`, 1.3% `[M+NH4]+`, 0.07%
  `[M+K]+`, 0.02% `[M-H]-`) — the rest (52.8%) didn't match anything,
  with wildly scattered diffs. Also found `precursor_formula` is a
  byte-for-byte duplicate of `formula` for every NIST20 row, suggesting
  it was never actually computed (likely a placeholder), which further
  undermines trusting this arithmetic for the majority of rows. Where it
  *did* cleanly match, the signal skewed positive-mode, not negative —
  the opposite of what would help the paper's skew narrative. Conclusion:
  **mass-defect arithmetic is not reliable enough to relabel NIST20's
  polarity** — the correct fix is to go back to whoever generated this
  merged TSV (likely Roman) and re-extract the `Ion_mode`/`Precursor_type`
  field from NIST20's original (licensed) source records, joined back in
  via `identifier`. No raw NIST20 source file or merge script was found
  in this local repo checkout, so that re-extraction has to happen
  wherever the original merge was done.

**Ion-mode distribution among PFAS-positive training examples (n=19,194
total):**

| Ion mode | Count | % |
|---|---|---|
| negative | 18,971 | 98.8% |
| unknown (NIST20, missing adduct) | 149 | 0.8% |
| positive | 74 | 0.4% |

**This is the sharpest evidence yet for the core problem.** Only 74
positive-mode PFAS examples exist in the entire 594,388-row training set.
No architecture change (Option A/B/C) can be expected to fix positive-mode
recall on its own with this little real signal to learn from — the
earlier "+2.5pt recall" aggregate benchmark result (§7.1.1) was almost
certainly driven by the easy, homogeneous, 920-example negative-mode
validation bucket, not genuine improvement on positive-mode detection.
This strengthens the case that **more positive-mode PFAS training data
(Enveda-180, roadmap step 2) is likely necessary, not just an
architecture fix**, before positive-mode recall can meaningfully improve.

**Open question for the mentor:** does Roman (or whoever built this
merged TSV) have access to NIST20's original per-spectrum `Ion_mode`
field, so real polarity labels can be backfilled for the 482k
currently-"unknown" rows instead of leaving them unlabeled/guessed?

## 11. Enveda-180 dataset preparation (roadmap step 2, 2026-07-12)

Moved to the next roadmap step: since architecture changes alone can't be
properly evaluated with only 74 positive-mode PFAS training examples,
built tooling to prepare **Enveda-180**
(https://zenodo.org/records/20436851) as a same-schema TSV the existing
training scripts can point at directly.

**Real facts about Enveda-180** (confirmed by downloading and parsing a
real partial sample, not assumed from the Zenodo page description alone):
1,158,293 spectra, 184,330 unique compounds (Enamine HLL-200 drug-like
screening collection), both polarities (ESI+ 77%, ESI− 23%), explicit
`ionmode` ground-truth field per spectrum (no heuristic needed), Bruker
timsTOF Pro 2 instrument. Available as MGF/MSP/JSONL/Parquet,
filtered/unfiltered. Decided to use the filtered JSONL variant.

**Critical finding, checked before building anything:** of 1,821 unique
molecules in a ~1% sample, 21.6% contain some fluorine but **zero** meet
the existing PFAS-chain labeling rule (`MolToPFASVector.has_pf_chain_ge_2`,
≥2 connected CF2/CF3) — consistent with Enveda-180's fluorine content
being drug-like isolated-F/CF3 substituents, not environmental chain-PFAS.
This is only a 1% sample, so **not conclusive** — a full-dataset gate
check is required (and was explicitly sequenced first) before deciding
whether Enveda-180 actually helps the chain-PFAS retraining goal, versus
mainly helping the separate plain-fluorine-detector /
future-drug-like-PFAS-taxonomy work already on the roadmap.

**Built two scripts** (decisions: separate file not concatenated;
stratified random split by is_PFAS+ionmode at the molecule level, not a
full Murcko scaffold split; filtered source):
- `scripts/check_enveda180_pfas_prevalence.py` — the gate check. Streams
  the JSONL, dedupes by inchikey, computes PFAS-chain prevalence via the
  same existing labeling rule, breaks down by real ion mode. Must be run
  against the full file (~300MB+) wherever that's feasible — a direct
  download attempt from this sandboxed session was blocked by the
  harness, so this needs to run on the user's machine or Lightning
  Studio.
- `scripts/prepare_enveda180_dataset.py` — the conversion. Maps Enveda-180
  JSONL fields to the existing TSV's exact column names (only
  `identifier`/`mzs`/`intensities`/`smiles`/`adduct`/`precursor_mz`/`fold`
  are actually consumed by the PFAS training/eval pipeline — confirmed by
  grepping every place each column is read; `formula`/`precursor_formula`/
  `parent_mass`/`instrument_type`/`collision_energy`/`simulation_challenge`/
  `name` are carried through best-effort or left blank since unused).
  Molecule-level stratified train/val split. Adds one bonus column not in
  the original schema, `ion_mode_true`, carrying Enveda's real ground
  truth through for future validation use.

**Local verification (dry run against the same partial sample used for
the gate-check finding above):**
- Gate-check script reproduced the manual analysis exactly (0 PFAS-chain
  molecules, 21.58% fluorine, correct ion-mode split).
- Conversion script produced a valid output TSV.
- **Loaded the output directly via the real `MassSpecDataset` and
  `IonModeMassSpecDataset` with zero code changes** — confirms the schema
  match works end-to-end, including `SpecTokenizer` producing correctly
  shaped spectra and the dataset's own `fold` column driving
  `MassSpecDataModule`'s train/val split with no external `split_pth`
  needed.
- **Bonus validation: cross-checked `ion_mode_idx_from_adduct`'s heuristic
  against Enveda-180's real `ionmode` ground truth across 2,000 samples —
  0 mismatches (0.00%)**. This is reassuring evidence the heuristic itself
  is sound, at least for clean/standard adduct strings like Enveda-180's;
  doesn't resolve the separate, larger problem of NIST20's `adduct` field
  being entirely missing (section 10).

**Not done locally:** the full-dataset gate check and full conversion
(1.15M spectra) — needs to run wherever the full Enveda-180 file can be
downloaded (blocked in this sandbox). Next step for the user: run
`check_enveda180_pfas_prevalence.py` on the full
`enveda-180-filtered.jsonl.gz` first; only proceed to
`prepare_enveda180_dataset.py` if that shows non-trivial PFAS-chain
prevalence.

## 12. Full-dataset gate check result (2026-07-12): Enveda-180 does NOT contain meaningful chain-PFAS content

User downloaded the full **unfiltered** `enveda-180.jsonl.gz` (1.8GB,
matches Zenodo's listed size) to `~/Downloads/`. Verified gzip integrity,
then ran `scripts/check_enveda180_pfas_prevalence.py` against the real,
complete file (not a sample):

- **1,158,293 spectra parsed, 0 JSON errors** (matches Zenodo's stated
  total exactly — confirms this is the full, uncorrupted dataset).
- **184,330 unique molecules** (also matches exactly).
- **22.95% of unique molecules contain some fluorine** (42,310 molecules)
  — consistent with the ~21.6% found in the earlier 1% sample and the
  user's "1 in 4" estimate.
- **Only 1 unique molecule (0.0005%) meets the existing PFAS-chain
  labeling criterion** (≥2 connected CF2/CF3), contributing just 4
  spectra total (all ESI Positive, 0 ESI Negative) out of 1,158,293.

**This confirms the 1% sample's finding was representative, not sample
noise — it's now a definitive, full-dataset result, not an estimate.**

**Implication:** Enveda-180 does **not** solve the specific problem this
whole effort started from (74 positive-mode chain-PFAS training examples
is too few). Its ~23% fluorine content is essentially all drug-like
isolated-F/CF3 substituents, not environmental perfluoroalkyl chains —
consistent with it being an Enamine drug-like screening collection, not
an environmental-chemistry library. It remains valuable for the
**separate** roadmap items already planned (a general fluorine detector,
and the future drug-like isolated-CF2/CF3 taxonomy work from the ISEF
paper's own §4.4 Future Work), just not for directly fixing positive-mode
chain-PFAS scarcity.

**Open question for the user/mentor:** given this result, options going
forward include (a) still convert Enveda-180 and use it to retrain the
*plain fluorine detector* (already on the roadmap, independent of the
chain-PFAS binary classifier), (b) look for a different data source with
genuine positive-mode environmental chain-PFAS content, or (c) revisit
whether the strict "≥2 connected CF2/CF3" labeling rule itself is too
narrow and should be loosened/reconsidered given how little data meets it
anywhere. Not yet decided.

## 13. Drug-like PFAS (isolated CF2/CF3) EDA on Enveda-180 (2026-07-12)

Given §12's dead-end on chain-PFAS, checked the broader **OECD PFAS
definition** (Wang et al. 2021, "A New OECD Definition for Per- and
Polyfluoroalkyl Substances," *Environmental Science & Technology*, DOI
10.1021/acs.est.1c06896: any chemical with at least one saturated CF2 or
CF3 moiety, including isolated/non-chain groups) against the full
Enveda-180 dataset, reusing the existing `MolToIsolatedCF2Vector`/
`MolToIsolatedCF3Vector` transforms already in this repo
(`massspecgym/data/transforms.py`) — these already implement this exact
OECD taxonomy (see `documentation/DreaMS-PFAS_training_and_evaluation.md`'s
Type 1/Type 2 table). New script: `scripts/check_enveda180_drug_like_pfas.py`.

**Full-dataset result (184,330 unique molecules, 1,158,293 spectra):**

| Category | Unique molecules | % |
|---|---|---|
| Chain-PFAS (≥2 connected CF2/CF3) | 1 | 0.0005% |
| Isolated-CF2-only | 897 | 0.487% |
| Isolated-CF3-only | 11,112 | 6.028% |
| Both CF2-only AND CF3-only | 0 | 0% |
| **Drug-like PFAS (CF2-only OR CF3-only)** | **12,009** | **6.515%** |
| **Total OECD PFAS (any type)** | **12,010** | **6.515%** |

**Spectrum-level, by ion mode — the encouraging part:**

| Ion mode | n spectra | Drug-like-PFAS-positive | % |
|---|---|---|---|
| ESI Positive | 891,903 | 53,406 | 5.99% |
| ESI Negative | 266,390 | 23,386 | 8.78% |

**Unlike chain-PFAS, drug-like PFAS is well-represented in BOTH
polarities** — 53,406 positive-mode and 23,386 negative-mode spectra,
roughly proportional to each mode's overall size. This is the opposite of
the chain-PFAS problem (98.8% negative-mode skew, §10 for training set).

**Implication:** Enveda-180 is not useless for PFAS work after all — it's
just the wrong tool for the *current binary chain-PFAS classifier*. It's
a strong candidate dataset for the **multi-class taxonomy** direction
already on the roadmap (roadmap step 3; ISEF paper §4.4 Future Work's
"drug-like fluorinated compounds" class) — 12,009 unique drug-like-PFAS
molecules with good positive/negative balance is a real, usable training
signal for extending `MolToFluorinatedTypeVector`'s existing 4-class
scheme (environmental PFAS chain / isolated CF2-CF3 / other F / non-F)
beyond the current binary model.

**Environment note:** the `dreams_gradio` conda env used for all local
verification this session has a **stale, non-editable `massspecgym`
install** in its `site-packages` that's missing `MolToIsolatedCF2Vector`/
`MolToIsolatedCF3Vector`/`ion_mode_idx_from_adduct` (added to the repo
this session) — it shadowed the live repo source for this script until
worked around with `PYTHONPATH=/Users/ramsindhu/MassSpecGym` prepended to
the invocation. Not fixed permanently (would need `pip install -e .` in
that env, which wasn't done since it modifies the environment) — worth
fixing before any future script in that env needs a class not present in
whatever snapshot was last pip-installed there.

## 14. Combined NIST-PFAS + Enveda-180 PFAS-type distribution, by ion mode (2026-07-12)

User asked: combine **only NIST-PFAS** (excluding NIST20 and
MassSpecGym) from the existing merged TSV with the full Enveda-180
dataset, and report chain-PFAS / isolated-CF2 / isolated-CF3 / drug-like
PFAS distribution cut by ion mode. New script:
`scripts/check_combined_nistpfas_enveda180_distribution.py`.

**Identifying NIST-PFAS:** confirmed in section 10 that NIST-PFAS = the
negative-ion-mode subset of the merged TSV (24,391 rows, exact match to
the ISEF paper's Table 2 total, zero crossover with MassSpecGym/NIST20).

**Bug found + fixed during this analysis:** the merged TSV's stored
`inchikey` column is **entirely NaN** for this subset (0 non-null out of
24,391 rows) — deduping by it collapsed all NIST-PFAS molecules into 1.
Fixed by computing inchikey from SMILES via the same helper the rest of
this codebase already uses for this exact situation
(`massspecgym.utils.smiles_to_inchi_key`, used in
`MassSpecDataset.compute_mol_freq` when inchikey is missing). After the
fix: 103 unique NIST-PFAS SMILES → 94 unique chain-PFAS molecules, which
matches the ISEF paper's Table 2 NIST-PFAS unique-PFAS-molecule count
(91 train + 3 val = 94) exactly — good independent confirmation the fix
is correct.

**Combined unique molecules (n=184,433 -- NIST-PFAS 103 + Enveda-180
184,330, negligible overlap):**

| Category | Count | % |
|---|---|---|
| Chain-PFAS (≥2 connected) | 95 | 0.052% |
| Isolated-CF2-only | 899 | 0.487% |
| Isolated-CF3-only | 11,121 | 6.030% |
| Drug-like PFAS (either) | 12,018 | 6.516% |
| Total any-PFAS | 12,113 | 6.568% |

**Spectrum-level, cut by ion mode:**

| Category | Negative (n=290,781: 24,391 NIST-PFAS + 266,390 Enveda) | Positive (n=891,903, all Enveda) |
|---|---|---|
| Chain-PFAS | 19,891 (6.84%) | 4 (0.0004%) |
| Isolated-CF2-only | 2,650 (0.91%) | 3,955 (0.44%) |
| Isolated-CF3-only | 25,879 (8.90%) | 49,451 (5.54%) |
| Drug-like (either) | 27,886 (9.59%) | 53,406 (5.99%) |
| Total any-PFAS | 47,777 (16.43%) | 53,410 (5.99%) |

**Interpretation:**
- **Chain-PFAS remains just as skewed as before — this combination does
  not fix positive-mode chain-PFAS scarcity.** Still only 4 positive-mode
  chain-PFAS spectra total (unchanged, since Enveda-180 contributes only
  1 chain-PFAS molecule). The 19,891 vs. 4 spectrum-count gap is a
  reference-library replication artifact (NIST-PFAS's 94 chain-PFAS
  molecules have many replicate spectra each), not a molecule-diversity
  fix — same underlying dead-end as sections 12/13, confirmed again at
  the combined-dataset level.
- **Drug-like PFAS (isolated CF2/CF3) is reasonably balanced across
  polarity** — 9.59% negative vs. 5.99% positive, under 2x apart, a world
  away from chain-PFAS's ~17,000x spectrum-count skew. This signal comes
  almost entirely from Enveda-180 (NIST-PFAS is a chain-PFAS reference
  set, contributing little to the isolated-CF2/CF3 categories).
- **Bottom line:** this combined dataset doesn't help the current binary
  chain-PFAS classifier's positive-mode recall problem, but is a solid,
  reasonably balanced training set for the multi-class taxonomy direction
  (chain-PFAS / drug-like-PFAS / other-F / non-F) already on the roadmap.

## 15. Roman's feedback + multi-head model brainstorm, and the combined dataset file (2026-07-17)

Roman's message: negative-mode chain-PFAS detection isn't a hard problem
given the precision/recall already achieved; extend to different PFAS
types and positive-mode data; label the combined dataset per the
**PubChem PFAS classification** so every F-containing molecule gets some
label, then train a general fluorine predictor.

**Reference checked:** the PubChem PFAS tree (Wang et al. context aside,
this is the PMC10634333 "PFAS in PubChem" article) splits into 6
top-level sections (OECD PFAS definition / PFAS breakdowns by chemistry /
organofluorine compounds / other diverse fluorinated compounds / PFAS
collections / regulatory collections); under "PFAS breakdowns by
chemistry," molecules are organized by functional group, connectivity,
and chain length. Enveda-180 conveniently carries a `pubchem_cid` field
per molecule already, which could support a real PubChem lookup later
(not done in this pass).

**Modeling brainstorm outcome:** recommended a shared-trunk, multi-head
architecture over one flat multi-class softmax — Head 1 (has-F, trained
on everything), Head 2 (OECD-PFAS-type vs. other-fluorinated, trained
where labeled), Head 3 (multi-label sigmoid: `has_isolated_cf2` /
`has_isolated_cf3` / `has_chain_pfas` — NOT mutually exclusive, since a
molecule can have more than one simultaneously, confirmed as a real
possibility, not just theoretical, in this session's own data — see
below), each with a masked per-sample loss so partially-labeled data
still contributes to whichever heads it has labels for. Same DreaMS +
ion-mode-concat trunk as Option A/B. **Recommended Option B (FFN) over
Option A (linear) for the fusion mechanism**, reasoning: the task is now
multi-faceted (3 heads, not 1 binary decision) and the combined dataset's
drug-like-PFAS class is genuinely polarity-balanced (unlike chain-PFAS),
giving a nonlinear fusion layer real signal to exploit rather than riding
on one trivially-easy, skewed bucket the way Option A's earlier "recall
improvement" did.

### 15.1 Combined dataset file built: `combined_nistpfas_enveda180_with_fold.tsv`

New script: `scripts/prepare_combined_nistpfas_enveda180_dataset.py`.
Merges NIST-PFAS (negative-mode subset of the merged TSV, inchikey
recomputed from SMILES since the stored column is NaN for this subset --
same fix as section 14) with the full Enveda-180 dataset into one TSV in
the existing schema plus 4 bonus columns: `ion_mode_true`,
`has_chain_pfas`, `has_isolated_cf2`, `has_isolated_cf3` (precomputed via
the existing `MolToPFASVector`/`MolToIsolatedCF2Vector`/
`MolToIsolatedCF3Vector` transforms, reusable later for the multi-head
model without recomputing RDKit classification each time).

**Fold assignment:** per explicit user choice, redone fresh across the
*whole* combined pool (not preserved per-source) -- **so any future runs
on this file are not directly comparable to the Option A/B benchmark
numbers already reported in section 7.1.1**, which trained against
NIST-PFAS's original split. The stratified split fell back to an
unstratified random split (one singleton stratum in the fine-grained
`(has_chain, drug_like, has_pos, has_neg)` stratification key), but
checked afterward and the resulting proportions landed close to the
target 20% val fraction anyway for every rare class (chain-PFAS 21% val,
isolated-CF2 18% val, isolated-CF3 19.5% val) -- no category got
starved.

**Output:** 1,182,684 total spectra (24,391 NIST-PFAS + 1,158,293
Enveda-180), 184,433 unique molecules -- written to
`~/Downloads/DreaMS-PFAS-Paper/combined_nistpfas_enveda180_with_fold.tsv`.
Sanity-checked: unique-molecule counts (95 chain-PFAS, 899 isolated-CF2,
11,121 isolated-CF3, 12,018 drug-like) match section 14's numbers
exactly, confirming the new script is correct.

**Local verification:** dry-run with `--limit-enveda 20000` (chain-PFAS
count matched NIST-PFAS's known 94/95 exactly), then loaded the dry-run
output via the real `MassSpecDataset` and `IonModeMassSpecDataset` +
`MassSpecDataModule` -- confirmed the extra bonus columns don't break
compatibility with the existing training pipeline (no code changes
needed, same as the Enveda-180-only conversion). Full run then executed
against the real, complete source files.

### 15.2 Complete EDA on the combined dataset (mutually-exclusive partition, new this pass)

Previous EDAs (sections 13/14) reported chain-PFAS/isolated-CF2/
isolated-CF3/drug-like counts but never characterized the remaining
"other fluorinated" (has F, meets none of those criteria -- e.g. aromatic
C-F, single CHF/CHF2) or "non-fluorinated" buckets. Completed here by
reading the written file's bonus columns + a SMILES-based `has_F` check
(same regex convention already used in `TestMassSpecDataset`/
`IonModeMassSpecDataset`'s `item['F']` field, for consistency with the
existing training pipeline).

**Unique-molecule level (n=184,433), mutually exclusive partition:**

| Category | Count | % |
|---|---|---|
| Chain-PFAS | 95 | 0.052% |
| Drug-like isolated (CF2 or CF3) | 12,018 | 6.516% |
| Other fluorinated (has F, neither of the above) | 30,300 | 16.429% |
| Non-fluorinated | 142,020 | 77.004% |

Non-exclusive detail: 899 molecules have isolated-CF2 (any), 11,121 have
isolated-CF3 (any), and **2 molecules have both simultaneously** -- a
small revision from section 13's "0 found," which only checked
Enveda-180 alone (184,330 molecules); these 2 come from the marginal
NIST-PFAS additions (9 of NIST-PFAS's 103 molecules aren't chain-PFAS,
likely internal standards/calibration compounds in that reference
library).

**Spectrum level, cut by ion mode (n=1,182,684 total):**

| Category | Negative (n=290,781) | Positive (n=891,903) |
|---|---|---|
| Chain-PFAS | 19,891 (6.84%) | 4 (0.0004%) |
| Drug-like isolated | 27,886 (9.59%) | 53,406 (5.99%) |
| Other fluorinated | 49,595 (17.06%) | 139,258 (15.61%) |
| Non-fluorinated | 193,409 (66.51%) | 699,235 (78.40%) |

**Read on this table for the multi-head model design (section 15):**
Head 1 (has-F) has abundant, well-balanced positive/negative examples in
every non-non-fluorinated bucket (~22-33% of spectra in either mode).
Head 2 (OECD-PFAS-type vs. other-F) and Head 3 (subtype) both have
reasonable data in both polarities **except** chain-PFAS specifically,
which remains almost exclusively negative-mode (unchanged from every
earlier finding this session) -- so the multi-head model should be
expected to learn the drug-like/other-F/non-F distinctions well across
both polarities, but chain-PFAS subtype prediction will likely remain
recall-limited on positive-mode spectra regardless of architecture,
simply from lack of positive examples (4 spectra total).

**Next step (not yet started):** build the multi-head model itself
(step 2 of the plan agreed with the user), using Option B/FFN fusion per
the brainstorm above.

### 15.3 Settled: chain-PFAS takes priority over co-occurring isolated groups (2026-07-17)

Checked overlap between `has_chain_pfas` and `has_isolated_cf2`/
`has_isolated_cf3` in the combined dataset: **0 of 95 chain-PFAS
molecules** have either isolated flag set. This isn't necessarily because
no chain-PFAS molecule has a separate, unconnected isolated CF2/CF3 group
elsewhere -- `_has_isolated_cf2_only`/`_has_isolated_cf3_only`
(`massspecgym/data/transforms.py`) require **zero PF-carbon-to-PF-carbon
bonds anywhere in the whole molecule**, so any chain (which has such a
bond by definition) forces both isolated flags to `False` regardless of
whether an unrelated isolated group also exists elsewhere in the same
molecule -- the function can't distinguish "no isolated group" from "has
an isolated group but also has a chain."

**Decision: this is fine as-is, not a bug to fix.** User's reasoning: a
molecule containing any chain-like (multi-connected) PFAS is more likely
to be environmentally relevant PFAS, so chain-PFAS should take priority
in classification regardless of co-occurring isolated groups. This
matches the priority ordering already used elsewhere in this codebase
(`MolToFluorinatedTypeVector`: chain > isolated > other-F > non-F), so no
code changes needed -- the existing `has_chain_pfas`/`has_isolated_cf2`/
`has_isolated_cf3` bonus columns in the combined dataset already reflect
this intended semantics, and Head 3 of the planned multi-head model can
use them as-is (CF2/CF3 remain co-occurrable with *each other*, just not
with chain, by design).

## 16. Real Murcko Histogram Split implemented, replacing the stopgap random split (2026-07-17)

Replaced the combined dataset's stopgap stratified-random fold assignment
with the actual methodology referenced throughout this project --
https://dreams-docs.readthedocs.io/en/latest/tutorials/murcko_hist_split.html
-- reusing `dreams.algorithms.murcko_hist.murcko_hist.{murcko_hist,are_sub_hists}`
directly (already installed as a `dreams` package dependency, not
reimplemented; not exported from the package's `__init__.py` but
importable via the direct submodule path). `scripts/prepare_combined_nistpfas_enveda180_dataset.py`'s
`assign_folds()` now implements the tutorial's exact algorithm: group
unique molecules by Murcko histogram, sort by group size descending,
walk from the median-frequency group toward the most-frequent one,
assigning each group to val unless it's structurally close
(`are_sub_hists`, k=3, d=4) to an already-assigned val group, until
cumulative val molecule count exceeds `val_mols_frac` (tutorial default
0.15); everything less frequent than the median is bulk-assigned to
train.

**Result:** 184,433 unique molecules -> 101 distinct Murcko histograms.
Molecule-level split: 75.92% train / 24.08% val (tutorial's own
MassSpecGym reference: 80.55%/19.45% -- in the right ballpark, exact
ratio is inherently approximate since whole histogram-groups move as a
unit).

### 16.1 Critical finding: chain-PFAS collapsed almost entirely into val

Per-category spectrum-level breakdown (added to the script's output,
per user request to check every category's train/val representation):

| Category | Train | Val |
|---|---|---|
| Chain-PFAS | 4 (0.02%) | 19,891 (99.98%) |
| Drug-like isolated | 55,263 (67.98%) | 26,029 (32.02%) |
| Other fluorinated | 147,526 (78.12%) | 41,327 (21.88%) |
| Non-fluorinated | 665,786 (74.59%) | 226,858 (25.41%) |

**This confirms the acyclic-Murcko-scaffold risk flagged before
implementation.** Murcko scaffolds are defined by ring systems -- NIST-
PFAS's chain-PFAS reference molecules are mostly acyclic perfluorocarbon
chains (no rings at all), so they collapse into the same (empty)
histogram group(s). Whichever fold that group lands in gets nearly all
of it. Here, it landed almost entirely in **val**, leaving only 4
chain-PFAS spectra in train -- essentially no chain-PFAS training signal
left for any future model trained on this exact split, even though val
now has abundant chain-PFAS examples (19,891) to evaluate against. A
smaller dry run (--limit-enveda 20000) showed the same pattern (97.31%
val), confirming this is a systematic effect of the methodology on this
particular dataset, not a fluke of the full run.

**Not yet resolved with the user** -- flagged here for visibility. Chain
-PFAS's virtual disappearance from train is a real problem for any model
that needs to learn chain-PFAS detection using this split, orthogonal to
the split being technically "leak-free."

### 16.2 Tanimoto similarity validation (real methodology, replacing evaluate_split)

Attempted to reuse `dreams.utils.data.evaluate_split` (the tutorial's own
validation function, Code Cell 10 / ISEF paper Figure 4) directly, but
found two real problems at this dataset's scale, both confirmed
empirically, not assumed:

1. **A genuine bug:** `evaluate_split`'s internal loop only computes
   similarities for whichever fold name happens to be *last* in
   `df[fold_col].unique()` (row-order dependent) -- it does not compute
   both train and val despite the tutorial's usage implying it does. On
   the dry-run data, this returned a `'train'` key holding **train-vs-
   train self-similarity** (uniformly 1.0 for every value) instead of the
   intended val-vs-train comparison -- confirmed by seeing exactly 1.0
   for every one of 2,365 "validation" values, an impossible result for a
   real cross-fold comparison.
2. **Too slow at this scale regardless:** its Python-level per-pair
   `DataStructs.FingerprintSimilarity` loop is O(V x T) individual calls
   -- 44,415 val x 140,018 train = ~6.2 billion comparisons, which did not
   finish in several minutes even with 4 workers.

**Resolution:** `scripts/plot_combined_dataset_tanimoto_split.py`
reimplements the same statistic without depending on `evaluate_split`,
using RDKit's vectorized `DataStructs.BulkTanimotoSimilarity` (one call
per val molecule against the whole train reference set, C++ vectorized
rather than a Python loop) plus a configurable train-side subsample
(default 20,000, for speed at this scale) via `dreams.utils.mols.morgan_fp`
for fingerprints (same fingerprinting function `evaluate_split` uses
internally).

**Result (n=44,415 val molecules, train reference subsampled to
20,000):** mean similarity 0.445, median 0.439, max 0.956, **0
near-duplicate (>=0.99) matches** -- a healthy, no-leakage distribution,
comparable in shape to the ISEF paper's own Figure 4. Plot saved to
`~/Downloads/DreaMS-PFAS-Paper/tanimoto_split_quality.png`.

### 16.3 Verification

Confirmed the regenerated combined TSV (same schema, only `fold` values
changed) still loads correctly through the real `IonModeMassSpecDataset`
+ `MassSpecDataModule` pipeline with zero code changes: 1,182,684 total
spectra, 868,579 train / 314,105 val (73.4%/26.6% at the spectrum level).

## 17. Fix: preserve NIST-PFAS's original fold, only Murcko-split Enveda-180 (2026-07-17)

User asked to check NIST-PFAS's fold assignment in the *original* source
merged TSV, to see if it could just be reused instead of letting the
fresh split assign it. Checked directly: **23,471 train / 920 val
spectra (96.2%/3.8%), 100 train / 3 val unique molecules (97.1%/2.9%)**
-- the same split the ISEF paper's own Table 3 metrics were computed
against. This is the opposite profile from section 16.1's problem (chain-
PFAS ending up 99.98% in val) -- confirming the original split correctly
kept the overwhelming majority of chain-PFAS in train while preserving a
small, real validation set.

**Fix implemented in `scripts/prepare_combined_nistpfas_enveda180_dataset.py`:**
NIST-PFAS's original `fold` column is now loaded and preserved as-is;
the Murcko Histogram Split (section 16) now runs *only* on Enveda-180's
184,330 molecules (which never had a prior split to preserve), not the
whole combined pool. `assign_folds()` itself is unchanged -- just called
on a filtered `mol_info` dict excluding NIST-PFAS's molecules, then the
two fold maps are merged (NIST-PFAS's preserved values take precedence
on the rare inchikey overlap).

**Result, full re-run:**

| Category | Train | Val |
|---|---|---|
| Chain-PFAS | 18,975 (95.38%) | 920 (4.62%) |
| Drug-like isolated | 59,763 (73.52%) | 21,529 (26.48%) |
| Other fluorinated | 147,526 (78.12%) | 41,327 (21.88%) |
| Non-fluorinated | 665,786 (74.59%) | 226,858 (25.41%) |

Chain-PFAS now matches NIST-PFAS's original ratio almost exactly (95.38%
vs. the source's 96.2%), and every other category retains healthy
representation in both folds. Molecule-level split: 75.97% train /
24.03% val (n=184,433). Re-verified: combined TSV still loads correctly
via `IonModeMassSpecDataset`/`MassSpecDataModule` (892,050 train / 290,634
val spectra), and the Tanimoto validation re-run on the corrected file
still shows a healthy, leak-free distribution (mean similarity 0.445,
max 0.930, 0 near-duplicate (>=0.99) matches out of 44,315 val
molecules) -- the fix didn't introduce leakage. Updated plot saved to
`~/Downloads/DreaMS-PFAS-Paper/tanimoto_split_quality.png`.

**This is now the dataset to use going forward** for the multi-head
model.

## 18. Multi-head fluorine/PFAS-type model implemented (2026-07-18)

Built the multi-head model itself (step 2), per Roman's feedback (section
15) and the brainstormed architecture. Extends `HalogenDetectorDreamsTest`
per explicit user choice (not `MassSpecGymModel` directly, despite that
being the initial recommendation -- matches the literal inheritance
pattern already used for Option A/B).

**New files:**
- `massspecgym/models/pfas/multihead.py` --
  `HalogenDetectorDreamsMultiHead(HalogenDetectorDreamsTest)`. Shared
  trunk: DreaMS embedding + ion-mode one-hot concat -> `Linear(1027,128)`
  -> ReLU (Option B fusion, same mechanism as `ion_mode_ffn.py`). Three
  heads off the shared representation: `head_has_f` (binary, trained on
  every example), `head_oecd_pfas` (binary, loss masked to `has_f=1`
  examples), `head_subtype` (3-way **multi-label sigmoid** -- not softmax
  -- `[cf2, cf3, chain]`, loss masked to `is_oecd_pfas=1` examples).
  `super().__init__()` reuses the DreaMS backbone loading + `random_init`
  ablation support from `HalogenDetectorDreamsTest`; `forward`/`step`/
  `on_batch_end`/metrics are all overridden (per-head
  `BinaryPrecision`/`Recall`/`Accuracy`, 3 sets total, stored in two
  separate `nn.ModuleDict`s -- `self.metrics_train`/`self.metrics_val`,
  **not** nested under `"train"`/`"val"` string keys, since `"train"`
  collides with `nn.Module`'s own `.train()` method and raised
  `KeyError: "attribute 'train' already exists"` on first attempt).
- `massspecgym/models/pfas/__init__.py` -- additive export.
- `scripts/train_PFAS_multihead_model.py` -- new `MultiHeadPFASDataset`
  reading the combined TSV's precomputed bonus columns
  (`has_chain_pfas`/`has_isolated_cf2`/`has_isolated_cf3`) directly rather
  than recomputing via RDKit; derives `ion_mode` from the real
  `ion_mode_true` bonus column (preferred over the adduct-heuristic since
  the ground truth is already in the file); derives `has_f` via the
  standard regex convention and `is_oecd_pfas` as the OR of the three
  bonus flags. Points `pth=` directly at
  `combined_nistpfas_enveda180_with_fold.tsv`. **batch_size=128** (up from
  the single-task scripts' 64) since Head 3's positive rate is low
  (~6.6% of molecules are `is_oecd_pfas`) -- a batch of 64 would
  frequently contain few or zero Head-3-eligible examples.

**Local verification (mocked DreaMS backbone, then real data):**
- Unit-tested `forward()` output shapes (`has_f`:[B], `oecd_pfas`:[B],
  `subtype`:[B,3]) and gradient flow into all 3 heads + shared trunk.
- Unit-tested the masking logic specifically, including the edge cases
  the plan called out: a batch with `is_oecd_pfas` all-zero (Head 3 fully
  masked -- confirmed `loss_subtype == 0.0` exactly, finite total loss,
  `.backward()` doesn't raise) and a batch with `has_f` all-zero (Head 2
  **and** 3 both fully masked -- same clean result).
- Ran the real end-to-end pipeline (dataset -> dataloader -> model ->
  train/val step) against a real 2,000-row slice of the actual combined
  TSV (1,925 train / 75 val spectra, all chain-PFAS in this particular
  slice since it falls within NIST-PFAS's block) -- confirmed finite
  losses and successful backward pass on both train and val batches.
- All touched/added files compile cleanly; `git status` confirms only
  the intended files changed.

**Not done locally:** full-scale training against the real DreaMS
checkpoint and the complete 1.18M-spectra combined dataset -- needs the
user's remote environment, same limitation as every prior training script
this session. Also explicitly out of scope per the plan: full
per-ion-mode-per-head PR-table/calibration-curve reporting (v1 has
epoch-level precision/recall/accuracy/F1 per head only) -- a natural
fast-follow once the first real training run's numbers are in.

## 19. Full-scale per-task reporting added (2026-07-18)

Extended the multi-head model with the same reporting depth already
built for the single-head models -- full threshold-sweep PR tables,
per-ion-mode breakdown, and calibration curves -- across all 5 binary
tasks (`has_f`, `oecd_pfas`, `cf2`, `cf3`, `chain`).

**Key decision (confirmed with user): report PR unconditionally, over the
whole validation set, for every task** -- not filtered to each task's
loss-applicable subset. Every example has a genuinely valid 0/1 label for
all 5 tasks (a non-PFAS molecule really is `chain=0`, not missing data),
so masking only matters for the *loss* during training (focusing subtype
learning within the OECD-PFAS-positive subset); for *reporting*, the more
useful question is "how does this head perform if run on any spectrum,"
matching real inference usage. This simplified the implementation
considerably -- no applicability filtering needed in the reporting path
at all, just `HalogenDetectorDreamsTest._compute_pr_table` (inherited
unchanged) called once per task on that task's full y_true/y_prob arrays.

**Changes, all in `massspecgym/models/pfas/multihead.py`:**
- `_reset_metrics_val` now also resets per-task collection dicts
  (`self.all_predicted_probs[task]`, `self.all_true_labels[task]`,
  plus shared `all_identifiers`/`all_ion_modes`).
- `on_batch_end` (val branch) now also collects predicted probability +
  true label for all 5 tasks per example, unconditionally.
- `on_validation_epoch_end` now loops over all 5 tasks after the existing
  epoch-metric logging: full PR table (`pr_table_{task}_{seed}.csv`),
  per-ion-mode breakdown (`pr_table_{task}_ion_mode_{mode}_{seed}.csv`,
  same skip-empty-subset behavior as the single-task version), and
  calibration data (`calibration_curve_{task}_{seed}.csv`,
  `score_distribution_{task}_{seed}.csv`).
- New `_save_combined_calibration_figure` -- per user's choice, **one
  combined figure** (5 rows x 2 cols: calibration curve + score
  distribution per row, one row per task) saved as
  `calibration_curve_multihead_{seed}.png`, rather than 5 separate PNGs.
- Not reimplemented (flagged as a low-stakes simplification): the
  single-task version's TP/FN identifier-dump text files -- would
  multiply into ~15 extra files per validation epoch for a debug aid not
  essential to this ask.

**Local verification:** re-ran the existing masked-batch loss unit tests
(unaffected -- loss masking is unchanged, only reporting changed) --
all still pass. New end-to-end test against the real
`combined_tiny_sample.tsv` slice (1,925 train / 75 val, 100% chain-PFAS
in this particular slice) confirmed all 21 expected output files are
produced (5 PR tables, 5 ion-mode-split PR tables -- only "negative" mode
present in this slice so 5 not 15, 5 calibration CSVs, 5 score-distribution
CSVs, 1 combined figure). Confirmed the `cf2`/`cf3` tasks correctly
produce a full (if uniformly zero precision/recall) PR table on this
all-negative-class slice, rather than erroring -- expected behavior for
unconditional reporting on a subset with no positive examples for that
task, not a bug.

## 20. Reporting output organized into seed_{seed}/ directories (2026-07-18)

Per user request: all PR-table/calibration-curve/identifier-dump output
now writes into a `seed_{seed}/` subdirectory (created on demand) instead
of loose files directly in cwd, so a run's full output can be purged with
one `rm -rf seed_{seed}/`. **Applied to both** the single-task models
(`HalogenDetectorDreamsTest` in `massspecgym/models/pfas/base.py`, used
by Option A/B and the original baseline) **and** the multi-head model,
per user's explicit choice for consistency across every PFAS model in
this repo.

**Changes:**
- New `HalogenDetectorDreamsTest._get_seed_output_dir()` helper --
  `os.makedirs(f"seed_{self.seed}", exist_ok=True)`, returns the path.
  All file writes in `save_precision_recall_table` (main + per-ion-mode
  PR tables), `save_calibration_curve` (CSVs + PNG), and the TP/FN
  identifier-dump block in `on_validation_epoch_end` now go through this.
  Also fixed a small pre-existing inconsistency while touching this code:
  the calibration PNG had no `_{seed}` suffix at all (`calibration_curve.png`)
  unlike every other output file -- now `calibration_curve_{seed}.png`,
  consistent with the rest.
- `HalogenDetectorDreamsMultiHead` reuses the same inherited
  `_get_seed_output_dir()` (no need to redefine) -- all 5 tasks' PR
  tables, per-ion-mode breakdowns, calibration CSVs, and the combined
  calibration figure now go under the same `seed_{seed}/` directory.
- `HalogenDetectorDreamsIonModeLinear`/`HalogenDetectorDreamsIonModeFFN`
  (Option A/B) get this behavior automatically too, for free -- they
  don't override the reporting methods, so the inherited change applies
  without touching those files at all.

**Local verification:** re-ran both the multi-head reporting smoke test
and a new equivalent test for the base class (mocked backbone, synthetic
predictions) -- confirmed all files land under `seed_{seed}/` with zero
loose files in the working directory in both cases. One process note:
had to use fresh scratch directories for repeat test runs rather than
`rm -rf`-cleaning old ones, since destructive deletes require explicit
user confirmation in this environment -- didn't affect the actual
verification, just the test mechanics.

