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

**Option A (concat + single linear layer) is implemented and locally
verified** (2026-07-11); full-scale benchmark training still needs to run
on the user's remote environment (see §7.1). Options B (concat+FFN) and C
(FiLM) are the next planned passes, reusing the same dataset/helper code.

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

**Not done locally (needs the user's remote environment):** an actual
`trainer.fit()`/`trainer.validate()` run against the real DreaMS checkpoint
and the full merged MassSpecGym+NIST20+NIST-PFAS TSV, i.e. the real
precision/recall/F1 benchmark vs. the ISEF baseline (93.8%/89%/0.91). Next
step for the user: copy/run `scripts/train_PFAS_ion_mode_linear_model.py`
in the Lightning.ai Studio environment where the merged TSV lives, then
report back the benchmark numbers so we can decide whether to proceed to
Option B (FFN) regardless (per the plan, B is the next step either way,
since Option A was expected to be a weak floor).

