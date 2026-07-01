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
- **Decision:** pending. Full writeup of the four options (concat+linear,
  concat+FFN, FiLM, gated heads) with diagrams, code sketches, parameter
  counts, and a recommendation was prepared for mentor review:
  `documentation/dreams-pfas-ion-mode-fusion-options.md`. Recommendation
  in that doc: prototype concat+FFN and FiLM side by side first; hold
  gated/expert heads until after the dataset is enlarged (they're most
  sensitive to the current mode imbalance).

## 7. Status

Architecture discussion in progress for Step 1 (see §6). No code written
yet. Once an ion-mode fusion approach is picked, build: (a) dataset
subclass exposing a clean `ion_mode` field, (b) new model class
implementing the chosen fusion, (c) rerun on the existing merged dataset to
compare against the ISEF paper's binary baseline, before touching
Enveda-180 (step 2 of Roman's roadmap).
