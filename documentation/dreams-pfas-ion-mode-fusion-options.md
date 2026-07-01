# Injecting Ionization Mode into DreaMS-PFAS: Four Architecture Options

Prepared for mentor review. Scope: Step 1 of the workshop-paper roadmap
(see `dreams-pfas-workshop-paper.md`) — add an ionization-mode (polarity)
signal to the PFAS classifier *without* modifying DreaMS itself (no
negative-mode DreaMS backbone exists yet). Evaluated on the existing
merged MassSpecGym + NIST20 + NIST-PFAS dataset, as a controlled ablation
against the current binary/spectral-only baseline before any dataset
changes (Enveda-180, etc.).

## 1. Motivation

The current model (`HalogenDetectorDreamsTest` in
`massspecgym/models/pfas/base.py`) is purely spectral:

```
spectrum → DreaMS (frozen/fine-tuned backbone) → embedding [1024] → Linear(1024,1) → logit
```

PFAS training data is heavily skewed toward negative-ion-mode spectra,
while DreaMS's own pretraining corpus (GNPS) is predominantly
positive-mode. The model currently has no way to know which regime a given
spectrum came from, even though fragmentation behavior (e.g. characteristic
CF2/HF neutral losses) is expected to differ by polarity. We want to give
the model access to ionization mode as a conditioning signal.

Ion mode is derivable today from `metadata["adduct"]` (e.g. `[M+H]+` →
positive, `[M-H]-` → negative), already used for this purpose in
`scripts/visualize_pfas_umap_ionization.py`. It is currently dropped in the
PFAS training dataset class due to a string→tensor casting issue that is
straightforward to fix by deriving the categorical feature before casting.

All four options below assume ion mode is available per-sample as a
category: `{negative, positive}` or `{negative, positive, unknown}`.

---

## 2. Option A — Concatenate + Linear head

Append ion mode as one extra scalar to the embedding, keep a single linear
layer.

```
embedding [1024]  ─┐
                    ├─ concat → [1025] → Linear(1025,1) → logit
ion_mode  [1]     ─┘
```

```python
combined = torch.cat([embedding, ion_mode.float().unsqueeze(1)], dim=1)  # [B,1025]
logits = self.lin_out(combined).squeeze(1)   # Linear(1025, 1)
```

- **New parameters:** ~1 (just one new weight for the appended scalar).
- **Expressiveness:** A linear layer can only add a *constant shift* to the
  logit based on mode — it cannot change how the other 1024 spectral
  dimensions are weighted per polarity.
- **Verdict:** Cheapest possible change, useful as a sanity-check floor,
  but likely too weak given that polarity is expected to change *which*
  spectral features matter, not just shift a baseline score.

---

## 3. Option B — Concatenate + small FFN

Same concatenation, but route through one hidden layer before the output.

```
embedding [1024]  ─┐
                    ├─ concat → [1025] → Linear(1025,H) → ReLU → Linear(H,1) → logit
ion_mode  [1]     ─┘
```

```python
h = F.relu(self.fc1(combined))     # Linear(1025, H)
logits = self.fc2(h).squeeze(1)    # Linear(H, 1)
```

- **New parameters:** `(1025+1)·H + (H+1)`, e.g. **~131K** for `H=128`.
- **Expressiveness:** The hidden layer + nonlinearity lets the network
  learn interaction terms between `ion_mode` and individual embedding
  dimensions (a ReLU unit can act as a soft gate conditioned on the
  appended scalar). Meaningfully more expressive than Option A for modest
  added cost.
- **Verdict:** Good, simple first upgrade over the baseline; minimal new
  code, no architectural surgery on the merge point.

---

## 4. Option C — FiLM conditioning (Feature-wise Linear Modulation)

Instead of adding ion mode as an extra input, use it to directly rescale
and shift the embedding itself, per dimension, before the (unchanged)
linear head.

```
h' = gamma[mode] ⊙ h + beta[mode]
```

```
                                   mode_idx (0=neg, 1=pos, [2=unknown])
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                              ▼
              nn.Embedding[num_modes,1024]   nn.Embedding[num_modes,1024]
                     gamma lookup                   beta lookup
                         │  g [1024]                    │  b [1024]
   spectrum              │                              │
        │                │                              │
        ▼                │                              │
   [ DreaMS ]             │                              │
        │ h [1024]        │                              │
        ▼                ▼                              │
      elementwise ⊙  ────┘                               │
        │  g⊙h [1024]                                    │
        ▼                                                │
      elementwise + ◄──────────────────────────────────┘
        │  h' = g⊙h + b  [1024]
        ▼
   Linear(1024,1)  (same shape as today's head)
        │
     logit
```

```python
self.gamma = nn.Embedding(num_modes, 1024)   # init ≈ 1.0 (identity at start)
self.beta  = nn.Embedding(num_modes, 1024)   # init ≈ 0.0 (identity at start)
self.lin_out = nn.Linear(1024, 1)            # unchanged from current model

def forward(self, x, mode_idx):
    h = self.main_model(x)[:, 0, :]
    h_mod = self.gamma(mode_idx) * h + self.beta(mode_idx)
    return self.lin_out(h_mod).squeeze(1)
```

- **New parameters:** `2 · num_modes · 1024` — **~4.1K** for 2 modes,
  **~6.1K** for 3 modes (incl. "unknown"). Notably *fewer* new parameters
  than Option B, despite being more expressive in a different way.
- **Expressiveness:** Every embedding dimension gets its own learned
  per-mode scale and shift — the network can re-read the *entire*
  representation differently per polarity, not just adjust one appended
  input. `gamma`/`beta` are ordinary learned parameters trained via
  backprop (chain rule through elementwise multiply/add and the embedding
  lookup — no special machinery needed).
- **Initialization matters:** initializing `gamma≈1`, `beta≈0` makes FiLM
  the identity function at the start of training, so it doesn't disrupt
  DreaMS's pretrained embedding space before fine-tuning begins.
- **Bonus:** because `gamma`/`beta` are attached to specific embedding
  dimensions, we can later inspect which dimensions shift most between
  polarities — a natural companion to the existing UMAP-by-fluorine-count
  analysis in the ISEF paper (Figs. 8–10).
- **Verdict:** More principled conditioning mechanism than concatenation;
  comparable or lower parameter cost than Option B. Standard technique
  (Perez et al. 2017) for conditioning a shared backbone on a discrete
  context variable.

---

## 5. Option D — Per-mode gated/expert heads

Two independent heads, one per polarity, selected by mode.

```
                    ┌── mode==negative ──► Linear(1024,1)_neg ──┐
embedding [1024] ───┤                                            ├──► logit
                    └── mode==positive ──► Linear(1024,1)_pos ──┘
```

```python
logit_neg = self.lin_out_neg(h)
logit_pos = self.lin_out_pos(h)
logits = torch.where(mode == 1, logit_pos, logit_neg)
```

Equivalent to a single `Linear(2048,1)` over `[h·mode, h·(1−mode)]`.

- **New parameters:** `num_modes · 1025` — **~2.1K** for 2 modes.
- **Expressiveness:** Fully decouples the decision function per mode — no
  shared weights between polarities at all.
- **Risk:** Each head only ever trains on its own polarity's examples. PFAS
  data is currently skewed toward negative mode, so the positive-mode head
  would train on a small, already-scarce subset — this option could worsen
  exactly the imbalance problem we're trying to fix, unless paired with a
  shared trunk + small per-mode residual instead of fully separate heads.
- **Verdict:** Most flexible in principle, but the current data skew makes
  it the highest-risk option for this ablation round. Worth revisiting
  after the dataset is enlarged (Enveda-180, roadmap step 2).

---

## 6. Comparison summary

| Option | Merge point | New params (approx.) | Can reweight spectral dims per mode? | Main risk |
|---|---|---|---|---|
| A. Concat + Linear | after embedding, into head input | ~1 | No — additive shift only | Likely underpowered |
| B. Concat + FFN | after embedding, into hidden layer | ~131K (H=128) | Indirectly, via hidden units | More params to fit on scarce positive-mode data |
| C. FiLM | on the embedding itself, before head | ~4–6K | Yes — per-dimension scale/shift | Under-trained rows if a mode is rare |
| D. Gated heads | fully separate heads | ~2K | Yes — fully separate function | Positive-mode head starved by data skew |

## 7. Recommendation

Prototype **B (concat+FFN)** and **C (FiLM)** side by side as the first
ablation, both evaluated against the existing binary baseline on the
current merged dataset (no dataset changes yet). Both avoid fully
partitioning training data by mode (Option D's main risk), and comparing
them gives a natural ablation table for the workshop paper (baseline vs.
concat+FFN vs. FiLM). Hold Option D for after the dataset is enlarged,
since it's the option most sensitive to the current positive/negative
mode imbalance we're trying to fix in the first place.

## 8. Open questions for mentor

- Does the recommendation above (skip Option D for now) match your
  intuition, or is there a reason to prioritize it sooner (e.g. as a
  diagnostic for how bad the imbalance is)?
- For FiLM, should `num_modes` be 2 (positive/negative) or 3
  (positive/negative/unknown), given some adducts may not cleanly map to a
  polarity?
- Should DreaMS's own backbone weights stay unfrozen (fine-tuned, as in
  the current ISEF setup) while adding any of these heads, or is it worth
  testing a frozen-backbone variant now that the head has more capacity?
