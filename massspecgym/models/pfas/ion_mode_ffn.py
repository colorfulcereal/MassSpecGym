import torch
import torch.nn.functional as F
from torch import nn

from massspecgym.models.base import Stage
from massspecgym.models.pfas.base import HalogenDetectorDreamsTest


class HalogenDetectorDreamsIonModeFFN(HalogenDetectorDreamsTest):
    """
    Option B ion-mode fusion: concatenate a one-hot ionization-mode feature
    onto the DreaMS embedding and feed the result through a small FFN
    (Linear -> ReLU -> Linear) instead of a single linear layer. This lets
    ion mode interact with individual embedding dimensions via the hidden
    layer, rather than only shifting the final logit by a constant per
    mode (as in HalogenDetectorDreamsIonModeLinear / Option A). Everything
    else (backbone, losses, metrics, epoch-end logging, per-ion-mode PR
    table reporting) is inherited unchanged from HalogenDetectorDreamsTest.
    """

    def __init__(self, ion_mode_vocab_size: int = 3, hidden_dim: int = 128,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ion_mode_vocab_size = ion_mode_vocab_size
        self.hidden_dim = hidden_dim

        # Replace the parent's Linear(1024, 1) head with a small FFN sized
        # for the concatenated [embedding, ion_mode_onehot] vector.
        del self.lin_out
        self.fc1 = nn.Linear(1024 + ion_mode_vocab_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, ion_mode):
        embedding = self.main_model(x)[:, 0, :]  # [B, 1024]
        ion_mode_onehot = F.one_hot(
            ion_mode.long(), num_classes=self.ion_mode_vocab_size
        ).float()  # [B, ion_mode_vocab_size]
        combined = torch.cat([embedding, ion_mode_onehot], dim=1)  # [B, 1024 + k]
        h = F.relu(self.fc1(combined))  # [B, hidden_dim]
        logits = self.fc2(h).squeeze(1)  # [B]
        return logits

    def _step_bce_loss(self, batch: dict, stage: Stage) -> dict:
        x = batch["spec"]                 # [B, n_peaks+1, 2]
        ion_mode = batch["ion_mode"]       # [B]
        y_vec = batch["mol"][:, 0]        # [B] PFAS label (should be 0/1)

        targets = (y_vec > 0.5).float()   # [B] in {0.0, 1.0}

        logits = self.forward(x, ion_mode)  # [B] logits

        F_present = batch["F"].float()    # [B]
        hard_neg = (F_present == 1) & (targets == 0)

        weights = torch.ones_like(targets)
        weights = torch.where(
            hard_neg,
            torch.tensor(float(self.hard_neg_weight), device=weights.device),
            weights
        )

        per_sample = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.to(logits.device),
            reduction="none"
        )  # [B]

        loss = (per_sample * weights).mean()

        return {"loss": loss}

    def _step_focal_loss(self, batch: dict, stage: Stage) -> dict:
        x = batch["spec"]  # shape: [batch_size, num_peaks + 1, 2]
        ion_mode = batch["ion_mode"]  # [B]
        halogen_vector_true = batch["mol"]  # shape: [batch_size, 4]

        logits = self.forward(x, ion_mode)  # [B] logits
        probs = torch.sigmoid(logits)  # [B] probabilities in (0, 1)

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"[WARNING] NaN/Inf in probs at stage={stage}, skipping batch")
            return {"loss": torch.tensor(0.0, device=probs.device, requires_grad=True)}

        true_values = halogen_vector_true[:, 0]

        targets = (true_values > 0.5).long()   # [B] in {0, 1}
        targets_f = targets.float()            # [B] float for BCE

        F_present = batch["F"].float()
        hard_negatives = (F_present == 1) & (targets == 0)
        weights = torch.ones_like(targets_f)
        weights = torch.where(
            hard_negatives,
            torch.tensor(float(self.hard_neg_weight), device=weights.device),
            weights,
        )

        probs = probs.clamp(min=1e-6, max=1.0 - 1e-6)

        bce = -(targets_f * torch.log(probs) + (1.0 - targets_f) * torch.log(1.0 - probs))  # [B]

        pt = probs * targets_f + (1.0 - probs) * (1.0 - targets_f)  # [B]
        focal_weight = (1.0 - pt).clamp(min=0.0).pow(self.gamma)    # [B]

        alpha_t = self.alpha_vec[1] * targets_f + self.alpha_vec[0] * (1.0 - targets_f)

        loss = alpha_t * focal_weight * bce
        loss = (loss * weights).mean()

        return {"loss": loss}

    def on_batch_end(self, outputs: dict, batch: dict, batch_idx: int, stage: Stage) -> None:
        x = batch["spec"]
        ion_mode = batch["ion_mode"]
        halogen_vector_true = batch["mol"]

        logits = self.forward(x, ion_mode)

        pred_probs = torch.sigmoid(logits)

        pred_labels = (pred_probs >= self.threshold).long()

        true_labels = halogen_vector_true[:, 0].long()

        if stage.to_pref() == 'train_':
            self.train_precision.update(pred_labels, true_labels)
            self.train_recall.update(pred_labels, true_labels)
            self.train_accuracy.update(pred_labels, true_labels)

        elif stage.to_pref() == 'val_':
            self.val_precision.update(pred_labels, true_labels)
            self.val_recall.update(pred_labels, true_labels)
            self.val_accuracy.update(pred_labels, true_labels)

            identifiers = batch.get("identifier", ["NA"] * len(true_labels))
            spectra = batch.get("spec", None)

            pred_probs_flat = pred_probs.detach().cpu().numpy().tolist()
            self.all_predicted_probs.extend(pred_probs_flat)
            self.all_true_labels.extend(true_labels.detach().cpu().numpy().tolist())
            self.all_identifiers.extend(identifiers)
            self.all_ion_modes.extend(ion_mode.detach().cpu().numpy().tolist())

            if spectra is not None:
                self.all_spectra.extend([s.detach().cpu().numpy() for s in spectra])

        self.log_dict(
            {f"{stage.to_pref()}/loss": outputs["loss"]},
            prog_bar=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
