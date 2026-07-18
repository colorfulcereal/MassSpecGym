import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAccuracy
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from massspecgym.models.base import Stage
from massspecgym.models.pfas.base import HalogenDetectorDreamsTest

TASKS = ["has_f", "oecd_pfas", "cf2", "cf3", "chain"]
METRIC_CLASSES = {"precision": BinaryPrecision, "recall": BinaryRecall, "accuracy": BinaryAccuracy}
ION_MODE_NAMES = {0: "negative", 1: "positive", 2: "unknown"}


class HalogenDetectorDreamsMultiHead(HalogenDetectorDreamsTest):
    """
    General fluorine/PFAS-type predictor with 3 heads sharing one DreaMS +
    ion-mode-concat trunk (Option B fusion, same as HalogenDetectorDreamsIonModeFFN):

      Head 1 (has_f):     binary, trained on every example (F or not).
      Head 2 (oecd_pfas):  binary, is_OECD_PFAS vs. other-fluorinated;
                           loss masked to examples where has_f=1.
      Head 3 (subtype):   multi-label sigmoid [cf2, cf3, chain] -- NOT
                           softmax, since isolated CF2/CF3 can co-occur;
                           loss masked to examples where is_oecd_pfas=1.

    All labels are computed deterministically for every example during
    dataset prep (not partial external annotations), so "masking" here
    means "only include this example's loss/metrics for heads where the
    label is semantically applicable," not "label is missing."

    Extends HalogenDetectorDreamsTest (not MassSpecGymModel directly) per
    explicit choice -- reuses its DreaMS backbone loading / random_init
    ablation support from __init__, but overrides forward/step/
    on_batch_end/metrics entirely, since the parent's single-binary-task
    structure doesn't fit a 3-head masked-loss model.
    """

    def __init__(self, ion_mode_vocab_size: int = 3, hidden_dim: int = 128,
                 loss_weights: tuple = (1.0, 1.0, 1.0), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ion_mode_vocab_size = ion_mode_vocab_size
        self.hidden_dim = hidden_dim
        self.loss_weights = loss_weights

        # Replace the parent's single Linear(1024,1) head + single metric
        # trio with the shared trunk + 3 heads + per-head metrics.
        del self.lin_out
        del self.train_precision, self.train_recall, self.train_accuracy
        del self.val_precision, self.val_recall, self.val_accuracy

        self.fc_shared = nn.Linear(1024 + ion_mode_vocab_size, hidden_dim)
        self.head_has_f = nn.Linear(hidden_dim, 1)
        self.head_oecd_pfas = nn.Linear(hidden_dim, 1)
        self.head_subtype = nn.Linear(hidden_dim, 3)  # order: cf2, cf3, chain
        for layer in [self.fc_shared, self.head_has_f, self.head_oecd_pfas, self.head_subtype]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        # NOTE: keyed by two separate ModuleDicts (not one nested under
        # "train"/"val" stage keys) because "train" collides with
        # nn.Module's own .train() method name -- nn.ModuleDict.add_module
        # raises if a key shadows an existing attribute.
        def _make_metric_dict():
            return nn.ModuleDict({
                f"{task}_{metric_name}": metric_cls()
                for task in TASKS
                for metric_name, metric_cls in METRIC_CLASSES.items()
            })
        self.metrics_train = _make_metric_dict()
        self.metrics_val = _make_metric_dict()

    def forward(self, x, ion_mode):
        embedding = self.main_model(x)[:, 0, :]  # [B, 1024]
        ion_mode_onehot = F.one_hot(
            ion_mode.long(), num_classes=self.ion_mode_vocab_size
        ).float()  # [B, k]
        combined = torch.cat([embedding, ion_mode_onehot], dim=1)  # [B, 1024+k]
        shared = F.relu(self.fc_shared(combined))  # [B, hidden_dim]
        return {
            "has_f": self.head_has_f(shared).squeeze(1),          # [B]
            "oecd_pfas": self.head_oecd_pfas(shared).squeeze(1),  # [B]
            "subtype": self.head_subtype(shared),                 # [B, 3]
        }

    def step(self, batch: dict, stage: Stage) -> dict:
        x = batch["spec"]
        ion_mode = batch["ion_mode"]
        has_f = batch["has_f"].float()
        is_oecd_pfas = batch["is_oecd_pfas"].float()
        subtype_targets = torch.stack(
            [batch["has_isolated_cf2"].float(), batch["has_isolated_cf3"].float(),
             batch["has_chain_pfas"].float()], dim=1
        )  # [B, 3]

        logits = self.forward(x, ion_mode)

        # Head 1: always active.
        loss1 = F.binary_cross_entropy_with_logits(logits["has_f"], has_f, reduction="mean")

        # Head 2: masked to has_f=1. Guard against a fully-masked batch
        # (mask.sum()==0) same as the NaN/Inf guard pattern already used
        # in HalogenDetectorDreamsTest._step_focal_loss.
        mask2 = has_f
        if mask2.sum() > 0:
            per_sample2 = F.binary_cross_entropy_with_logits(logits["oecd_pfas"], is_oecd_pfas, reduction="none")
            loss2 = (per_sample2 * mask2).sum() / mask2.sum()
        else:
            loss2 = torch.tensor(0.0, device=logits["oecd_pfas"].device)

        # Head 3: masked to is_oecd_pfas=1.
        mask3 = is_oecd_pfas
        if mask3.sum() > 0:
            per_sample3 = F.binary_cross_entropy_with_logits(
                logits["subtype"], subtype_targets, reduction="none"
            ).mean(dim=1)  # [B], averaged over the 3 sub-labels
            loss3 = (per_sample3 * mask3).sum() / mask3.sum()
        else:
            loss3 = torch.tensor(0.0, device=logits["subtype"].device)

        w1, w2, w3 = self.loss_weights
        loss = w1 * loss1 + w2 * loss2 + w3 * loss3

        return {
            "loss": loss,
            "loss_has_f": loss1.detach(),
            "loss_oecd_pfas": loss2.detach() if loss2.requires_grad else loss2,
            "loss_subtype": loss3.detach() if loss3.requires_grad else loss3,
        }

    def on_batch_end(self, outputs: dict, batch: dict, batch_idx: int, stage: Stage) -> None:
        x = batch["spec"]
        ion_mode = batch["ion_mode"]
        has_f = batch["has_f"].long()
        is_oecd_pfas = batch["is_oecd_pfas"].long()
        has_cf2 = batch["has_isolated_cf2"].long()
        has_cf3 = batch["has_isolated_cf3"].long()
        has_chain = batch["has_chain_pfas"].long()

        logits = self.forward(x, ion_mode)
        pred_has_f = (torch.sigmoid(logits["has_f"]) >= self.threshold).long()
        pred_oecd = (torch.sigmoid(logits["oecd_pfas"]) >= self.threshold).long()
        pred_subtype = (torch.sigmoid(logits["subtype"]) >= self.threshold).long()  # [B, 3]

        m = self.metrics_train if stage.to_pref() == "train_" else self.metrics_val

        m["has_f_precision"].update(pred_has_f, has_f)
        m["has_f_recall"].update(pred_has_f, has_f)
        m["has_f_accuracy"].update(pred_has_f, has_f)

        mask2 = has_f.bool()
        if mask2.any():
            m["oecd_pfas_precision"].update(pred_oecd[mask2], is_oecd_pfas[mask2])
            m["oecd_pfas_recall"].update(pred_oecd[mask2], is_oecd_pfas[mask2])
            m["oecd_pfas_accuracy"].update(pred_oecd[mask2], is_oecd_pfas[mask2])

        mask3 = is_oecd_pfas.bool()
        if mask3.any():
            m["cf2_precision"].update(pred_subtype[mask3, 0], has_cf2[mask3])
            m["cf2_recall"].update(pred_subtype[mask3, 0], has_cf2[mask3])
            m["cf2_accuracy"].update(pred_subtype[mask3, 0], has_cf2[mask3])
            m["cf3_precision"].update(pred_subtype[mask3, 1], has_cf3[mask3])
            m["cf3_recall"].update(pred_subtype[mask3, 1], has_cf3[mask3])
            m["cf3_accuracy"].update(pred_subtype[mask3, 1], has_cf3[mask3])
            m["chain_precision"].update(pred_subtype[mask3, 2], has_chain[mask3])
            m["chain_recall"].update(pred_subtype[mask3, 2], has_chain[mask3])
            m["chain_accuracy"].update(pred_subtype[mask3, 2], has_chain[mask3])

        if stage.to_pref() == "val_":
            # Full-scale per-task reporting: collect predicted probability +
            # true label for every task UNCONDITIONALLY (whole val set, no
            # applicability masking -- see multihead.py module docstring /
            # plan for why reporting differs from the loss's masking).
            probs_has_f = torch.sigmoid(logits["has_f"])
            probs_oecd = torch.sigmoid(logits["oecd_pfas"])
            probs_subtype = torch.sigmoid(logits["subtype"])  # [B, 3]
            task_probs = {
                "has_f": probs_has_f, "oecd_pfas": probs_oecd,
                "cf2": probs_subtype[:, 0], "cf3": probs_subtype[:, 1], "chain": probs_subtype[:, 2],
            }
            task_labels = {
                "has_f": has_f, "oecd_pfas": is_oecd_pfas,
                "cf2": has_cf2, "cf3": has_cf3, "chain": has_chain,
            }
            for task in TASKS:
                self.all_predicted_probs[task].extend(task_probs[task].detach().cpu().numpy().tolist())
                self.all_true_labels[task].extend(task_labels[task].detach().cpu().numpy().tolist())

            identifiers = batch.get("identifier", ["NA"] * len(has_f))
            self.all_identifiers.extend(identifiers)
            ion_modes = batch.get("ion_mode", None)
            if ion_modes is not None:
                self.all_ion_modes.extend(ion_modes.detach().cpu().numpy().tolist())
            else:
                self.all_ion_modes.extend([None] * len(has_f))

        self.log_dict(
            {
                f"{stage.to_pref()}/loss": outputs["loss"],
                f"{stage.to_pref()}/loss_has_f": outputs["loss_has_f"],
                f"{stage.to_pref()}/loss_oecd_pfas": outputs["loss_oecd_pfas"],
                f"{stage.to_pref()}/loss_subtype": outputs["loss_subtype"],
            },
            prog_bar=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )

    def _reset_metrics_train(self):
        for metric in self.metrics_train.values():
            metric.reset()

    def _reset_metrics_val(self):
        for metric in self.metrics_val.values():
            metric.reset()
        self.all_predicted_probs = {task: [] for task in TASKS}
        self.all_true_labels = {task: [] for task in TASKS}
        self.all_identifiers = []
        self.all_ion_modes = []

    def _compute_and_log_epoch_metrics(self, stage_key: str, log_prefix: str) -> None:
        m = self.metrics_train if stage_key == "train" else self.metrics_val
        log_dict = {}
        for task in TASKS:
            precision = m[f"{task}_precision"].compute()
            recall = m[f"{task}_recall"].compute()
            accuracy = m[f"{task}_accuracy"].compute()
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else torch.tensor(0.0)
            log_dict[f"{log_prefix}/{task}_precision"] = precision
            log_dict[f"{log_prefix}/{task}_recall"] = recall
            log_dict[f"{log_prefix}/{task}_accuracy"] = accuracy
            log_dict[f"{log_prefix}/{task}_f1"] = f1
        self.log_dict(log_dict, prog_bar=True, on_epoch=True, on_step=False)

    def on_train_epoch_end(self) -> None:
        self._compute_and_log_epoch_metrics("train", "train_")

    def on_validation_epoch_end(self) -> None:
        self._compute_and_log_epoch_metrics("val", "val_")

        if len(self.all_predicted_probs["has_f"]) == 0:
            return

        ion_modes = np.array(self.all_ion_modes, dtype=object)
        has_ion_mode = (ion_modes != None).any()  # noqa: E711 (elementwise None check)
        seed_dir = self._get_seed_output_dir()  # inherited from HalogenDetectorDreamsTest

        for task in TASKS:
            y_true = np.array(self.all_true_labels[task])
            y_prob = np.array(self.all_predicted_probs[task])

            df_thresh = self._compute_pr_table(y_true, y_prob)
            print(f"\n=== [{task}] Precision / Recall / Accuracy / F1 / TPR / FPR by Threshold "
                  f"(full val set, n={len(y_true)}) ===")
            print(df_thresh.to_string(index=False))
            df_thresh.to_csv(os.path.join(seed_dir, f"pr_table_{task}_{self.seed}.csv"), index=False)

            if has_ion_mode:
                for mode_idx, mode_name in ION_MODE_NAMES.items():
                    mask = (ion_modes == mode_idx)
                    n = int(mask.sum())
                    if n == 0:
                        continue
                    df_mode = self._compute_pr_table(y_true[mask], y_prob[mask])
                    out_pth = os.path.join(seed_dir, f"pr_table_{task}_ion_mode_{mode_name}_{self.seed}.csv")
                    df_mode.to_csv(out_pth, index=False)
                    print(f"Wrote per-mode PR table for [{task}] {mode_name} mode (n={n}) to {out_pth}")

        self._save_combined_calibration_figure()

    def _save_combined_calibration_figure(self) -> None:
        """
        One combined figure, 5 rows (one per task) x 2 cols (calibration
        curve + score distribution), evaluated over the full val set
        unconditionally for every task -- generalizes
        HalogenDetectorDreamsTest.save_calibration_curve (single-task) to
        all 5 binary tasks. Per-task CSVs are still saved individually.
        """
        seed_dir = self._get_seed_output_dir()  # inherited from HalogenDetectorDreamsTest
        fig, axes = plt.subplots(len(TASKS), 2, figsize=(12, 5 * len(TASKS)))
        for row, task in enumerate(TASKS):
            y_true = np.array(self.all_true_labels[task])
            y_prob = np.array(self.all_predicted_probs[task])

            try:
                fraction_of_positives, mean_predicted_prob = calibration_curve(
                    y_true, y_prob, n_bins=10, strategy='uniform'
                )
            except ValueError as e:
                print(f"[WARNING] Could not compute calibration curve for task '{task}' "
                      f"(likely only one class present in this val set): {e}")
                fraction_of_positives, mean_predicted_prob = np.array([]), np.array([])

            pd.DataFrame({
                'mean_predicted_prob': mean_predicted_prob,
                'fraction_of_positives': fraction_of_positives,
            }).to_csv(os.path.join(seed_dir, f"calibration_curve_{task}_{self.seed}.csv"), index=False)

            pd.DataFrame({
                'predicted_prob': y_prob,
                'true_label': y_true,
            }).to_csv(os.path.join(seed_dir, f"score_distribution_{task}_{self.seed}.csv"), index=False)

            ax_cal = axes[row, 0]
            ax_cal.plot(mean_predicted_prob, fraction_of_positives,
                        marker='o', color='steelblue', lw=2, label=task)
            ax_cal.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
            ax_cal.set_xlabel('Mean Predicted Probability')
            ax_cal.set_ylabel('Fraction of Positives')
            ax_cal.set_title(f'Calibration Curve: {task}')
            ax_cal.legend(loc='upper left')

            ax_dist = axes[row, 1]
            ax_dist.hist(y_prob[y_true == 0], bins=20, alpha=0.6, color='steelblue', label='negative')
            ax_dist.hist(y_prob[y_true == 1], bins=20, alpha=0.6, color='tomato', label='positive')
            ax_dist.set_xlabel('Predicted Probability')
            ax_dist.set_ylabel('Count')
            ax_dist.set_title(f'Score Distribution: {task}')
            ax_dist.legend()

        plt.tight_layout()
        out_pth = os.path.join(seed_dir, f"calibration_curve_multihead_{self.seed}.png")
        plt.savefig(out_pth, dpi=200)
        plt.close(fig)
        print(f"\nSaved combined calibration figure to {out_pth}")
