import torch
from pytorch_lightning import Trainer
import numpy, sys
import wandb
from pathlib import Path
import torch.optim as optim
import pandas as pd

from massspecgym.data import MassSpecDataset, MassSpecDataModule
from massspecgym.data.transforms import SpecTokenizer, MolFingerprinter
from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import MassSpecGymModel
from sklearn.metrics import precision_score, recall_score, accuracy_score
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import grad_norm
from torch import nn

import torch.nn.functional as F
from massspecgym.models.base import Stage
from dreams.api import PreTrainedModel
from dreams.models.dreams.dreams import DreaMS as DreaMSModel
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAccuracy
import numpy as np
from rdkit import Chem
from massspecgym.data.transforms import MolToHalogensVector, MolToPFASVector
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import roc_auc_score, average_precision_score

numpy.set_printoptions(threshold=sys.maxsize)
n_examples_to_sample = 100

class HalogenDetectorDreamsTest(MassSpecGymModel):
    def __init__(
        self,
        alpha: float=0.25,
        gamma: float=0.5,
        batch_size: int=64,
        threshold: float=0.5,
        hard_neg_weight: float=1,
        loss: str="bce",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.register_buffer("pos_weight", torch.tensor([1.0], dtype=torch.float32))  # placeholder

        # if mps_device is not None:
        #     self.alpha = torch.tensor([1-alpha, alpha], device=mps_device)
        # else:
        #     self.alpha = torch.tensor([1-alpha, alpha]).cuda() 

        self.register_buffer("alpha_vec", torch.tensor([1 - alpha, alpha], dtype=torch.float32))
        self.gamma = gamma
        self.batch_size = batch_size
        self.threshold = threshold
        self.hard_neg_weight = hard_neg_weight
        self.loss = loss

        print(f"Training with threshold: {threshold}, \
            alpha: {alpha}, \
            gamma: {gamma}, \
            batch_size: {batch_size},\
            hard_neg_weight: {hard_neg_weight}, \
            loss: {loss}") 
        
        # Metrics
        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_accuracy = BinaryAccuracy()

        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_accuracy = BinaryAccuracy()
                
        # loading the DreaMS model weights from the internet
        self.main_model = PreTrainedModel.from_ckpt(
            # ckpt_path should be replaced with the path to the ssl_model.ckpt model downloaded from https://zenodo.org/records/10997887
            ckpt_path="https://zenodo.org/records/10997887/files/ssl_model.ckpt?download=1", ckpt_cls=DreaMSModel, n_highest_peaks=60
        ).model.train()

        # New classification head for PFAS detection
        self.lin_out = nn.Linear(1024, 1) # for F

    
    def forward(self, x):
        output_main_model = self.main_model(x)[:, 0, :]  # [B,1024]
        logits = self.lin_out(output_main_model).squeeze(1)  # [B]
        return logits # numerical stability during loss computation

    def step(
        self, batch: dict, stage: Stage
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.loss == "bce":
            return self._step_bce_loss(batch, stage)
        else:
            raise ValueError("Only BCELoss supported now")
            #return self._step_focal_loss(batch, stage)

    def _step_bce_loss(self, batch: dict, stage: Stage) -> dict:
        # Unpack inputs
        x = batch["spec"]                 # [B, n_peaks+1, 2]
        y_vec = batch["mol"][:, 0]        # [B] PFAS label (should be 0/1)

        # ✅ Force clean binary targets (prevents gather/index issues)
        targets = (y_vec > 0.5).float()   # [B] in {0.0, 1.0}

        # ✅ Forward should return logits (so update forward() accordingly)
        logits = self.forward(x)          # [B] logits (NOT probabilities)

        # --- hard negative weighting (keep your idea) ---
        F_present = batch["F"].float()    # [B]
        hard_neg = (F_present == 1) & (targets == 0)

        weights = torch.ones_like(targets)
        weights = torch.where(
            hard_neg,
            torch.tensor(float(self.hard_neg_weight), device=weights.device),
            weights
        )

        # ✅ Stable loss for rare positives
        per_sample = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.to(logits.device),  # buffer set in __init__
            reduction="none"
        )  # [B]

        loss = (per_sample * weights).mean()

        # Optional debug prints (remove once stable)
        # probs = torch.sigmoid(logits)
        # print("targets pos rate:", targets.mean().item(), "probs mean:", probs.mean().item())

        return {"loss": loss}
    
    def _step_focal_loss(self, batch: dict, stage: Stage) -> dict:
        print("here...Focal")
        """Implement your custom logic of using predictions for training and inference."""
        # Unpack inputs
        x = batch["spec"]  # shape: [batch_size, num_peaks + 1, 2]

        halogen_vector_true = batch["mol"] # shape: [batch_size, 4]

        logits = self.forward(x)  # [B] logits

        # True PFAS labels from vector
        true_values = halogen_vector_true[:, 0]

        # SAFETY: force labels to be clean 0/1 longs (prevents gather crash)
        # If your PFAS vector is already 0/1, this is harmless.
        targets = (true_values > 0.5).long()          # [B] in {0,1}
        targets_f = targets.float()                   # [B] float for BCE

        # Hard negatives weighting
        F_present = batch["F"].float()
        hard_negatives = (F_present == 1) & (targets == 0)
        weights = torch.ones_like(targets_f)
        weights = torch.where(hard_negatives, torch.tensor(float(self.hard_neg_weight), device=weights.device), weights)

        # ---- Stable focal loss on logits ----
        bce = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="none")  # [B]
        pt = torch.exp(-bce)  # [B]

        # alpha per-sample: alpha for class1, (1-alpha) for class0
        alpha_t = self.alpha_vec[1] * targets_f + self.alpha_vec[0] * (1.0 - targets_f)

        loss = alpha_t * (1 - pt).pow(self.gamma) * bce
        loss = (loss * weights).mean()

        # Debug stats (optional)
        # probs = torch.sigmoid(logits)
        # print("p(min/mean/max)=", probs.min().item(), probs.mean().item(), probs.max().item())

        return {"loss": loss}


    def on_batch_end(self, outputs: dict, batch: dict, batch_idx: int, stage: Stage) -> None:
        x = batch["spec"]
        halogen_vector_true = batch["mol"]

        # forward now returns logits: [B]
        logits = self.forward(x)

        # ✅ convert logits -> probabilities: [B]
        pred_probs = torch.sigmoid(logits)

        # thresholding (keep as is)
        pred_labels = (pred_probs >= self.threshold).long()

        # true labels as int/long
        true_labels = halogen_vector_true[:, 0].long()

        if stage.to_pref() == 'train_':
            self.train_precision.update(pred_labels, true_labels)
            self.train_recall.update(pred_labels, true_labels)
            self.train_accuracy.update(pred_labels, true_labels)

        elif stage.to_pref() == 'val_':
            self.val_precision.update(pred_labels, true_labels)
            self.val_recall.update(pred_labels, true_labels)
            self.val_accuracy.update(pred_labels, true_labels)

            # ---- store probabilities for AUROC/AP ----
            identifiers = batch.get("identifier", ["NA"] * len(true_labels))
            spectra = batch.get("spec", None)

            pred_probs_flat = pred_probs.detach().cpu().numpy().tolist()
            self.all_predicted_probs.extend(pred_probs_flat)
            self.all_true_labels.extend(true_labels.detach().cpu().numpy().tolist())
            self.all_identifiers.extend(identifiers)

            if spectra is not None:
                self.all_spectra.extend([s.detach().cpu().numpy() for s in spectra])

        self.log_dict(
            {f"{stage.to_pref()}/loss": outputs["loss"]},
            prog_bar=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )

    #added with ai^

    def _reset_metrics_train(self):
        # Reset states for next epoch
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_accuracy.reset()

    def _reset_metrics_val(self):
        # Reset states for next epoch
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_accuracy.reset()
        self.all_predicted_probs = []  # reset the list of predicted probabilities for validation
        self.all_true_labels = []
        self.all_identifiers = []
        self.all_spectra = []


    def on_train_epoch_start(self) -> None:
        self._reset_metrics_train()

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics_val()

    def on_train_epoch_end(self) -> None:
        precision = self.train_precision.compute()
        recall = self.train_recall.compute()
        accuracy = self.train_accuracy.compute()
        f1_score = (2*precision*recall)/(precision + recall) if (precision + recall) != 0 else 0
        self.log_dict({
                f"train_/precision": precision,
                f"train_/recall": recall,
                f"train_/accuracy": accuracy,
                f"train_/f1_score": f1_score
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False
        )
        
    def on_validation_epoch_end(self) -> None:
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        accuracy = self.val_accuracy.compute()
        f1_score = (2*precision*recall)/(precision + recall) if (precision + recall) != 0 else 0
        #val_auroc = roc_auc_score(self.all_true_labels, self.all_predicted_probs)
        #val_ap_score = average_precision_score(self.all_true_labels, self.all_predicted_probs)

        self.log_dict({
                f"val_/precision": precision,
                f"val_/recall": recall,
                f"val_/accuracy": accuracy,
                f"val_/f1_score": f1_score
            #    f"val_/auroc": val_auroc,
            #    f"val_/ap_score": val_ap_score
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False
        )

         # Stop if no predictions
        if len(self.all_predicted_probs) == 0:
            return

        # Create a dataframe of predictions
        df_preds = pd.DataFrame({
            "identifier": self.all_identifiers,
            "true_label": self.all_true_labels,
            "pred_prob": self.all_predicted_probs,
            "spec": self.all_spectra
        })
        df_preds["pred_label"] = (df_preds["pred_prob"] >= self.threshold).astype(int)

        # Identify True Positives and False Negatives
        df_tp = df_preds[(df_preds["true_label"] == 1) & (df_preds["pred_label"] == 1)]
        # Identify False Negatives which have probability <= 0.2
        df_fn = df_preds[(df_preds["true_label"] == 1) & (df_preds["pred_label"] == 0) & (df_preds["pred_prob"] < 0.2)]

        print(f"\n🧪 Validation Summary:")
        print(f"  True Positives (TP): {len(df_tp)}")
        print(f"  False Negatives (FN): {len(df_fn)}")

        # ---- PRINT IDENTIFIERS ----
        # ---- Randomly sample examples ----
        tp_samples = df_tp.sample(min(n_examples_to_sample, len(df_tp)), random_state=42)
        fn_samples = df_fn.sample(min(n_examples_to_sample, len(df_fn)), random_state=42)

        # ---- Print Identifiers ----
        tp_filename = "true_positive_identifiers.txt"
        fn_filename = "false_negative_identifiers.txt"

        # Write True Positive identifiers
        with open(tp_filename, "w") as f:
            f.write("---True Positive Identifiers:\n")
            for i, ident in enumerate(tp_samples["identifier"].tolist(), 1):
                f.write(f"  {i}. {ident}\n")

        # Write False Negative identifiers
        with open(fn_filename, "w") as f:
            f.write("---False Negative Identifiers:\n")
            for i, ident in enumerate(fn_samples["identifier"].tolist(), 1):
                f.write(f"  {i}. {ident}\n")

        pred_prob_filename = "pfas_pred_probs.csv"
        # Find the predicted probabilities for all PFAS
        df_p = df_preds[(df_preds["true_label"] == 1)]
        df_p["pred_prob"].to_csv(pred_prob_filename, index=False)

        print(f"✅ True Positive identifiers written to {tp_filename}")
        print(f"✅ False Negative identifiers wth probability < 0.2 written to {fn_filename}")
        print(f"✅ Predicted Probabilities Positives written to {pred_prob_filename}")

        self.save_precision_recall_table()


    def save_precision_recall_table(self) -> None:
    # --- after you already have self.all_true_labels / self.all_predicted_probs ---
        y_true = np.array(self.all_true_labels)
        y_prob = np.array(self.all_predicted_probs)

        thresholds = np.arange(0.0, 1.01, 0.1)   # 0.00 → 1.00 in steps of 0.1
        rows = []

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            precision = precision_score(y_true, y_pred, zero_division="warn")
            recall    = recall_score(y_true, y_pred, zero_division="warn")
            accuracy  = accuracy_score(y_true, y_pred)
            # Compute F1 (safe even if precision+recall = 0)
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            rows.append((t, precision, recall, accuracy, f1))

        # Convert to dataframe for cleaner printing
        df_thresh = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "accuracy", "f1"])

        print("\n=== Precision / Recall / Accuracy / F1 by Threshold ===")
        print(df_thresh.to_string(index=False))
        df_thresh.to_csv("pr_table.csv", index=False)