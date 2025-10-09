import torch
import torch.nn as nn
import pytorch_lightning as pl
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
from sklearn.metrics import precision_score, recall_score
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

numpy.set_printoptions(threshold=sys.maxsize)

pl.seed_everything(0)

DEBUG = False
n_examples_to_sample = 30

if DEBUG:
    mgf_pth = Path("/teamspace/studios/this_studio/MassSpecGym/data/debug/example_5_spectra.mgf")
    split_pth = Path("/teamspace/studios/this_studio/MassSpecGym/data/debug/example_5_spectra_split.tsv")
else:
    mgf_pth = None
    split_pth = None

# Check if MPS is available, otherwise use CUDA
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    mps_device = None

# final model containing the network definition

class HalogenDetectorDreamsTest(MassSpecGymModel):
    def __init__(
        self,
        alpha: float=0.25,
        gamma: float=0.5,
        batch_size: int=64,
        threshold: float=0.5,
        hard_neg_weight: float=2.5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if mps_device is not None:
            self.alpha = torch.tensor([1-alpha, alpha], device=mps_device)
        else:
            self.alpha = torch.tensor([1-alpha, alpha]).cuda()
        self.gamma = gamma
        self.batch_size = batch_size
        self.threshold = threshold
        self.hard_neg_weight = hard_neg_weight
        print(f"Training with threshold: {self.threshold}, alpha: {self.alpha}, gamma: {self.gamma}, batch_size: {self.batch_size}, hard_neg_weight: {self.hard_neg_weight}")
        
        # Metrics
        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_accuracy = BinaryAccuracy()

        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_accuracy = BinaryAccuracy()
        
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_accuracy = BinaryAccuracy()
        
        # loading the DreaMS model weights from the internet
        self.main_model = PreTrainedModel.from_ckpt(
            # ckpt_path should be replaced with the path to the ssl_model.ckpt model downloaded from https://zenodo.org/records/10997887
            ckpt_path="https://zenodo.org/records/10997887/files/ssl_model.ckpt?download=1", ckpt_cls=DreaMSModel, n_highest_peaks=60
        ).model.train()
        self.lin_out = nn.Linear(1024, 1) # for F

    def forward(self, x):
        output_main_model = self.main_model(x)[:, 0, :] # to get the precursor peak token embedding 
        fl_probability = F.sigmoid(self.lin_out(output_main_model))
        return fl_probability

    def step(
        self, batch: dict, stage: Stage
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Implement your custom logic of using predictions for training and inference."""
        # Unpack inputs
        x = batch["spec"]  # shape: [batch_size, num_peaks + 1, 2]

        halogen_vector_true = batch["mol"] # shape: [batch_size, 4]

        # the forward pass - predicting using the model
        predicted_probs = self.forward(x) # shape [batch_size x 1]

        # Extract true PFAS labels (1 = PFAS, 0 = not PFAS)
        true_values = halogen_vector_true[:, 0].float()  # shape [batch_size]

        
        # # Build per-sample weights
        # weights = torch.ones_like(true_values, dtype=torch.float32, device=true_values.device)

        # # Whether fluorine is present in the molecule
        # F_present = batch["F"].float()  # shape [batch_size], e.g. 1.0 if 'F' in formula, else 0.0

        # # --- Step 1: Define hard negatives ---
        # # Hard negatives are molecules containing F but not PFAS
        # hard_negatives = (F_present == 1) & (true_values == 0)

        # # --- Step 2: Boost weight for hard negatives 
        # weights = torch.where(hard_negatives, weights * self.hard_neg_weight, weights)

        # --- Step 0: Base weights ---
        weights = torch.ones_like(true_values, dtype=torch.float32, device=true_values.device)

        # --- Step 1: Fluorine presence mask ---
        F_present = batch["F"].float()  # 1.0 if 'F' in formula, else 0.0

        # --- Step 2: Hard negatives mask (F present but not PFAS) ---
        hard_negatives = (F_present == 1) & (true_values == 0)

        # --- Step 3: Abundance-based weights (from dataset preprocessing) ---
        # Each sample has a value in batch["abundance_weight"], already normalized
        abundance_weights = batch["abundance_weight"].to(true_values.device).float()

        # --- Step 4: Combine all weighting schemes ---
        #   - Start with abundance weights
        #   - Boost hard negatives by `self.hard_neg_weight`
        #   - Optionally cap or normalize if you want stable loss scaling
        weights = abundance_weights * torch.where(hard_negatives, self.hard_neg_weight, 1.0)

        # Optional: Normalize weights to keep total scale consistent
        weights = weights / weights.mean()

        if DEBUG:
            predicted_probs = predicted_probs[0] # for testing
        else:
            predicted_probs = predicted_probs.squeeze() # shape [batch_size]

        #print("--predicted_probs", predicted_probs)

        ### Focal Loss: https://amaarora.github.io/posts/2020-06-29-FocalLoss.html ### 
        # Increase loss for minority misclassification (F = 1 but predicted as 0) and 
        # decreases loss for majority class misclassification (F = 0 but predicted as 1)
        # Our MassSpecGym training data is skewed with only 5% of molecules containing Fluorine
       
        bce_loss = nn.BCELoss(reduction='none')
        per_sample_loss = bce_loss(predicted_probs, true_values)

        # Apply sample-specific weighting
        weighted_loss = (weights * per_sample_loss).mean()

        # Continue with focal loss logic if needed
        pt = torch.exp(-per_sample_loss)
        targets = true_values.long()
        at = self.alpha.gather(0, targets.data.view(-1))
        F_loss = at * (1 - pt) ** self.gamma * weighted_loss
        loss = F_loss.mean()

        # targets = true_values.type(torch.long)
        # at = self.alpha.gather(0, targets.data.view(-1))
        # pt = torch.exp(-loss)
        # F_loss = at * (1 - pt)**self.gamma * loss
        return { 'loss': F_loss.mean() } 

    def on_batch_end(
        self, outputs: [], batch: dict, batch_idx: int, stage: Stage
    ) -> None:
        x = batch["spec"] # shape: [batch_size, num_peaks + 1, 2]
        halogen_vector_true = batch["mol"] # shape [batch_size]
        # updated predictions with the updated weights at the end of the batch
        pred_probs = self.forward(x) # shape [batch_size x 1]

        # thresholding
        halogen_vector_pred_binary = torch.where(pred_probs >= self.threshold, 1, 0)

        # Extract the 1st column --> fluorine predictions
        true_labels = halogen_vector_true[:, 0] # shape [batch_size]
        
        # make shape [batch_size x 1] into shape [batch_size]
        pred_bool_labels = halogen_vector_pred_binary.squeeze() # shape [batch_size]

        if stage.to_pref() == 'train_':
            self.train_precision.update(pred_bool_labels, true_labels)
            self.train_recall.update(pred_bool_labels, true_labels)
            self.train_accuracy.update(pred_bool_labels, true_labels)
        elif stage.to_pref() == 'val_':
            self.val_precision.update(pred_bool_labels, true_labels)
            self.val_recall.update(pred_bool_labels, true_labels)
            self.val_accuracy.update(pred_bool_labels, true_labels)

            ## debugging the false negatives ##
            # Collect optional metadata if available
            identifiers = batch.get("identifier", ["NA"] * len(true_labels))
            spectra = batch.get("spec", None)

            pred_probs_flat = pred_probs.squeeze()  # remove extra dimensions
            if pred_probs_flat.ndim == 0:           # single sample
                pred_probs_flat = [pred_probs_flat.item()]
            else:
                pred_probs_flat = pred_probs_flat.tolist()

            self.all_predicted_probs.extend(pred_probs_flat)
            self.all_true_labels.extend(true_labels.tolist())
            self.all_identifiers.extend(identifiers)

            if spectra is not None:
                self.all_spectra.extend([s.detach().cpu().numpy() for s in spectra])

        self.log_dict({ f"{stage.to_pref()}/loss": outputs['loss'] },
                prog_bar=True,
                on_epoch=True,
                batch_size=self.batch_size
        )

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

    def _reset_metrics_test(self):
        # Reset states for next epoch
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_accuracy.reset()

    def on_train_epoch_start(self) -> None:
        self._reset_metrics_train()

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics_val()

    def on_test_epoch_start(self) -> None:
        self._reset_metrics_test()

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
        self.log_dict({
                f"val_/precision": precision,
                f"val_/recall": recall,
                f"val_/accuracy": accuracy,
                f"val_/f1_score": f1_score
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
        df_fn = df_preds[(df_preds["true_label"] == 1) & (df_preds["pred_label"] == 0)]

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

        print(f"✅ True Positive identifiers written to {tp_filename}")
        print(f"✅ False Negative identifiers written to {fn_filename}")

    def on_test_epoch_end(self) -> None:
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()
        accuracy = self.test_accuracy.compute()
        f1_score = (2*precision*recall)/(precision + recall) if (precision + recall) != 0 else 0
        self.log_dict({
                f"test_/precision": precision,
                f"test_/recall": recall,
                f"test_/accuracy": accuracy,
                f"test_/f1_score": f1_score
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False
        )

import re

# removed adduct due to a str error
class TestMassSpecDataset(MassSpecDataset):

    def __getitem__(
        self, i: int, transform_spec: bool = True, transform_mol: bool = True
    ) -> dict:
        spec = self.spectra[i]
        metadata = self.metadata.iloc[i]
        mol = metadata["smiles"]

        # Apply all transformations to the spectrum
        item = {}

        item['abundance_weight'] = metadata['abundance_weight']
        # check the SMILES
        if re.search("f", mol, re.IGNORECASE):
            item['F'] = 1
        else:
            item['F'] = 0

        if transform_spec and self.spec_transform:
            if isinstance(self.spec_transform, dict):
                for key, transform in self.spec_transform.items():
                    item[key] = transform(spec) if transform is not None else spec
            else:
                item["spec"] = self.spec_transform(spec)
        else:
            item["spec"] = spec

        # Apply all transformations to the molecule
        if transform_mol and self.mol_transform:
            if isinstance(self.mol_transform, dict):
                for key, transform in self.mol_transform.items():
                    item[key] = transform(mol) if transform is not None else mol
            else:
                item["mol"] = self.mol_transform(mol)
        else:
            item["mol"] = mol

        # Add other metadata to the item
        item.update({
            k: metadata[k] for k in ["precursor_mz"] # removed adduct due to a str error
        })

        if self.return_mol_freq:
            item["mol_freq"] = metadata["mol_freq"]

        if self.return_identifier:
            item["identifier"] = metadata["identifier"]


        # TODO: this should be refactored
        for k, v in item.items():
            if not isinstance(v, str):
                item[k] = torch.as_tensor(v, dtype=self.dtype)
        
        return item

### Training Code ###
torch.set_float32_matmul_precision('high')
#python3.11 train_PFAS_model.py
# Init hyperparameters
max_epochs = 1
n_peaks = 60
threshold = 0.9
alpha = 0.25 # 0.25, 0.5, 0.75, 1 - found 0.25 as best
gamma = 0.75 #  0.25, 0.5, 0.75, 1 - found 0.75 as best
lr = 1e-5 # 3e-5, 5e-5 - found 1e-5 as best
hard_neg_weights = [1.5] # [2, 3, 3.5, 4]
num_iterations = 1

if DEBUG:
    batch_size = 1
else:
    batch_size = 64 # 64 is best (tried 128)

for weight in hard_neg_weights:
    for i in range(0, num_iterations):
        # Load dataset
        dataset = TestMassSpecDataset(
            spec_transform=SpecTokenizer(n_peaks=n_peaks),
            mol_transform = MolToPFASVector(),
            #pth='/teamspace/studios/this_studio/files/merged_massspec_nist20_with_fold.tsv'
            pth='/teamspace/studios/this_studio/files/pfas_labeled_with_inverse_abundance_weights.tsv'
        )

        # Init data module
        data_module = MassSpecDataModule(
            dataset=dataset,
            batch_size=batch_size,
            split_pth=split_pth,
            num_workers=4
        )

        # Init model
        model = HalogenDetectorDreamsTest(
            threshold=threshold,
            alpha=alpha,
            gamma=gamma,
            batch_size=batch_size,
            lr=lr,
            hard_neg_weight=weight
        )

        # initialise the wandb logger and name your wandb project
        wandb_logger = WandbLogger(project='PFASDetection-FocalLoss-MergedMassSpecNIST20OECDWith_PFASExceptions')
        trainer = Trainer(accelerator="auto", devices="auto", max_epochs=max_epochs, logger=wandb_logger, val_check_interval=0.2, strategy="ddp")

        # add your batch size to the wandb config
        wandb_logger.experiment.config["batch_size"] = batch_size
        wandb_logger.experiment.config["n_peaks"] = n_peaks
        wandb_logger.experiment.config["threshold"] = threshold
        wandb_logger.experiment.config["alpha"] = alpha
        wandb_logger.experiment.config["gamma"] = gamma
        wandb_logger.experiment.config["hard_neg_weight"] = weight

        # Validate before training
        data_module.prepare_data() 
        data_module.setup()  # Explicit call needed for validate before fit

        trainer.validate(model, datamodule=data_module)

        # # Train
        trainer.fit(model, datamodule=data_module)

        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()