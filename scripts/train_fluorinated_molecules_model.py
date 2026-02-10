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
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import CSVLogger

from torch import nn
import torch.nn.functional as F
from massspecgym.models.base import Stage
from dreams.api import PreTrainedModel
from dreams.models.dreams.dreams import DreaMS as DreaMSModel
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassAccuracy
import numpy as np
from rdkit import Chem
from massspecgym.data.transforms import MolToFluorinatedTypeVector
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import precision_score, recall_score, accuracy_score
from typing import Optional
import re

numpy.set_printoptions(threshold=sys.maxsize)

pl.seed_everything(0)

DEBUG = False

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


# Class names for reporting
CLASS_NAMES = ["Isolated CF2/CF3", "PFAS Chain", "Other Fluorine", "Non-Fluorinated"]
NUM_CLASSES = 4


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

        if 'abundance_weight' in metadata:
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


class FluorineDetectorDreamsTest(MassSpecGymModel):
    """
    Multi-class fluorinated-molecule classifier fine-tuned on the DreaMS transformer.

    Outputs logits for 4 mutually exclusive classes:
      0 - Isolated CF2
      1 - Isolated CF3
      2 - PFAS chain
      3 - None / Other
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        batch_size: int = 64,
        class_weights: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.num_classes = num_classes
        self.batch_size = batch_size

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        # Metrics
        self.train_precision = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.train_recall = MulticlassRecall(num_classes=num_classes, average='macro')
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')

        self.val_precision = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.val_recall = MulticlassRecall(num_classes=num_classes, average='macro')
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')

        # Per-class accuracy for detailed reporting
        self.val_accuracy_per_class = MulticlassAccuracy(num_classes=num_classes, average=None)

        # Storage for validation epoch-end analysis
        self.all_val_preds = []
        self.all_val_targets = []
        self.all_identifiers = []

        # Load pretrained DreaMS model
        self.main_model = PreTrainedModel.from_ckpt(
            ckpt_path="https://zenodo.org/records/10997887/files/ssl_model.ckpt?download=1",
            ckpt_cls=DreaMSModel,
            n_highest_peaks=60
        ).model.train()

        # Classification head: 1024-dim DreaMS output -> num_classes logits
        self.lin_out = nn.Linear(1024, num_classes)

    def forward(self, x):
        output_main_model = self.main_model(x)[:, 0, :]  # [B, 1024] (CLS token)
        logits = self.lin_out(output_main_model)          # [B, num_classes]
        return logits

    def step(self, batch: dict, stage: Stage) -> dict:
        x = batch["spec"]                      # [B, n_peaks+1, 2]
        targets = batch["mol"][:, 0].long()    # [B] class indices {0,1,2,3}

        logits = self.forward(x)               # [B, num_classes]

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=weight)(logits, targets)

        return {"loss": loss}

    def on_batch_end(self, outputs: dict, batch: dict, batch_idx: int, stage: Stage) -> None:
        x = batch["spec"]
        targets = batch["mol"][:, 0].long()    # [B]

        logits = self.forward(x)               # [B, num_classes]
        preds = torch.argmax(logits, dim=1)    # [B]

        if stage.to_pref() == 'train_':
            self.train_precision.update(preds, targets)
            self.train_recall.update(preds, targets)
            self.train_accuracy.update(preds, targets)

        elif stage.to_pref() == 'val_':
            self.val_precision.update(preds, targets)
            self.val_recall.update(preds, targets)
            self.val_accuracy.update(preds, targets)
            self.val_accuracy_per_class.update(preds, targets)

            identifiers = batch.get("identifier", ["NA"] * len(targets))
            self.all_val_preds.extend(preds.detach().cpu().numpy().tolist())
            self.all_val_targets.extend(targets.detach().cpu().numpy().tolist())
            self.all_identifiers.extend(identifiers)

        self.log_dict(
            {f"{stage.to_pref()}/loss": outputs["loss"]},
            prog_bar=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )

    def _reset_metrics_train(self):
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_accuracy.reset()

    def _reset_metrics_val(self):
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_accuracy.reset()
        self.val_accuracy_per_class.reset()
        self.all_val_preds = []
        self.all_val_targets = []
        self.all_identifiers = []

    def on_train_epoch_start(self) -> None:
        self._reset_metrics_train()

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics_val()

    def on_train_epoch_end(self) -> None:
        precision = self.train_precision.compute()
        recall = self.train_recall.compute()
        accuracy = self.train_accuracy.compute()
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        self.log_dict(
            {
                "train_/precision": precision,
                "train_/recall": recall,
                "train_/accuracy": accuracy,
                "train_/f1_score": f1_score,
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

    def on_validation_epoch_end(self) -> None:
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        accuracy = self.val_accuracy.compute()
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        self.log_dict(
            {
                "val_/precision": precision,
                "val_/recall": recall,
                "val_/accuracy": accuracy,
                "val_/f1_score": f1_score,
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        if len(self.all_val_preds) == 0:
            return

        # Per-class accuracy breakdown
        per_class_acc = self.val_accuracy_per_class.compute().cpu().numpy()
        print("\n=== Per-Class Validation Accuracy ===")
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            print(f"  {cls_name}: {per_class_acc[cls_idx]:.4f}")

        # Class distribution in validation set
        from collections import Counter
        target_counts = Counter(self.all_val_targets)
        pred_counts = Counter(self.all_val_preds)
        print("\n=== Validation Class Distribution ===")
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            true_n = target_counts.get(cls_idx, 0)
            pred_n = pred_counts.get(cls_idx, 0)
            print(f"  {cls_name}: true={true_n}, predicted={pred_n}")

        # Write misclassified identifiers
        df_preds = pd.DataFrame({
            "identifier": self.all_identifiers,
            "true_label": self.all_val_targets,
            "pred_label": self.all_val_preds,
        })

        misclassified = df_preds[df_preds["true_label"] != df_preds["pred_label"]]
        misclassified_file = "fluorine_misclassified_identifiers.txt"
        with open(misclassified_file, "w") as f:
            f.write("--- Misclassified Identifiers (true_class -> pred_class): ---\n")
            for _, row in misclassified.iterrows():
                true_name = CLASS_NAMES[int(row["true_label"])]
                pred_name = CLASS_NAMES[int(row["pred_label"])]
                f.write(f"  {row['identifier']}: {true_name} -> {pred_name}\n")
        print(f"\nMisclassified identifiers written to {misclassified_file}")


### Training Code ###
torch.set_float32_matmul_precision('high')
# python3 scripts/train_fluorinated_molecules_model.py

# Hyperparameters
max_epochs = 1
n_peaks = 60
lr = 1e-5
num_iterations = 1

if DEBUG:
    batch_size = 1
else:
    batch_size = 64

for i in range(0, num_iterations):
    # Load dataset with the new multi-class transform
    dataset = TestMassSpecDataset(
        spec_transform=SpecTokenizer(n_peaks=n_peaks),
        mol_transform=MolToFluorinatedTypeVector(),
        pth='/teamspace/studios/this_studio/files/merged_massspec_nist20_nist_new_with_fold.tsv'
    )

    # Init data module
    data_module = MassSpecDataModule(
        dataset=dataset,
        batch_size=batch_size,
        split_pth=split_pth,
        num_workers=4
    )

    print(f'learning_rate = {lr}')

    # Init model (no class_weights by default; pass computed weights to address imbalance)
    model = FluorineDetectorDreamsTest(
        num_classes=NUM_CLASSES,
        batch_size=batch_size,
        lr=lr
    )

    # Initialise the wandb logger
    wandb_logger = WandbLogger(project='FluorinatedMolecules-MultiClass-MergedMassSpecNIST20')

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=max_epochs,
        logger=wandb_logger,
        val_check_interval=0.2
    )

    # Add config to wandb
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["n_peaks"] = n_peaks
    wandb_logger.experiment.config["num_classes"] = NUM_CLASSES
    wandb_logger.experiment.config["lr"] = lr

    # Validate before training
    data_module.prepare_data()
    data_module.setup()

    trainer.validate(model, datamodule=data_module)

    # Train
    trainer.fit(model, datamodule=data_module)

    # Finish the wandb run
    wandb.finish()
