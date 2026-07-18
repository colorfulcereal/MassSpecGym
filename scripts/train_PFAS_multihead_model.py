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
from massspecgym.models.pfas import HalogenDetectorDreamsMultiHead
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
from massspecgym.data.transforms import MolToPFASVector
import numpy as np
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import time
import re

numpy.set_printoptions(threshold=sys.maxsize)

seed = int(time.time())
pl.seed_everything(seed)

DEBUG = False

if DEBUG:
    split_pth = None  # combined TSV's own 'fold' column drives the split (see below)
else:
    split_pth = None

# Check if MPS is available, otherwise use CUDA
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
else:
    mps_device = None


def ion_mode_idx_from_true_label(ion_mode_true) -> int:
    """0 = negative, 1 = positive, 2 = unknown. Uses the dataset's real
    ion_mode_true bonus column (ground truth for Enveda-180, derived from
    adduct for NIST-PFAS -- see prepare_combined_nistpfas_enveda180_dataset.py)
    rather than re-deriving from adduct, since the real label is already
    in the file (validated 0% mismatch against the adduct-heuristic,
    dreams-pfas-workshop-paper.md section 14)."""
    m = str(ion_mode_true).lower()
    if "positive" in m:
        return 1
    if "negative" in m:
        return 0
    return 2


class MultiHeadPFASDataset(MassSpecDataset):
    """
    Dataset for the multi-head fluorine/PFAS-type model. Reads the
    precomputed bonus columns already in
    combined_nistpfas_enveda180_with_fold.tsv (has_chain_pfas,
    has_isolated_cf2, has_isolated_cf3, ion_mode_true) rather than
    recomputing them via RDKit; derives has_f (same regex convention used
    everywhere in this repo) and is_oecd_pfas on the fly (cheap).
    """

    def __getitem__(
        self, i: int, transform_spec: bool = True, transform_mol: bool = True
    ) -> dict:
        spec = self.spectra[i]
        metadata = self.metadata.iloc[i]
        mol = metadata["smiles"]

        item = {}

        item['has_f'] = 1 if re.search("f", mol, re.IGNORECASE) else 0
        item['ion_mode'] = ion_mode_idx_from_true_label(metadata["ion_mode_true"])
        item['has_chain_pfas'] = int(metadata["has_chain_pfas"])
        item['has_isolated_cf2'] = int(metadata["has_isolated_cf2"])
        item['has_isolated_cf3'] = int(metadata["has_isolated_cf3"])
        item['is_oecd_pfas'] = int(
            item['has_chain_pfas'] or item['has_isolated_cf2'] or item['has_isolated_cf3']
        )

        if transform_spec and self.spec_transform:
            if isinstance(self.spec_transform, dict):
                for key, transform in self.spec_transform.items():
                    item[key] = transform(spec) if transform is not None else spec
            else:
                item["spec"] = self.spec_transform(spec)
        else:
            item["spec"] = spec

        if transform_mol and self.mol_transform:
            if isinstance(self.mol_transform, dict):
                for key, transform in self.mol_transform.items():
                    item[key] = transform(mol) if transform is not None else mol
            else:
                item["mol"] = self.mol_transform(mol)
        else:
            item["mol"] = mol

        item.update({
            k: metadata[k] for k in ["precursor_mz"]
        })

        if self.return_mol_freq:
            item["mol_freq"] = metadata["mol_freq"]

        if self.return_identifier:
            item["identifier"] = metadata["identifier"]

        for k, v in item.items():
            if not isinstance(v, str):
                item[k] = torch.as_tensor(v, dtype=self.dtype)

        return item


### Training Code ###
torch.set_float32_matmul_precision('high')
# python3.11 train_PFAS_multihead_model.py
# Init hyperparameters
max_epochs = 1
n_peaks = 60
threshold = 0.5  # multi-head, not yet threshold-tuned per the single-task 0.2 convention
lr = 1e-5
num_iterations = 1
random_init = False  # Set to True to ablate transfer learning (random DreaMS weights)
hidden_dim = 128  # Option B FFN hidden layer size
loss_weights = (1.0, 1.0, 1.0)  # (has_f, oecd_pfas, subtype)

# Larger than the single-task scripts' default 64: Head 3's positive rate
# is low (~6.6% of molecules are is_oecd_pfas), so a batch of 64 would
# frequently contain 0 or very few Head-3-eligible examples. Cheap to
# increase since DreaMS's forward pass dominates compute regardless of
# head count.
if DEBUG:
    batch_size = 1
else:
    batch_size = 128

for i in range(0, num_iterations):
    dataset = MultiHeadPFASDataset(
        spec_transform=SpecTokenizer(n_peaks=n_peaks),
        mol_transform=MolToPFASVector(),
        pth='/teamspace/studios/this_studio/files/combined_nistpfas_enveda180_with_fold.tsv',
    )

    data_module = MassSpecDataModule(
        dataset=dataset,
        batch_size=batch_size,
        split_pth=split_pth,
        num_workers=4
    )

    print(f'learning_rate = {lr}')
    model = HalogenDetectorDreamsMultiHead(
        seed=seed,
        hidden_dim=hidden_dim,
        loss_weights=loss_weights,
        threshold=threshold,
        batch_size=batch_size,
        lr=lr,
        random_init=random_init,
    )

    init_tag = "RandomInit" if random_init else "Pretrained"
    wandb_logger = WandbLogger(project=f'PFASDetection-MultiHead-CombinedNISTPFASEnveda180')

    trainer = Trainer(
            accelerator="auto",
            devices="auto",
            max_epochs=max_epochs,
            logger=wandb_logger,
            val_check_interval=0.2)

    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["n_peaks"] = n_peaks
    wandb_logger.experiment.config["threshold"] = threshold
    wandb_logger.experiment.config["random_init"] = random_init
    wandb_logger.experiment.config["hidden_dim"] = hidden_dim
    wandb_logger.experiment.config["loss_weights"] = loss_weights

    data_module.prepare_data()
    data_module.setup()

    trainer.validate(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)

    wandb.finish()
