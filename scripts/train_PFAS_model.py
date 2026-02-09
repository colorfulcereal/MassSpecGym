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
from massspecgym.models.pfas import HalogenDetectorDreamsTest
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
from massspecgym.data.transforms import MolToHalogensVector, MolToPFASVector, MolToIsolatedCF3Vector, MolToIsolatedCF2Vector
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import roc_auc_score, average_precision_score

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

### Training Code ###
torch.set_float32_matmul_precision('high')
#python3.11 train_PFAS_model.py
# Init hyperparameters
max_epochs = 1
n_peaks = 60
threshold = 0.9
# alpha = 1, gamma = 0 is BCELoss
alpha = 1
gamma = 0
lr = 1e-5 # found 1e-5 as best
num_iterations = 1
loss='bce'

if DEBUG:
    batch_size = 1
else:
    batch_size = 64

for i in range(0, num_iterations):
    # Load dataset
    dataset = TestMassSpecDataset(
        spec_transform=SpecTokenizer(n_peaks=n_peaks),
        mol_transform = MolToIsolatedCF2Vector(),
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
    # Init model
    model = HalogenDetectorDreamsTest(
        threshold=threshold,
        alpha=alpha,
        gamma=gamma,
        batch_size=batch_size,
        lr=lr,
        loss=loss
    )

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(project='HalogenDetection-FocalLoss-MergedMassSpecNIST20_NISTNew_NormalPFAS')
    #wandb_logger = WandbLogger(project='PFASDetection-FocalLoss-MergedMassSpecNIST20OECDWith_PFASExceptions')
    #wandb_logger = WandbLogger(project='HalogenDetection-FocalLoss-MergedMassSpecNIST20')

    trainer = Trainer(
            accelerator="auto", 
            devices="auto", 
            max_epochs=max_epochs, 
            logger=wandb_logger, 
            val_check_interval=0.2)

    # add your batch size to the wandb config
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["n_peaks"] = n_peaks
    wandb_logger.experiment.config["threshold"] = threshold
    wandb_logger.experiment.config["alpha"] = alpha
    wandb_logger.experiment.config["gamma"] = gamma
    wandb_logger.experiment.config["loss"] = loss

    # Validate before training
    data_module.prepare_data() 
    data_module.setup()  # Explicit call needed for validate before fit

    trainer.validate(model, datamodule=data_module)

    # # Train
    trainer.fit(model, datamodule=data_module)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()