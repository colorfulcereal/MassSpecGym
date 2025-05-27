# filepath: /Users/ramsindhu/MassSpecGym/fluorine_dataset.py
import numpy as np
from massspecgym.data import MassSpecDataset
from massspecgym.data.transforms import MolToHalogensVector

class FluorineBalancedDataset(MassSpecDataset):
    """
    Dataset containing balanced set of Fluorine training examples
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self):
        super().load_data()
        checker = MolToHalogensVector()
        num_negatives = 0
        num_positives = 0
        indices_to_drop = []
        for idx, row in self.metadata.iterrows():
            halogen_vector = checker.from_smiles(row.get("smiles"))
            if halogen_vector[0] == 0 and row.get("fold") == 'train': 
                if num_negatives >= 8718:
                    indices_to_drop.append(idx)
                else:
                    num_negatives += 1

        self.metadata = self.metadata.drop(indices_to_drop).reset_index(drop=True)
        self.spectra = self.spectra.drop(indices_to_drop).reset_index(drop=True)
        print("---train", len(self.metadata[self.metadata['fold'] == 'train']))
        print("---val", len(self.metadata[self.metadata['fold'] == 'val']))