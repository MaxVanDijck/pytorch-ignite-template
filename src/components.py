from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

@dataclass
class Datasets:
    """Class for keeping track of datasets"""
    train: Dataset = None
    val: Dataset = None
    test: Dataset = None

@dataclass
class Dataloaders:
    """Class for keeping track of dataloaders"""
    train: DataLoader = None
    val: DataLoader = None
    test: DataLoader = None