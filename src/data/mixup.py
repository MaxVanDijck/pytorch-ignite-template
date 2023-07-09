import random

import torch
from torch.utils.data import Dataset


class MixupDataset(Dataset):
    def __init__(self, dataset_1: Dataset, dataset_2: Dataset, num_classes: int):
        assert len(dataset_1) == len(dataset_2)
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.num_classes = num_classes
        self.beta = torch.distributions.beta.Beta(torch.tensor([1.0]), torch.tensor([1.0]))

        # keep track of the number of datapoints accessed 
        # shuffle the index access map of dataset2 when needed
        self.index_map = [i for i in range(len(self.dataset_1))]
        random.shuffle(self.index_map)
        self.num_accessed = 0
        self.dataset_len = len(dataset_1)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, index):
        # get samples
        x1, y1 = self.dataset_1.__getitem__(index)
        dataset2_index = self.index_map[index]
        x2, y2 = self.dataset_2.__getitem__(dataset2_index)

        y1 = torch.nn.functional.one_hot(torch.tensor(y1), num_classes=self.num_classes)
        y2 = torch.nn.functional.one_hot(torch.tensor(y2), num_classes=self.num_classes)

        # mixup
        lam = self.beta.sample()
        x = lam * x1 + (1-lam) * x2
        y = lam * y1 + (1-lam) * y2

        # shuffle the pairings if all elements have been accessed
        self.num_accessed += 1
        if self.num_accessed >= self.dataset_len:
            self.num_accessed = 0
            random.shuffle(self.index_map)
        
        return x, y
