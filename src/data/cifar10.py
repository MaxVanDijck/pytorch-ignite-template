import torchvision
from src.components import Datasets

class CIFAR10TrainDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10TrainDataset, self).__init__(root, train, transform, target_transform, download)

class CIFAR10ValDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=False):
        super(CIFAR10ValDataset, self).__init__(root, train, transform, target_transform, download)

def get_datasets(root, transform):
    return Datasets(
        train = CIFAR10TrainDataset(root, transform=transform),
        val = CIFAR10ValDataset(root, transform=transform),
        test = None
    )