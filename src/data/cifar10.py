import torchvision
from src.components import Datasets
from torchvision.transforms import Compose, Normalize, ToTensor

class CIFAR10TrainDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10TrainDataset, self).__init__(root, train, transform, target_transform, download)

class CIFAR10ValDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=False):
        super(CIFAR10ValDataset, self).__init__(root, train, transform, target_transform, download)

def get_datasets(root):
    transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.23, 0.225))])

    return Datasets(
        train = CIFAR10TrainDataset(root, transform=transform),
        val = CIFAR10ValDataset(root, transform=transform),
        test = None
    )