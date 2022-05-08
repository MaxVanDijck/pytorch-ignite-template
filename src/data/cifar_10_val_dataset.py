import torchvision

class CIFAR10ValDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=False, transform=None, target_transform=None, download=True):
        super(CIFAR10ValDataset, self).__init__(root, train, transform, target_transform, download)