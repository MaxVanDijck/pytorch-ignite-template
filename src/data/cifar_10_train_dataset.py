import torchvision

class CIFAR10TrainDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10TrainDataset, self).__init__(root, train, transform, target_transform, download)
