from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .fashionmnist import FashionMNIST_Dataset
from .svhn import SVHN_Dataset
from .btn_dset import BTNDataset


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'fashionmnist', 'svhn', 'cifar10-svhn')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'fashionmnist':
        dataset = FashionMNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == "svhn":
        dataset = SVHN_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == "cifar10-svhn":
        ind = 'cifar10'
        ood = 'svhn'
        dataset = BTNDataset(root=data_path, ind=ind, ood=ood)

    return dataset
