from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10, SVHN, MNIST, FashionMNIST
from base.torchvision_dataset import TorchvisionDataset
from torchvision.datasets.vision import VisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import numpy as np
import torchvision.transforms as transforms


class BTNDataset(TorchvisionDataset):

    def __init__(self, root: str, ind: str, ood: str):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier

        transform = transforms.Compose([transforms.ToTensor()])
        target_transform = None
        
        # InD
        assert ind in ['mnist', 'fashionmnist', 'cifar10'], f"Unknown dataset: {ind}"
        if ind == 'cifar10':
            c=3
            ind_train = CIFAR10(root=self.root, train=True, download=True, transform=transform, target_transform=target_transform)
            ind_test = CIFAR10(root=self.root, train=False, download=True, transform=transform, target_transform=target_transform)
            train_set = ind_train.data
            ind_test_set = ind_test.data
            # print(f"InD Train set shape: {ind_train.data.shape}, InD Test set shape: {ind_test.data.shape}")
        elif ind == 'mnist':
            c=1
            ind_train = MNIST(root=self.root, train=True, download=True, transform=transform, target_transform=target_transform)
            ind_test = MNIST(root=self.root, train=False, download=True, transform=transform, target_transform=target_transform)
            train_set = ind_train.train_data
            ind_test_set = ind_test.test_data
        print(f"InD Train set shape: {ind_train.data.shape}, InD Test set shape: {ind_test.data.shape}")

        # OOD
        assert ood in ['mnist', 'fashionmnist', 'svhn'], f"Unknown dataset: {ood}"
        if ood == 'svhn':
            ood_test = SVHN(root=self.root, split='test', download=True, transform=transform, target_transform=target_transform)
            ood_test.data = ood_test.data.transpose((0, 2, 3, 1))  # SVHN data is in (N, C, H, W) format
            # print(f"OoD Test set shape: {ood_test.data.shape}")
            ood_test_set = ood_test.data
        elif ood == 'fashionmnist':
            ood_test = FashionMNIST(root=self.root, train=False, download=True, transform=transform, target_transform=target_transform)
            ood_test_set = ood_test.test_data
        print(f"OoD Test set shape: {ood_test.data.shape}")

        # Train set creation
        train_labels = np.zeros(len(ind_train))
        self.train_set = MyBTNDataset(root=root, transform=transform, target_transform=target_transform, c=c)
        self.train_set.data = train_set
        self.train_set.labels = train_labels

        # Test set creation
        test_set = np.concatenate((ind_test_set, ood_test_set), axis=0)
        label_ind = np.zeros(len(ind_test))
        label_ood = np.ones(len(ood_test))
        print(f"Test Set Size: {len(test_set)}, InD: {len(ind_test)}, OOD: {len(ood_test)}")
        test_labels = np.concatenate((label_ind, label_ood))
        self.test_set = MyBTNDataset(root=root, transform=transform, target_transform=target_transform, c=c)
        self.test_set.data = test_set
        self.test_set.labels = test_labels


class MyBTNDataset(VisionDataset):
    """Torchvision dataset class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self,  c=3, *args, **kwargs):
        super(MyBTNDataset, self).__init__(*args, **kwargs)
        self.c = c

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.c != 1:
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed
