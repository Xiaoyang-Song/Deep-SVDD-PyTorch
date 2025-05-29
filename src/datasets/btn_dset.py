from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10, SVHN, MNIST, FashionMNIST
from base.torchvision_dataset import TorchvisionDataset
from torchvision.datasets.vision import VisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
import numpy as np
import torch
import torchvision.transforms as transforms
from .imagenet import *


class BTNDataset(TorchvisionDataset):

    def __init__(self, root: str, ind: str, ood: str):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier

        ind_transform = transforms.Compose([transforms.ToTensor()])
        ind_target_transform = None
        # InD
        assert ind in ['mnist', 'fashionmnist', 'cifar10',
                       'mnist-32'], f"Unknown dataset: {ind}"
        if ind == 'cifar10':
            c=3
            ind_train = CIFAR10(root=self.root, train=True, download=True, transform=ind_transform, target_transform=ind_target_transform)
            ind_test = CIFAR10(root=self.root, train=False, download=True, transform=ind_transform, target_transform=ind_target_transform)
            train_set = ind_train.data
            ind_test_set = ind_test.data
            # print(f"InD Train set shape: {ind_train.data.shape}, InD Test set shape: {ind_test.data.shape}")
        elif ind == 'mnist':
            c=1
            ind_train = MNIST(root=self.root, train=True, download=True, transform=ind_transform, target_transform=ind_target_transform)
            ind_test = MNIST(root=self.root, train=False, download=True, transform=ind_transform, target_transform=ind_target_transform)
            train_set = ind_train.train_data
            ind_test_set = ind_test.test_data

        elif ind == 'mnist-32':
            c=3
            print("InD MNIST-32 dataset selected - Processing...")
            # ind_transform = transforms.Compose([ transforms.Resize((32, 32)), 
            #                                 transforms.Grayscale(num_output_channels=3),
            #                                 transforms.ToTensor()])
            transform = transforms.Compose([transforms.Resize((32, 32)), 
                                                        transforms.Grayscale(num_output_channels=3),
                                                        transforms.ToTensor()])
            ind_train = MNIST(root=self.root, download=True, train=True, transform=transform)
            ind_test = MNIST(root=self.root, download=True, train=False, transform=transform)
            # n_test=10000
            # test_set = Subset(test_set, range(n_test))
            # train_set = ind_train.train_data
            # ind_test_set = ind_test.test_data

            train_set = (torch.stack([img for img, _ in ind_train]).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            ind_test_set = (torch.stack([img for img, _ in ind_test]).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            # print(train_set[0])

        print(f"InD Train set shape: {train_set.shape}, InD Test set shape: {ind_test_set.shape}")

        ood_transform = transforms.Compose([transforms.ToTensor()])
        ood_target_transform = None
        # OOD
        assert ood in ['mnist', 'fashionmnist', 'svhn',
                       'fashionmnist-32', 'cifar10', 'imagenet'], f"Unknown dataset: {ood}"
        if ood == 'svhn':
            ood_test = SVHN(root=self.root, split='test', download=True, transform=ood_transform, target_transform=ood_target_transform)
            ood_test.data = ood_test.data.transpose((0, 2, 3, 1))  # SVHN data is in (N, C, H, W) format
            # print(f"OoD Test set shape: {ood_test.data.shape}")
            ood_test_set = ood_test.data
        elif ood == 'fashionmnist':
            ood_test = FashionMNIST(root=self.root, train=False, download=True, transform=ood_transform, target_transform=ood_target_transform)
            ood_test_set = ood_test.test_data

        elif ood == 'fashionmnist-32':
            print("OoD FashionMNIST-32 dataset selected - Processing...")
            transform = transforms.Compose([transforms.Resize((32, 32)), 
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor()])
            ood_test = FashionMNIST(root=self.root, download=True, train=False, transform=transform)
            # n_test=5000
            # tset = Subset(tset, range(n_test))
            # ood_test_set = ood_test.test_data
            ood_test_set = (torch.stack([img for img, _ in ood_test]).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        
        elif ood == 'cifar10':
            ood_test = CIFAR10(root=self.root, train=False, download=True, transform=ood_transform, target_transform=ood_target_transform)
            ood_test_set = ood_test.data
            # print(ood_test_set[0])

        elif ood == 'imagenet':
            # raise NotImplementedError("Imagenet dataset is not implemented yet.")
            train_set, test_set = imagenet10_set_loader(dset_id=0)

        print(f"OoD Test set shape: {ood_test_set.shape}")

        # Train set creation
        train_labels = np.zeros(len(ind_train))
        self.train_set = MyBTNDataset(root=root, transform=ind_transform, target_transform=ind_target_transform, c=c)
        self.train_set.data = train_set
        self.train_set.labels = train_labels

        # Test set creation
        test_set = np.concatenate((ind_test_set, ood_test_set), axis=0)
        label_ind = np.zeros(len(ind_test))
        label_ood = np.ones(len(ood_test))
        print(f"Test Set Size: {len(test_set)}, InD: {len(ind_test)}, OOD: {len(ood_test)}")
        test_labels = np.concatenate((label_ind, label_ood))
        self.test_set = MyBTNDataset(root=root, transform=ood_transform, target_transform=ood_target_transform, c=c)
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
        # print(type(img))
        if self.c != 1:
            if type(img) == np.ndarray:
                img = Image.fromarray(img)
            else:
                img = Image.fromarray(img.numpy())
        else:
            if type(img) == np.ndarray:
                img = Image.fromarray(img, mode='L')
            else:
                img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed
