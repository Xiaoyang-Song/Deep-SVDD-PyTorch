from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST, SVHN
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization
from collections import Counter
import torchvision.transforms as transforms
import numpy as np


class SVHN_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        if type(normal_class) == int:
            self.outlier_classes.remove(normal_class)
        else:
            self.outlier_classes = list(set(self.outlier_classes) - set(normal_class))
        print(f"Outlier classes: {self.outlier_classes}")


        transform = transforms.Compose([transforms.ToTensor()])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MySVHN(root=self.root, split='train', download=True, transform=transform, target_transform=target_transform)
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.labels, self.normal_classes)
        # print(Counter(train_set.labels))
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MySVHN(root=self.root, split='test', download=True, transform=transform, target_transform=target_transform)


class MySVHN(SVHN):
    """Torchvision SVHN class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MySVHN, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the SVHN class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """

        img, target = self.data[index], self.labels[index]
        # print(img.shape)
        # print(type(img))

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index  # only line changed

