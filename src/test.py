import click
import torch
import logging
import random
import numpy as np
import re
import os

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset


@click.command()
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10', 'svhn', 'fashionmnist']), default='svhn')
@click.argument('data_path', type=click.Path(exists=True), default='../data')
@click.option('--normal_class', type=str, default="0")

def main(dataset_name, data_path, normal_class):

    normal_class = [int(i) for i in normal_class.split(',')]
    dataset = load_dataset(dataset_name, data_path, normal_class)
    # Test the dataset
    train_loader = dataset.train_set
    print(len(train_loader))
    test_loader = dataset.test_set

    for i, (img, target, index) in enumerate(train_loader):
        print(f"Image shape: {img.shape}, Target: {target}, Index: {index}")
        if i == 5:
            break


main()