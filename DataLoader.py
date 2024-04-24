import os
import pickle
import numpy as np
import torch
from torchvision import datasets, transforms

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    # x_train = np.array([data[0].numpy() for data in train_set])
    # y_train = np.array([data[1] for data in train_set])
    # x_test = np.array([data[0].numpy() for data in test_set])
    # y_test = np.array([data[1] for data in test_set])

    x_train = torch.stack([data[0] for data in train_set])
    y_train = torch.tensor([data[1] for data in train_set])
    x_test = torch.stack([data[0] for data in test_set])
    y_test = torch.tensor([data[1] for data in test_set])
    ### END CODE HERE
    
    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    x_test = [data[0] for data in test_set]
    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    split_idx = int(len(x_train) * train_ratio)
    x_train_new = x_train[:split_idx]
    y_train_new = y_train[:split_idx]
    x_valid = x_train[split_idx:]
    y_valid = y_train[split_idx:]

    # print(f"Type of x_valid after split: {type(x_valid)}")
    # print(f"Type of y_valid after split: {type(y_valid)}")

    # print(f"Shape of x_valid: {x_valid.shape}")
    # print(f"Shape of y_valid: {y_valid.shape}")

    # x_valid = torch.tensor(x_valid, dtype=torch.float32) if not isinstance(x_valid, torch.Tensor) else x_valid
    # y_valid = torch.tensor(y_valid, dtype=torch.int64) if not isinstance(y_valid, torch.Tensor) else y_valid

    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

