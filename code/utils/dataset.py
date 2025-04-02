import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np

import logging

class CorruptedMNISTDataset(Dataset):
    def __init__(self, root, train=True, corruption_config=None, transform=None, download=True):
        self.dataset = MNIST(root=root, train=train, transform=transform, download=download)
        self.true_labels = np.array(self.dataset.targets)
        self.corrupted_labels = self.true_labels.copy()
        
        if train and corruption_config is not None:
            for rule in corruption_config:
                if rule['type'] == 'swap':
                    # For samples with label equal to rule['from'], with probability p swap to rule['to'].
                    from_val = rule['from']
                    to_val = rule['to']
                    p = rule['p']
                    indices = np.where(self.true_labels == from_val)[0]
                    random_vals = np.random.rand(len(indices))
                    for i, idx in enumerate(indices):
                        if random_vals[i] < p:
                            self.corrupted_labels[idx] = to_val
                elif rule['type'] == 'random_swap':
                    # For samples with label equal to rule['from'], with probability p swap to a random label (!= true label).
                    from_val = rule['from']
                    p = rule['p']
                    indices = np.where(self.true_labels == from_val)[0]
                    random_vals = np.random.rand(len(indices))
                    for i, idx in enumerate(indices):
                        if random_vals[i] < p:
                            possible = list(range(10))
                            possible.remove(int(self.true_labels[idx]))
                            self.corrupted_labels[idx] = np.random.choice(possible)
                else:
                    raise ValueError(f"Unsupported corruption type: {rule['type']}")
        
        self.is_correct = self.corrupted_labels == self.true_labels
        
    def change_labels(self, indices, new_labels):
        is_correct_before = self.is_correct[indices].copy()
        
        self.corrupted_labels[indices] = new_labels
        self.is_correct[indices] = self.corrupted_labels[indices] == self.true_labels[indices]
        
        is_correct_after = self.is_correct[indices]
        
        fp = np.sum(is_correct_before & ~is_correct_after)
        tp = np.sum(~is_correct_before & is_correct_after)
        total = len(indices)
        
        logging.info(f"Labels changed for {total} samples: {fp} FP, {tp} TP.")
    
    def __getitem__(self, index):
        image, _ = self.dataset[index]
        corrupted_label = int(self.corrupted_labels[index])
        true_label = int(self.true_labels[index])
        return {
            'image': image,
            'label': corrupted_label,
            'true_label': true_label,
            'index': index
        }
    
    def __len__(self):
        return len(self.dataset)

def mnist_collate_fn(batch):
    """
    Custom collate function that merges a list of dictionary items into a single batch dictionary.
    
    The returned dictionary contains:
        - 'image': a tensor of stacked images.
        - 'label': a tensor of labels.
        - 'true_label': a tensor of true labels.
        - 'index': a tensor of sample indices.
    """
    images = torch.stack([item['image'] for item in batch], dim=0)
    labels = torch.tensor([item['label'] for item in batch])
    true_labels = torch.tensor([item['true_label'] for item in batch])
    indices = torch.tensor([item['index'] for item in batch])
    
    return {
        'image': images,
        'label': labels,
        'true_label': true_labels,
        'index': indices
    }

def create_datasets(data_config, root='./data', transform_train=None, transform_val=None, download=True):
    """
    Creates training and validation datasets for MNIST.
    
    The training dataset applies label corruptions as specified in data_config, while the validation dataset
    remains uncorrupted.
    
    Args:
        data_config (dict): Contains:
            - 'dataset': must be 'mnist' (currently only MNIST is supported).
            - 'corruption': a list of corruption rules for the training set.
        root (str): Root directory for MNIST data.
        transform_train: Transformations for training images.
        transform_val: Transformations for validation images.
        download (bool): Whether to download the dataset if not present.
    
    Returns:
        (train_dataset, val_dataset): A tuple of dataset objects.
    """
    dataset_name = data_config.get('dataset', None)
    if dataset_name != 'mnist':
        raise ValueError("Currently only MNIST is supported.")
    
    corruption_config = data_config.get('corruption', None)
    
    train_dataset = CorruptedMNISTDataset(
        root=root,
        train=True,
        corruption_config=corruption_config,
        transform=transform_train,
        download=download
    )
    
    val_dataset = CorruptedMNISTDataset(
        root=root,
        train=False,
        corruption_config=None,
        transform=transform_val,
        download=download
    )
    
    return train_dataset, val_dataset
