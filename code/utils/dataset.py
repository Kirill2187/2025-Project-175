import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision import transforms
import numpy as np

import logging

class CorruptedImageDataset(Dataset):
    def __init__(self, root, dataset_name='mnist', train=True, corruption_config=None, transform=None, download=True):
        # Load the appropriate dataset based on dataset_name
        if (dataset_name.lower() == 'mnist'):
            self.dataset = MNIST(root=root, train=train, transform=transform, download=download)
        elif (dataset_name.lower() == 'cifar10'):
            self.dataset = CIFAR10(root=root, train=train, transform=transform, download=download)
        elif (dataset_name.lower() == 'cifar100'):
            self.dataset = CIFAR100(root=root, train=train, transform=transform, download=download)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Get dataset attributes
        self.dataset_name = dataset_name.lower()
        self.true_labels = np.array(self.dataset.targets)
        self.corrupted_labels = self.true_labels.copy()
        
        # Get number of classes based on the dataset
        if self.dataset_name == 'mnist' or self.dataset_name == 'cifar10':
            self.num_classes = 10
        elif self.dataset_name == 'cifar100':
            self.num_classes = 100
        
        if train and corruption_config is not None:
            for rule in corruption_config:
                if rule['type'] == 'flip':
                    # For each label choose a fixed random label
                    # Flip this label to the chosen label with probability p
                    p = rule['p']
                    
                    label_mapping = {}
                    for i in range(self.num_classes):
                        available_labels = list(range(self.num_classes))
                        available_labels.remove(i)
                        label_mapping[i] = np.random.choice(available_labels)
                    
                    logging.info(f"Flip mapping: {label_mapping}")
                    
                    # Apply the mapping with probability p
                    for idx in range(len(self.corrupted_labels)):
                        true_label = self.true_labels[idx]
                        if (np.random.rand() < p):
                            self.corrupted_labels[idx] = label_mapping[true_label]
                    
                    logging.info(f"Applied 'flip' corruption with p={p}")
                    
                elif rule['type'] == 'unif':
                    # Flip any label to a random label (including correct one) with probability p
                    p = rule['p']
                    
                    for idx in range(len(self.corrupted_labels)):
                        if (np.random.rand() < p):
                            # Choose a completely random label
                            self.corrupted_labels[idx] = np.random.randint(0, self.num_classes)
                    
                    logging.info(f"Applied 'unif' corruption with p={p}")
                    
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

def image_collate_fn(batch):
    """
    Custom collate function that merges a list of dictionary items into a single batch dictionary.
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
    Creates training and validation datasets for image classification tasks.
    
    The training dataset applies label corruptions as specified in data_config, while the validation dataset
    remains uncorrupted.
    
    Args:
        data_config (dict): Contains:
            - 'dataset': Name of dataset (e.g., 'mnist', 'cifar10', 'cifar100')
            - 'corruption': a list of corruption rules for the training set.
        root (str): Root directory for dataset data.
        transform_train: Transformations for training images.
        transform_val: Transformations for validation images.
        download (bool): Whether to download the dataset if not present.
    
    Returns:
        (train_dataset, val_dataset): A tuple of dataset objects.
    """
    dataset_name = data_config.get('dataset', 'mnist')
    supported_datasets = ['mnist', 'cifar10', 'cifar100']
    
    if dataset_name.lower() not in supported_datasets:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {supported_datasets}")
    
    corruption_config = data_config.get('corruption', None)
    
    train_dataset = CorruptedImageDataset(
        root=root,
        dataset_name=dataset_name,
        train=True,
        corruption_config=corruption_config,
        transform=transform_train,
        download=download
    )
    
    val_dataset = CorruptedImageDataset(
        root=root,
        dataset_name=dataset_name,
        train=False,
        corruption_config=None,
        transform=transform_val,
        download=download
    )
    
    return train_dataset, val_dataset
