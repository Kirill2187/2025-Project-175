import os
import yaml 
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import (
    load_config,
    setup_logging,
    set_seed,
    create_model,
    create_datasets,
    mnist_collate_fn,
    AdversarialOptimizer
)

def evaluate(model, data_loader, device):
    """Evaluate model on given data loader and return accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

def main(args):
    setup_logging(args.log_level)
    config = load_config(args.config)
    logging.info(f"Loaded configuration from {args.config}")

    set_seed(config['experiment']['seed'])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    
    train_dataset, val_dataset = create_datasets(
        config['data'],
        root='./data',
        transform_train=transform,
        transform_val=transform,
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=mnist_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=mnist_collate_fn
    )
    
    input_shape = (1, 28, 28)
    model = create_model(config, input_shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Model created and moved to device {device}")
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    optimizer_config = config['training']['optimizer']
    if optimizer_config['type'].lower() not in ['adamw', 'adversarial']:
        raise ValueError("Unknown optimizer type")
    
    optimizer_kwargs = {
        k: v
        for k, v in optimizer_config.items() if k != 'type'
    }
    if optimizer_config['type'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)
    else:
        optimizer = AdversarialOptimizer(model.parameters(), **optimizer_kwargs)
    
    num_epochs = config['training']['epochs']
    train_size = len(train_dataset)
    losses_array = np.zeros((num_epochs, train_size))
    val_accuracies = []
    
    logging.info(f"Beginning training for {num_epochs} epochs")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            indices = batch['index']
            
            def closure(weights, loss_scale):
                optimizer.zero_grad()
                outputs = model(images)
                losses = criterion(outputs, labels) * weights * loss_scale
                loss = losses.sum()
                loss.backward()
                return losses, loss
            closure.device = device
            
            if optimizer_config['type'].lower() == 'adversarial':
                optimizer.step(closure, indices)
                loss = optimizer.loss  
            else:
                optimizer.zero_grad()
                outputs = model(images)
                
                loss_vector = criterion(outputs, labels)
                loss = loss_vector.mean()
                loss.backward()
                optimizer.step()
            
                losses_array[epoch, indices] = loss_vector.detach().cpu().numpy()
            
            epoch_loss_sum += loss.item() * images.size(0)
        
        if optimizer_config['type'].lower() == 'adversarial':
            losses_array[epoch] = optimizer.pi.detach().cpu().numpy()
        avg_loss = epoch_loss_sum / train_size
        
        val_accuracy = evaluate(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        
        logging.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}, validation accuracy: {val_accuracy:.4f}")
    
    output_dir = config['experiment']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    name = "losses.npy" if optimizer_config['type'].lower() == 'adamw' else "weights.npy"
    losses_path = os.path.join(output_dir, name)
    np.save(losses_path, losses_array)
    logging.info(f"Saved array to {losses_path}")
    
    true_labels = train_dataset.true_labels
    corrupted_labels = train_dataset.corrupted_labels
    np.save(os.path.join(output_dir, "true_labels.npy"), true_labels)
    np.save(os.path.join(output_dir, "corrupted_labels.npy"), corrupted_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on corrupted MNIST using AdamW")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (e.g., DEBUG, INFO)")
    args = parser.parse_args()

    args.log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    main(args)
