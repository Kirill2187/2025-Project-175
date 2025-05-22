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
from tqdm import tqdm

from utils import (
    load_config,
    setup_logging,
    set_seed,
    create_model,
    create_datasets,
    image_collate_fn,
    AdversarialOptimizer
)

from utils.config import deep_merge, resolve_variables
from utils.correction import create_correction_strategy
from utils.augument import get_transform
from utils.selc import SELCLoss

def evaluate(model, data_loader, device):
    """Evaluate model on given data loader and return accuracy."""
    model.eval()
    correct = 0
    total = 0
    preds = np.zeros(len(data_loader.dataset))
    losses = np.zeros(len(data_loader.dataset))
    criterion = nn.CrossEntropyLoss(reduction='none')
    preds.fill(-1)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            if len(outputs) == 3:
                    outputs, _, _ = outputs
            
            _, predicted = torch.max(outputs.data, 1)
            preds[batch['index']] = predicted.cpu().numpy()
            losses[batch['index']] = criterion(outputs, labels).cpu().numpy()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return preds, losses, accuracy

def main(args):
    setup_logging(args.log_level)
    config = load_config(f'configs/runs/{args.config}.yaml')
    logging.info(f"Loaded configuration from {args.config}")

    if args.seed is not None:
        set_seed(args.seed)
                
    optimizer_config = load_config(f'configs/optimizers/{args.optimizer}.yaml')
    config['training'] = deep_merge(config['training'], optimizer_config)
    try:
        model_config = load_config(f'configs/models/{args.model}.yaml')
    except FileNotFoundError:
        model_config = {'model': args.model}
    config = deep_merge(config, model_config)
    
    dataset_config = load_config(f'configs/datasets/{args.dataset}.yaml')
    if args.corruption is not None:
        dataset_config['data']['corruption'].append({
            'type': args.corruption,
            'p': args.noise
        })
    
    config = deep_merge(config, dataset_config)
        
    config = resolve_variables(config)
    
    logging.debug(f"Resolved configuration: {config}")
        
    train_transform, val_transform = get_transform(config['data']['dataset'])
    train_dataset, val_dataset = create_datasets(
        config['data'],
        root='./data',
        transform_train=train_transform,
        transform_val=val_transform,
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=image_collate_fn,
        num_workers=8,
        prefetch_factor=4,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=image_collate_fn,
        num_workers=8,
        prefetch_factor=4
    )
    
    input_shape = (1, 28, 28) if config['data']['dataset'] == 'mnist' else (3, 32, 32)
    model = create_model(config, input_shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Model created and moved to device {device}")
    
    if args.weights is not None:
        logging.info(f"Loading object weights from {args.weights}")
        weights = np.load(args.weights, allow_pickle=True)
    else:
        weights = None
    
    if args.loss == 'ce':
        # Just CrossEntropyLoss
        criterion = SELCLoss(
            labels=train_dataset.corrupted_labels,
            num_classes=train_dataset.num_classes,
            es=int(1e8),
            reduction='none',
            weights=weights
        )
    elif args.loss == 'selc':
        assert weights is None
        criterion = SELCLoss(
            labels=train_dataset.corrupted_labels,
            num_classes=train_dataset.num_classes,
            momentum=0.9,
            es=40,
            reduction='none',
        )
    
    correction_strategy = create_correction_strategy(config, model, train_dataset)
    
    optimizer_config = config['training']['optimizer']
    if optimizer_config['type'].lower() not in ['adamw', 'adversarial', 'sgd']:
        raise ValueError("Unknown optimizer type")
    
    optimizer_kwargs = {
        k: v
        for k, v in optimizer_config.items() if k != 'type'
    }
    
    optimizer = None
        
    if optimizer_config['type'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)
    elif optimizer_config['type'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), **optimizer_kwargs)
    else:
        optimizer = AdversarialOptimizer(model.parameters(), **optimizer_kwargs)
    
    if 'annealing' in config['training']:
        annealing_config = config['training']['annealing']
        kwargs = {k: v for k, v in annealing_config.items() if k != 'type'}
        if annealing_config['type'].lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                **kwargs
            )
        elif annealing_config['type'].lower() == 'step':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                **kwargs
            )
        else:
            raise ValueError("Unknown annealing type")
    else:
        scheduler = None
    
    num_epochs = config['training']['epochs']
    train_size = len(train_dataset)
    losses_array = np.zeros((num_epochs, train_size))
    labels_history = np.zeros((num_epochs, train_size))
    train_predictions = np.zeros((num_epochs, train_size)) 
    train_predictions.fill(-1)
    val_accuracies = []
    train_accuracies = []
    
    exp_name = args.name
    if exp_name is None:
        exp_name = f"{args.optimizer}_{args.model}_{args.dataset}_{args.loss}"
        if args.corruption is not None:
            exp_name += f"_{args.corruption}_{args.noise}"
        if weights is not None:
            exp_name += f"_weights"
        exp_name += f"_{args.seed}"
        
    
    output_dir = f'outputs/{exp_name}'
    if os.path.exists(output_dir):
        logging.critical(f"Output directory {output_dir} already exists, continue? (y/n)")
        if input().strip().lower() != 'y':
            logging.info("Exiting...")
            return
    os.makedirs(output_dir, exist_ok=True)
    losses_file_name = "losses.npy"
    losses_path = os.path.join(output_dir, losses_file_name)
    val_accuracies_path = os.path.join(output_dir, "val_accuracies.npy")
    train_accuracies_path = os.path.join(output_dir, "train_accuracies.npy")
    
    true_labels = train_dataset.true_labels
    corrupted_labels = train_dataset.corrupted_labels
    np.save(os.path.join(output_dir, "true_labels.npy"), true_labels)
    np.save(os.path.join(output_dir, "corrupted_labels.npy"), corrupted_labels)
    
    logging.info(f"Beginning training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        model.train()
        predictions = np.zeros(len(train_loader.dataset))
        epoch_loss_sum = 0.0
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        for batch in tqdm(train_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            indices = batch['index']
            
            def closure(weights, loss_scale):
                optimizer.zero_grad()
                outputs = model(images)
                if len(outputs) == 3:
                    outputs, _, _ = outputs
                # predictions[indices] = outputs.argmax(dim=1).cpu().numpy()
                losses = criterion(outputs, labels, indices, epoch) * loss_scale
                loss = (weights * losses).sum()
                loss.backward()
                return losses, losses.mean().item() / loss_scale
                        
            closure.device = device
            
            if optimizer_config['type'].lower() == 'adversarial':
                optimizer.step(closure, indices)
                loss = optimizer.loss
            else:
                optimizer.zero_grad()
                outputs = model(images)
                if len(outputs) == 3:
                    outputs, _, _ = outputs
                
                loss_vector = criterion(outputs, labels, indices, epoch)
                loss = loss_vector.mean()
                loss.backward()
                optimizer.step()
                
                predictions[indices] = outputs.argmax(dim=1).cpu().numpy()
                losses_array[epoch, indices] = loss_vector.detach().cpu().numpy()
                loss = loss.item()
            
            epoch_loss_sum += loss * images.size(0)
        
        logging.info("Evaluating model...")
        preds_after, _, train_accuracy = evaluate(model, train_loader, device)
        train_predictions[epoch] = preds_after
        
        if optimizer_config['type'].lower() == 'adversarial':
            losses_array[epoch] = optimizer.pi.detach().cpu().numpy()
            predictions = preds_after
        avg_loss = epoch_loss_sum / train_size
        
        train_accuracies.append(train_accuracy)
        
        _, _, val_accuracy = evaluate(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        
        logging.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}, val acc: {val_accuracy:.4f}, train acc: {train_accuracy:.4f}")
        
        labels_history[epoch] = train_dataset.corrupted_labels
        correction_strategy.step(losses_array[epoch], predictions=predictions, reset_f=lambda *args, **kwargs: None)
    
        np.save(os.path.join(output_dir, "train_predictions.npy"), train_predictions)
        np.save(losses_path, losses_array)
        np.save(os.path.join(output_dir, "labels_history.npy"), labels_history)
        np.save(val_accuracies_path, val_accuracies)
        np.save(train_accuracies_path, train_accuracies)
        logging.info(f"Saved arrays")
        
        if scheduler is not None:
            scheduler.step()
            logging.debug(f"Scheduler step: {scheduler.get_last_lr()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on corrupted MNIST using AdamW")
    parser.add_argument("--config", type=str, default="default", help="Run config file")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (e.g., DEBUG, INFO)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--name", type=str, default=None, help="Name of the experiment")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer config to use")
    parser.add_argument("--model", type=str, default="resnet18", help="Model config to use")
    parser.add_argument("--dataset", type=str, default="cifar", help="Dataset config to use")
    parser.add_argument("--loss", type=str, default="ce", help="Loss function to use")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights file")
    parser.add_argument("--corruption", type=str, default=None, help="Corruption type")
    parser.add_argument("--noise", type=float, default=None, help="Noise level")
    args = parser.parse_args()

    args.log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    main(args)
