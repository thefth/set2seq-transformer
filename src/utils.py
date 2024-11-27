# -*- coding: utf-8 -*-
import numpy as np
import os
import random
import torch

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class EarlyStopping:
    """
    Early stopping utility to halt training when performance stops improving.

    Args:
        accuracy (bool): Whether to monitor accuracy instead of loss.
        patience (int): Number of epochs to wait before stopping after no improvement.
        verbose (bool): If True, prints a message when saving the model.
        delta (float): Minimum improvement required to reset the patience counter.
        path (str): Path to save the best model checkpoint.
    """
    def __init__(self, accuracy=False, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        self.accuracy = accuracy  # True to maximize, False to minimize
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.mode = "accuracy" if accuracy else "loss"
        

    def __call__(self, epoch, current_score, model, optimizer, lr_scheduler=None):
        """
        Monitor performance and stop training if no improvement.
    
        Args:
            epoch (int): Current epoch number.
            current_score (float): Validation score (accuracy or loss).
            model (torch.nn.Module): The model to save if performance improves.
            optimizer (torch.optim.Optimizer): Optimizer for the model.
            lr_scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        """
        if self.best_score is None:
            # Initialize the best score and save the first checkpoint
            self.best_score = current_score
            self.save_checkpoint(epoch, current_score, model, optimizer, lr_scheduler)
            if self.verbose:
                print(f"Initial model saved with {self.mode} = {current_score:.6f}.")
        else:
            # Determine improvement direction (maximize accuracy or minimize loss)
            improvement = (current_score - self.best_score) if self.accuracy else (self.best_score - current_score)
            
            if improvement > self.delta:
                # Update best score and reset patience counter
                if self.verbose:
                    print(
                        f"{self.mode.capitalize()} improved from {self.best_score:.6f} to {current_score:.6f}."
                    )
                self.best_score = current_score
                self.save_checkpoint(epoch, current_score, model, optimizer, lr_scheduler)
                self.counter = 0
            else:
                # Increment patience counter if no improvement
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}.')
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, epoch, current_score, model, optimizer, lr_scheduler=None):
        """
        Save the current best model if performance improves.

        Args:
            epoch (int): Current epoch number.
            current_score (float): Validation score (accuracy or loss).
            model (torch.nn.Module): Model to save.
            optimizer (torch.optim.Optimizer): Optimizer state to save.
            lr_scheduler (torch.optim.lr_scheduler, optional): Scheduler state to save.
        """
        
        if self.path:
            
            if self.verbose:
                print(f'Saving model: {self.mode.capitalize()} improved to {current_score:.6f}.')
            
            checkpoint = {
                'epoch': epoch,
                'score': current_score,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
    
            if lr_scheduler is not None:
                checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            
            torch.save(checkpoint, self.path)
                

def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_gpu(device_index=0):
    """
    Set the GPU device to use for training.

    Args:
        device_index (int): Index of the GPU to use.

    Returns:
        torch.device: Device object pointing to the specified GPU.
    """
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
    print('Training on:', device)
    return device


def set_tensorboard(tensorboard_path):
    """
    Initialize a TensorBoard writer.

    Args:
        tensorboard_path (str): Path to store TensorBoard logs.

    Returns:
        SummaryWriter: TensorBoard writer instance.
    """
    if tensorboard_path is None:
        raise ValueError("Tensorboard path cannot be None.")
    return SummaryWriter(tensorboard_path)


def print_model(model, dataloader, device=torch.device('cpu'), verbose=False):
    """
    Print the model architecture and optionally its summary.

    Args:
        model (torch.nn.Module): Model to print.
        dataloader (dict): Dataloader with training data.
        device (torch.device): Device to place the model on.
        verbose (bool): If True, prints model summary.
    """
    print(model)
    if verbose:
        if len(dataloader['train']) == 0:
            raise ValueError("The dataloader is empty. Cannot print model summary.")
        summary(model.to(device), dataloader['train'].dataset[0][0].shape)


def load_model(model, optimizer, lr_scheduler=None, load_path='model.tar', device='cpu'):
    """
    Load a saved model checkpoint.

    Args:
        model (torch.nn.Module): Model to load.
        optimizer (torch.optim.Optimizer): Optimizer to load.
        lr_scheduler (torch.optim.lr_scheduler, optional): Scheduler to load. Default is None.
        load_path (str): Path to the saved checkpoint.
        device (torch.device): Device to load the model on.

    Returns:
        tuple: Loaded model, optimizer, scheduler, epoch, and validation loss.
    """
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint.get('lr_scheduler_state_dict', {}))
        
    epoch = checkpoint['epoch']
    score = checkpoint.get('score', None)
    return model, optimizer, lr_scheduler, epoch, score