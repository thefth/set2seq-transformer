# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 23:20:34 2022

@author: theft
"""

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
    """Early stopping."""
    def __init__(self, accuracy=False, patience=10,
                 verbose=False, delta=0, path=None):
        """
        Args:
            patience (int): Number of epochs with no improvement until stopping.
                            Default: 10
            verbose (bool): Verbosity.
                            Default: False
            delta (float): Controls rate of improvement.
                            Default: 0
            path (str): Path for saving.
                            Default: 'checkpoint.pt'        
        """
        self.accuracy = accuracy
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path
        

    def __call__(self, epoch, current_score, model, optimizer, lr_scheduler):

        score = current_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, current_score, model, optimizer, lr_scheduler)
        elif (not self.accuracy and score >= self.best_score + self.delta) or\
        (self.accuracy and score <= self.best_score + self.delta) :
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch, current_score, model, optimizer, lr_scheduler)
            self.counter = 0

    def save_checkpoint(self, epoch, current_score, model, optimizer, lr_scheduler):
        '''Save checkpoint when there is improvement.'''
        if self.verbose:
            print(f'Validation scrore improved ({self.val_loss_min:.6f} --> {current_score:.6f}). Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = current_score
        
        if self.path:
            
            torch.save({
                'epoch': epoch,
                'current_score':current_score,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                }, self.path)
                

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
   
def set_gpu():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Training on: ',device)
    
    return device
   
 
def set_tensorboard(tensorboard_path):
    
    return SummaryWriter(tensorboard_path)


def print_model(model, dataloader,
                device=torch.device('cpu'), verbose=False):
    
    print(model)
    
    if verbose:
        
        summary(model.to(device), dataloader['train'].dataset[0][0].shape)   
    
    
def load_model(model, optimizer, exp_lr_scheduler, load_path='model.tar'):


    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    exp_lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
        
    return model, optimizer, exp_lr_scheduler, epoch, val_loss