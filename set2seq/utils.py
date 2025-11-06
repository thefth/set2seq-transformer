"""
Utility functions for training, checkpointing, and data preprocessing.
"""
import numpy as np
import os
import pickle
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# Training Utilities
# =============================================================================

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine learning rate scheduler with warmup."""
    
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
    """Early stopping utility to halt training when performance stops improving."""
    
    def __init__(self, accuracy=False, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        self.accuracy = accuracy
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.mode = "accuracy" if accuracy else "loss"

    def __call__(self, epoch, current_score, model, optimizer, lr_scheduler=None):
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(epoch, current_score, model, optimizer, lr_scheduler)
            if self.verbose:
                print(f"Initial model saved with score = {current_score:.6f}.")
        else:
            improvement = (current_score - self.best_score) if self.accuracy else (self.best_score - current_score)
            
            if improvement > self.delta:
                if self.verbose:
                    print(f"{self.mode.capitalize()} improved from {self.best_score:.6f} to {current_score:.6f}.")
                self.best_score = current_score
                self.save_checkpoint(epoch, current_score, model, optimizer, lr_scheduler)
                self.counter = 0
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}.')
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, epoch, current_score, model, optimizer, lr_scheduler=None):
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
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_model(model, optimizer, lr_scheduler=None, load_path='model.tar', device='cpu'):
    """Load a saved model checkpoint."""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint.get('lr_scheduler_state_dict', {}))
        
    epoch = checkpoint['epoch']
    score = checkpoint.get('score', None)
    return model, optimizer, lr_scheduler, epoch, score


# =============================================================================
# Data Loading Utilities
# =============================================================================

def group_elements_in_dict_list(dict_list, k):
    """
    Group dictionary values into chunks of size k.
    
    Args:
        dict_list (list): List of dictionaries with sequential keys
        k (int): Grouping factor
        
    Returns:
        list: List of dictionaries with grouped values
    """
    grouped_dicts = []
    for original_dict in dict_list:
        grouped_dict = {}
        values = list(original_dict.values())
        for i in range(0, len(values), k):
            grouped_dict[i // k] = [item for sublist in values[i:i + k] for item in sublist]
        grouped_dicts.append(grouped_dict)
    return grouped_dicts


def load_mesogeos_data(data_path, features_path, setting=1, set_aggregate=None, model='Set2SeqTransformer'):
    """
    Load and prepare Mesogeos dataset.
    
    Args:
        data_path (str): Path to dataset pickle
        features_path (str): Path to features pickle
        setting (int): Grouping factor
        set_aggregate (callable): Aggregation function
        model (str): Model name
        
    Returns:
        tuple: (train_data, val_data, test_data, features)
    """
    from dataloader import MesogeosDataset
    
    data = pickle.load(open(data_path, 'rb'))
    features = pickle.load(open(features_path, 'rb'))
    
    X_train_grouped = group_elements_in_dict_list(data['train']['x'], setting)
    X_val_grouped = group_elements_in_dict_list(data['val']['x'], setting)
    X_test_grouped = group_elements_in_dict_list(data['test']['x'], setting)
    
    train_data = [X_train_grouped, data['train']['y'],
                  data['train']['temporal_values'], data['train']['metadata']]
    val_data = [X_val_grouped, data['val']['y'],
                data['val']['temporal_values'], data['val']['metadata']]
    test_data = [X_test_grouped, data['test']['y'],
                 data['test']['temporal_values'], data['test']['metadata']]
    
    train_dataset = MesogeosDataset(train_data, features, set_aggregate, True, model, setting)
    val_dataset = MesogeosDataset(val_data, features, set_aggregate, True, model, setting)
    test_dataset = MesogeosDataset(test_data, features, set_aggregate, True, model, setting)
    
    # Calculate min_max years
    all_years = {int(date.split('-')[0]) 
                 for sample in data['train']['temporal_values']
                 for date in sample.values()}
    min_year = int(min(all_years))
    max_year = int(max(all_years))
    
    # Calculate max sequence length
    max_seq_len = int(max(len(x) for x in X_train_grouped))
    
    return train_dataset, val_dataset, test_dataset, min_year, max_year, max_seq_len


def load_wikiart_data(data_path, features_path, ranking='overall_ranking', split='stratified_split', model='Set2SeqTransformer'):
    """
    Load and prepare WikiArt dataset.
    
    Args:
        data_path (str): Path to dataset pickle
        features_path (str): Path to features pickle
        ranking (str): Ranking type to use
        split (str): Split type ('stratified_split' or 'time_series_split')
        model (str): Model name
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    from dataloader import WikiArtDataset
    
    data = pickle.load(open(data_path, 'rb'))
    
    # Select split
    data = data[split]
    
    # Remove empty sequences
    data = {k: v for k, v in data.items() if len(v['sequence']) > 0}
    
    # Calculate min_year
    all_years = {year for v in data.values() for year in v['year'].values()}
    min_year = int(min(all_years))
    max_year = int(max(all_years))
    
    # Calculate max sequence length
    max_seq_len = int(max(len(v['sequence']) for v in data.values()))

    features = pickle.load(open(features_path, 'rb'))
    
    # Fix feature keys if needed
    if '/' in list(features)[0]:
        features = {k.split('/')[-1]: v for k, v in features.items()}
    
    X_train = [v['sequence'] for v in data.values() if 'train' in v['rankings'][ranking]]
    y_train = [v['rankings'][ranking]['train'] for v in data.values() if 'train' in v['rankings'][ranking]]
    dates_train = [{k: year - min_year for k, year in v['year'].items()} 
               for v in data.values() if 'train' in v['rankings'][ranking]]
    names_train = [k for k, v in data.items() if 'train' in v['rankings'][ranking]]
    
    X_val = [v['sequence'] for v in data.values() if 'val' in v['rankings'][ranking]]
    y_val = [v['rankings'][ranking]['val'] for v in data.values() if 'val' in v['rankings'][ranking]]
    dates_val = [{k: year - min_year for k, year in v['year'].items()} 
               for v in data.values() if 'val' in v['rankings'][ranking]]
    names_val = [k for k, v in data.items() if 'val' in v['rankings'][ranking]]
    
    X_test = [v['sequence'] for v in data.values() if 'test' in v['rankings'][ranking]]
    y_test = [v['rankings'][ranking]['test'] for v in data.values() if 'test' in v['rankings'][ranking]]
    dates_test = [{k: year - min_year for k, year in v['year'].items()} 
               for v in data.values() if 'test' in v['rankings'][ranking]]
    names_test = [k for k, v in data.items() if 'test' in v['rankings'][ranking]]
    
    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()
    y_val = scaler.transform(np.array(y_val).reshape(-1, 1)).ravel()
    y_test = scaler.transform(np.array(y_test).reshape(-1, 1)).ravel()
    
    train_lists = [X_train, y_train, dates_train, names_train]
    val_lists = [X_val, y_val, dates_val, names_val]
    test_lists = [X_test, y_test, dates_test, names_test]
    
    use_sets = 'Set2Seq' in model
    train_dataset = WikiArtDataset(train_lists, features, True, use_sets)
    val_dataset = WikiArtDataset(val_lists, features, True, use_sets)
    test_dataset = WikiArtDataset(test_lists, features, True, use_sets)
    
    return train_dataset, val_dataset, test_dataset, min_year, max_year, max_seq_len