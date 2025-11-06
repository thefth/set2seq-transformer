"""
Unified dataloader for Set2Seq Transformer.

This module provides dataset classes and collate functions for:
- Mesogeos: Wildfire forecasting with temporal sensor data
- WikiArt-Seq2Rank: Artist career trajectory prediction with artwork sequences
"""
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from abc import ABC, abstractmethod


# =============================================================================
# Collate Functions
# =============================================================================

def collate_fn_mesogeos(batch):
    """
    Collate function for Mesogeos dataset with padding support.

    Args:
        batch (list): List of tuples (data, label, positions, temporal_values, metadata)

    Returns:
        tuple: (padded_data, labels, positions, temporal_values, metadata)
            - padded_data: [BatchSize, MaxSeqLen, Features]
            - labels: [BatchSize]
            - positions: List of lists with -1 padding
            - temporal_values: List of lists with -1 padding
            - metadata: [BatchSize]
    """
    data = pad_sequence([i[0] for i in batch], batch_first=True)
    labels = torch.tensor([i[1] for i in batch])
    
    max_len = data.shape[1]

    # Pad positions and temporal values with -1
    positions = [i[2] + [-1] * (max_len - len(i[2])) for i in batch]
    temporal_values = [i[3] + [-1] * (max_len - len(i[3])) for i in batch]
    
    metadata = torch.tensor([i[-1] for i in batch])

    return data, labels, positions, temporal_values, metadata


def collate_fn_wikiart_flattened(batch):
    """
    Collate function for WikiArt with flattened sequences (non-Set2Seq models).

    Args:
        batch (list): List of tuples (data, label, positions, years, metadata)

    Returns:
        tuple: (padded_data, labels, names, positions, years)
            - padded_data: [BatchSize, MaxSeqLen, Features]
            - labels: List of labels
            - positions: List of lists with -1 padding
            - years: List of lists with 0 padding
            - metadata: List of artist names
    """
    data = pad_sequence([i[0] for i in batch], batch_first=True)
    max_len = data.shape[1]
    
    positions = [i[2] + [-1] * (max_len - len(i[2])) for i in batch]
    years = [i[3] + [0] * (max_len - len(i[3])) for i in batch]
    
    metadata = [i[-1] for i in batch]
    
    return data, [i[1] for i in batch], positions, years, metadata


def collate_fn_wikiart_sets(batch):
    """
    Collate function for WikiArt with variable-length sets (Set2Seq models).

    Args:
        batch (list): List of tuples (data_dict, label, temporal_dict, metadata)

    Returns:
        tuple: (data, labels, positions, years, metadata)
            - data: List of lists of tensors (variable-length sets per timestep)
            - labels: List of labels
            - positions: Tensor of positions with -1 padding
            - years: Tensor of years with 0 padding
            - metadata: List of artist names
    """
    max_len = max([len(i[0].keys()) for i in batch])
    
    # Extract data as list of lists of tensors
    data = [
        [v for v in i[0].values()] + 
        [[torch.zeros(list(i[0].values())[0][0].shape[0])]] * (max_len - len(i[0]))
        for i in batch
    ]
    
    # Extract positions and years with padding
    positions = [
        torch.stack([torch.tensor(v) for v in i[2].keys()] + 
                   [torch.tensor(-1)] * (max_len - len(i[0])))
        for i in batch
    ]
    
    years = [
        torch.stack([torch.tensor(v) for v in i[2].values()] + 
                   [torch.tensor(0)] * (max_len - len(i[0])))
        for i in batch
    ]
    
    metadata = [i[-1] for i in batch]
    
    # Create attention mask only if there's actual padding
    seq_lengths = [len(i[0]) for i in batch]
    if all(length == max_len for length in seq_lengths):
        # No padding - no mask needed
        mask = None
    else:
        # Has padding - create mask
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for idx, length in enumerate(seq_lengths):
            mask[idx, :length] = True

    return data, [i[1] for i in batch], positions, years, metadata, mask


# =============================================================================
# Base Dataset Class
# =============================================================================

class BaseSequentialDataset(ABC, torch.utils.data.Dataset):
    """
    Abstract base class for sequential multiple-instance learning datasets.
    
    Subclasses should implement __getitem__ to return:
        (data, label, positions, temporal_values, metadata)
    """
    
    def __init__(self, data_list, features, temporality=True):
        """
        Args:
            data_list (list): List containing [X, y, temporal_values, metadata]
            features (dict): Dictionary mapping feature IDs to embeddings
            temporality (bool): Whether to include temporal information
        """
        self.data_list = data_list
        self.features = features
        self.temporality = temporality
    
    @abstractmethod
    def __getitem__(self, index):
        """Return (data, label, positions, temporal_values, metadata)"""
        pass
    
    def __len__(self):
        return len(self.data_list[0])


# =============================================================================
# Mesogeos Dataset
# =============================================================================

class MesogeosDataset(BaseSequentialDataset):
    """
    Dataset for Mesogeos wildfire forecasting.
    
    Handles temporal sequences of sensor readings with:
    - Variable-length sequences
    - Grouped timesteps (controlled by 'setting' parameter)
    - Temporal metadata (timestamps)
    
    Args:
        data_list (list): [X, y, temporal_values, metadata]
            - X: List of dicts mapping timestep -> list of feature IDs
            - y: List of labels (0 or 1 for fire/no-fire)
            - temporal_values: List of dicts mapping timestep -> timestamps
            - metadata: List of metadata (e.g., region IDs)
        features (dict): Feature embeddings dict
        set_aggregate (callable, optional): Aggregation function (np.mean, np.max)
        temporality (bool): Include temporal information
        model (str): Model type ('Set2SeqTransformer', 'Transformer', etc.)
        setting (int): Grouping factor for timesteps (1=no grouping, >1=group by N)
    """
    
    def __init__(self, data_list, features, set_aggregate=None, 
                 temporality=True, model='Set2SeqTransformer', setting=1):
        super().__init__(data_list, features, temporality)
        self.set_aggregate = set_aggregate
        self.model = model
        self.setting = setting

    def __getitem__(self, index):
        # Fetch data
        data_inputs = self.data_list[0][index]  # Dict: {timestep: [feature_ids]}
        target = self.data_list[1][index]
        metadata = self.data_list[-1][index]
        
        # Extract positions (timestep indices)
        positions = [k for k, v in data_inputs.items() for _ in v]
        
        # Extract timestamps
        timestamps = list(self.data_list[2][index].values())
        n = len(data_inputs[0])  # Items per timestep
        timestamps = [timestamps[i:i + n] for i in range(0, len(timestamps), n)]
        
        if self.temporality:
            if self.set_aggregate is None:
                # Add set dimension
                if self.setting == 1:
                    data = torch.stack([
                        torch.from_numpy(self.features[item.split('/')[-1]])
                        for sublist in data_inputs.values() 
                        for item in sublist
                    ])
                    data = data.unsqueeze(1)
                # Keep raw features per set
                elif self.setting > 1 and self.model != 'Transformer':
                    # Grouped: [NumGroups, SetSize, FeatureDim]
                    data = torch.stack([
                        torch.stack([
                            torch.from_numpy(self.features[item.split('/')[-1]]) 
                            for item in sublist
                        ])
                        for sublist in data_inputs.values()
                    ])
                    positions = list(data_inputs.keys())
                else:
                    # Flattened: [NumTimesteps, FeatureDim]
                    data = torch.stack([
                        torch.from_numpy(self.features[item.split('/')[-1]])
                        for sublist in data_inputs.values() 
                        for item in sublist
                    ])
            else:
                # Apply aggregation (mean/max) over sets
                data = torch.stack([
                    torch.from_numpy(
                        self.set_aggregate(
                            [self.features[item.split('/')[-1]] for item in sublist],
                            axis=0
                        )
                    )
                    for sublist in data_inputs.values()
                ])
        else:
            # Non-temporal: aggregate everything
            data = torch.from_numpy(
                self.set_aggregate([
                    self.features[item.split('/')[-1]]
                    for sublist in data_inputs.values() 
                    for item in sublist
                ], axis=0)
            )
       
        return data, target, positions, timestamps, metadata


# =============================================================================
# WikiArt Dataset
# =============================================================================

class WikiArtDataset(BaseSequentialDataset):
    """
    Dataset for WikiArt-Seq2Rank artist career prediction.
    
    Handles sequences of artworks with:
    - Variable-length artist careers
    - Temporal information (years)
    - Two modes: flattened (for Transformer/LSTM) or sets (for Set2SeqTransformer)
    
    Args:
        data_list (list): [X, y, names, temporal_values]
            - X: List of dicts mapping position -> list of artwork IDs
            - y: List of ranking scores
            - names: List of artist names
            - temporal_values: List of dicts mapping position -> years
        features (dict): Feature embeddings dict
        temporality (bool): Include temporal information
        use_sets (bool): If True, keep sets separate (for Set2SeqTransformer)
                        If False, flatten (for other models)
    """
    
    def __init__(self, data_list, features, temporality=True, use_sets=False):
        super().__init__(data_list, features, temporality)
        self.use_sets = use_sets

    def __getitem__(self, index):
        data_inputs = self.data_list[0][index]  # Dict: {position: [artwork_ids]}
        target = self.data_list[1][index]
        name = self.data_list[-1][index]
        
        positions = [k for k, v in data_inputs.items() for _ in v]
        years = [self.data_list[2][index][i] for i in 
                [k for k, v in data_inputs.items() for _ in v]]
        
        if self.use_sets:
            # Keep as dict of lists (for Set2SeqTransformer with variable set sizes)
            data = {
                k: [torch.from_numpy(self.features[item.split('/')[-1]]) for item in v]
                for k, v in data_inputs.items()
            }
            temporal_dict = self.data_list[2][index]
            return data, target, temporal_dict, name
        else:
            # Flatten all artworks (for Transformer/LSTM)
            data = torch.from_numpy(
                np.asarray([
                    self.features[item.split('/')[-1]]
                    for v in data_inputs.values() 
                    for item in v
                ])
            )
            return data, torch.from_numpy(np.array(target)), positions, years, name