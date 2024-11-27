# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn_(batch):
    """
    Generalized collate function for handling sequences with positions and temporal values.

    Args:
        batch (list): A list of tuples, where each tuple contains:
            - Tensor: Sequence data.
            - Tensor: Targets.
            - List: Metadata (e.g., names).
            - List: Positions.
            - List: Temporal values.

    Returns:
        tuple: Padded data, targets, metadata, positions, and temporal values.
    """
    data = pad_sequence([i[0] for i in batch], batch_first=True)
    labels = torch.tensor([i[1] for i in batch])
    metadata = torch.tensor([i[2] for i in batch])
    
    max_len = data.shape[1]

    positions = [i[-2] + [-1]*(max_len - len(i[-2])) for i in batch]
    temporal_values = [i[-1] + [-1]*(max_len - len(i[-1])) for i in batch]

    return data, labels, metadata, positions, temporal_values


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, list_, features, 
                 set_aggregate=None, temporality=True, 
                 model='Set2SeqTransformer', setting=1):
        """
        Unified dataset for handling embeddings, positional data, and temporal information.

        Args:
            list_ (list): Dataset annotations and metadata.
            features (dict): Precomputed features or embeddings.
            set_aggregate (callable, optional): Aggregation function for sets of features.
            temporality (bool): Whether to handle temporal data.
            model (str): Model that is used.
            setting (int): Seeting for SWDF.
        """
        self.list_ = list_
        self.features = features
        self.set_aggregate = set_aggregate
        self.temporality = temporality
        self.model = model
        self.setting = setting

    def __getitem__(self, index):
        # Fetch the data inputs, targets, and metadata
        data_inputs = self.list_[0][index]  # Dictionary
        target = self.list_[1][index]  # Target labels
        metadata = self.list_[2][index]  # metadata
        
        positions = [k for k, v in self.list_[0][index].items() for v_ in v]
                
        timestamps = [i for i in self.list_[-1][index].values()]

        n = len(data_inputs[0])
        
        # Split the `years` list into chunks of size `n`
        timestamps = [timestamps[i:i + n] for i in range(0, len(timestamps), n)]
        
        
        if self.temporality:
            if self.set_aggregate is None:
                # Extract raw feature embeddings
                if self.setting>1 and self.model!='Transformer':
                    # Handle grouped inputs: Keep shape [G, S, F] (groups, size per group, feature dim)
                    data = torch.stack([
                        torch.stack([torch.from_numpy(self.features[item.split('/')[-1]]) for item in sublist])
                        for sublist in data_inputs.values()
                    ])
                    
                    positions = [k for k, v in self.list_[0][index].items()]
                    
                else:
                    # Handle individual timesteps: Flatten to [T, F] (timesteps, feature dim)
                    data = torch.stack([
                        torch.from_numpy(self.features[item.split('/')[-1]])
                        for sublist in data_inputs.values() for item in sublist
                    ])
            else:
                # Apply aggregation function (e.g., mean or max)
                data = torch.stack([
                        torch.from_numpy(self.set_aggregate(list(map(self.features.get,
                                                                     [item.split('/')[-1] for item in sublist])), 
                                                            axis=0))
                        for sublist in data_inputs.values()
                    ])
                
        else:
            # Flatten non-temporal data and apply aggregation
            data = torch.from_numpy(self.set_aggregate([
                self.features[item.split('/')[-1]]
                for sublist in data_inputs.values() for item in sublist
            ], axis=0))
       
        # Return the processed data
        return data, target, metadata, positions, timestamps

    def __len__(self):
        return len(self.list_[0])