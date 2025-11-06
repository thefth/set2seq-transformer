"""
Utility functions for simple baseline models.
Aggregation functions and dataset for Linear Regression / XGBoost baselines.
"""
import numpy as np
import torch


def aggregate_features(data, features, set_aggregate='mean', temporality=False):
    """
    Aggregate artwork features for each artist.
    
    Args:
        data: List of artist sequences where each sequence is a dict {year: [artwork_paths]}
        features: Dict mapping artwork IDs to feature vectors
        set_aggregate: Aggregation function ('mean' or 'max')
        temporality: If True, preserve temporal dimension (not used in baselines)
    
    Returns:
        numpy array of aggregated features [num_artists, feature_dim]
    """
    aggregated = []
    
    if set_aggregate == 'mean':
        agg_func = np.mean
    elif set_aggregate == 'max':
        agg_func = np.max
    else:
        raise ValueError(f"Unknown set_aggregate: {set_aggregate}")
    
    for artist_sequence in data:
        # artist_sequence is a dict: {year: [artwork_path1, artwork_path2, ...], ...}
        all_artworks = []
        
        # Iterate over all years
        for year, artwork_paths in artist_sequence.items():
            # artwork_paths is a list of paths
            for artwork_path in artwork_paths:
                # Extract artwork ID (filename without path)
                artwork_id = artwork_path.split('/')[-1]
                
                if artwork_id in features:
                    all_artworks.append(features[artwork_id])
        
        if len(all_artworks) > 0:
            # Stack and aggregate all artworks for this artist
            all_artworks = np.stack(all_artworks)  # [num_artworks, feature_dim]
            artist_features = agg_func(all_artworks, axis=0)  # [feature_dim]
            aggregated.append(artist_features)
        else:
            # Handle edge case: no features found
            feature_dim = len(next(iter(features.values())))
            aggregated.append(np.zeros(feature_dim))
    
    return np.array(aggregated)


class SimpleBaselineDataset(torch.utils.data.Dataset):
    """
    Dataset that returns pre-aggregated features.
    Compatible with the original baseline implementation.
    """
    def __init__(self, sequences, labels, names, features, set_aggregate='mean'):
        """
        Args:
            sequences: List of artist sequences
            labels: List of ranking labels
            names: List of artist names
            features: Dict of artwork features
            set_aggregate: 'mean' or 'max'
        """
        self.aggregated_features = aggregate_features(sequences, features, set_aggregate)
        self.labels = labels
        self.names = names
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.aggregated_features[idx],
            self.labels[idx],
            self.names[idx]
        )


def simple_collate_fn(batch):
    """Simple collate function for aggregated features."""
    features = torch.FloatTensor([item[0] for item in batch])
    labels = torch.FloatTensor([item[1] for item in batch])
    names = [item[2] for item in batch]
    return features, labels, names