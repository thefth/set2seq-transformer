# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class DeepSets(nn.Module):
    def __init__(self, input_dim=24, dim_hidden=256, output_dim=512, 
                 pool="max", layer_norm=False, num_classes=None):
        """
        A DeepSet implementation that can adapt to different configurations.

        Args:
            input_dim (int): Input feature dimensionality.
            dim_hidden (int): Hidden layer dimensionality.
            output_dim (int): Output dimensionality.
            pool (str): Pooling method, one of {"max", "mean", "sum"}.
            layer_norm (bool): Whether to use layer normalization in the decoder.
            num_classes (int, optional): Final output dimensionality. If None, no final layer.
        """
        super().__init__()
        
        self.layer_norm = layer_norm
        self.num_classes = num_classes
        
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )

        # Decoder layers
        dec_layers = [
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, output_dim)
        ]
        
        if self.layer_norm:
            dec_layers.append(nn.LayerNorm(output_dim))
        
        if self.num_classes:
            dec_layers += [nn.ReLU(),
                nn.Linear(512, self.num_classes)]
        

        self.dec = nn.Sequential(*dec_layers)

        # Validate pooling method
        if pool not in {"max", "mean", "sum"}:
            raise ValueError(f"Invalid pooling method: {pool}. Choose from 'max', 'mean', or 'sum'.")
        self.pool = pool

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Timesteps, Features].

        Returns:
            torch.Tensor: Output tensor after encoding, pooling, and decoding.
        """
        if self.num_classes and x.ndim == 4: # Static DeepSets
            x = x.squeeze(2)

        x = self.enc(x.float())

        # Apply pooling
        if self.pool == "max":
            x = x.max(dim=-2)[0]
        elif self.pool == "mean":
            x = x.mean(dim=-2)
        elif self.pool == "sum":
            x = x.sum(dim=-2)

        # Pass through the decoder
        x = self.dec(x)
        return x



class HierarchicalDeepSets(nn.Module):
    """
    A hierarchical DeepSets architecture that processes data in two stages:
    first over sets of features and then over sequences of aggregated sets.

    Args:
        set_input_dim (int): Input feature dimensionality for the set-level DeepSets.
        seq_input_dim (int): Input feature dimensionality for the sequence-level DeepSets.
        dim_hidden (int): Hidden layer dimensionality for both DeepSets.
        num_classes (int): Output dimensionality. For regression tasks, this should be 1.
        pool (str): Pooling method for both DeepSets, one of {"max", "mean", "sum"}.

    Attributes:
        set_base (DeepSets): The first DeepSets module for processing sets.
        sequence_base (DeepSets): The second DeepSets module for processing aggregated sets over time.
        fc (nn.Linear): Final fully connected layer for predictions.

    Methods:
        forward(x):
            Perform the forward pass through the hierarchical DeepSets model.
            Args:
                x (torch.Tensor): Input tensor of shape [Batch, Timesteps, Features].
            Returns:
                torch.Tensor: Output tensor of shape [Batch, OutputDim].
    """
    def __init__(self, set_input_dim=24, seq_input_dim=512,
                 dim_hidden=256, num_classes=2, pool="max"):
        super().__init__()
        self.set_base = DeepSets(input_dim=set_input_dim, dim_hidden=dim_hidden,
                                 pool=pool)
        self.sequence_base = DeepSets(input_dim=seq_input_dim, dim_hidden=dim_hidden, 
                                      pool=pool)
        self.fc = nn.Linear(dim_hidden * 2, num_classes)

    def forward(self, x):
        """
        Forward pass through the hierarchical DeepSets model.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Timesteps, Features].
        
        Returns:
            torch.Tensor: Output tensor of shape [Batch, OutputDim].
        """
        x = self.set_base(x)
        x = self.sequence_base(x)
        x = self.fc(x)
        return x