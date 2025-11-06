import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ln0 = nn.LayerNorm(dim_V) if ln else None
        self.ln1 = nn.LayerNorm(dim_V) if ln else None

    def forward(self, Q, K):
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), dim=2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if self.ln0 is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        return O if self.ln1 is None else self.ln1(O)


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)




class SetTransformer_ISAB_PMA_SAB(nn.Module):
    """
    Set Transformer variant with ISAB + PMA + SAB for 3D input.

    Args:
        input_dim (int): Dimensionality of input features.
        num_outputs (int): Number of output sequences for PMA.
        output_dim (int): Dimensionality of output features.
        num_inds (int): Number of inducing points for ISAB.
        dim_hidden (int): Dimensionality of hidden layers.
        num_heads (int): Number of attention heads.
        num_classes (int): Number of output classes for final prediction. None for set-to-sequence.
        ln (bool): Whether to use layer normalization.
    """
    def __init__(self, input_dim=24, num_outputs=1, output_dim=512,
                 num_inds=1, dim_hidden=256, num_heads=4,
                 num_classes=None, ln=False):
        super().__init__()
        
        self.num_classes = num_classes

        # Encoder layers
        self.enc = nn.Sequential(
            ISAB(input_dim, dim_hidden, num_heads, num_inds, ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln),
        )

        # Decoder layers
        dec_layers = [
            PMA(dim_hidden, num_heads, num_outputs, ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln),
            nn.Linear(dim_hidden, output_dim),
        ]

        if self.num_classes:
            dec_layers += [nn.Linear(output_dim, num_classes)]

        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        if self.num_classes and x.ndim == 4: # Static SetTransformer
            x = x.squeeze(2)
            
        x = self.enc(x)
        x = self.dec(x)

        # Remove dimension of size 1 at position 1 if applicable
        if x.size(1) == 1:
            x = x.squeeze(1)

        return x


class SetTransformer4D_ISAB_PMA_SAB(SetTransformer_ISAB_PMA_SAB):
    """
    Set Transformer variant with ISAB + PMA + SAB for 4D input.

    Args:
        Same as SetTransformer_ISAB_PMA_SAB but designed to handle 4D inputs.

    Handles input of shape [Batch, Timesteps, Sets, Features].
    """
    def forward(self, x):
        batch_size, timesteps, num_sets, feature_dim = x.shape
        x = x.view(batch_size * timesteps, num_sets, feature_dim)
        x = self.enc(x)
        x = self.dec(x)
        return x.view(batch_size, timesteps, -1)


class SetTransformer_ISAB_PMA(nn.Module):
    """
    Set Transformer variant with ISAB + PMA for 3D input.

    Args:
        input_dim (int): Dimensionality of input features.
        num_outputs (int): Number of output sequences for PMA.
        output_dim (int): Dimensionality of output features.
        num_inds (int): Number of inducing points for ISAB.
        dim_hidden (int): Dimensionality of hidden layers.
        num_heads (int): Number of attention heads.
        num_classes (int): Number of output classes for final prediction. None for set-to-sequence.
        ln (bool): Whether to use layer normalization.
    """
    def __init__(self, input_dim=24, num_outputs=1, output_dim=512,
                 num_inds=1, dim_hidden=256, num_heads=4,
                 num_classes=None, ln=False):
        super().__init__()
        
        self.num_classes = num_classes

        # Encoder layers
        self.enc = nn.Sequential(
            ISAB(input_dim, dim_hidden, num_heads, num_inds, ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln),
        )

        # Decoder layers
        dec_layers = [
            PMA(dim_hidden, num_heads, num_outputs, ln),
            nn.Linear(dim_hidden, output_dim),
        ]

        if self.num_classes:
            dec_layers += [nn.Linear(output_dim, num_classes)]

        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        if self.num_classes and x.ndim == 4: # Static SetTransformer
            x = x.squeeze(2)
            
        x = self.enc(x)
        x = self.dec(x)

        # Remove dimension of size 1 at position 1 if applicable
        if x.size(1) == 1:
            x = x.squeeze(1)

        return x


class SetTransformer4D_ISAB_PMA(SetTransformer_ISAB_PMA):
    """
    Set Transformer variant with ISAB + PMA for 4D input.

    Args:
        Same as SetTransformer_ISAB_PMA but designed to handle 4D inputs.

    Handles input of shape [Batch, Timesteps, Sets, Features].
    """
    def forward(self, x):
        batch_size, timesteps, num_sets, feature_dim = x.shape
        x = x.view(batch_size * timesteps, num_sets, feature_dim)
        x = self.enc(x)
        x = self.dec(x)
        return x.view(batch_size, timesteps, -1)


class SetTransformer_SAB_PMA(nn.Module):
    """
    Set Transformer variant with SAB + PMA for 3D input.

    Args:
        input_dim (int): Dimensionality of input features.
        num_outputs (int): Number of output sequences for PMA.
        output_dim (int): Dimensionality of output features.
        dim_hidden (int): Dimensionality of hidden layers.
        num_heads (int): Number of attention heads.
        num_classes (int): Number of output classes for final prediction. None for set-to-sequence.
        ln (bool): Whether to use layer normalization.
    """
    def __init__(self, input_dim=24, num_outputs=1, output_dim=512,
                 dim_hidden=256, num_heads=4, num_classes=None, ln=False):
        super().__init__()
        
        self.num_classes = num_classes

        # Encoder layers
        self.enc = nn.Sequential(
            SAB(input_dim, dim_hidden, num_heads, ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln),
        )

        # Decoder layers
        dec_layers = [
            PMA(dim_hidden, num_heads, num_outputs, ln),
            nn.Linear(dim_hidden, output_dim),
        ]

        if self.num_classes:
            dec_layers += [nn.Linear(output_dim, num_classes)]

        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        if self.num_classes and x.ndim == 4: # Static SetTransformer
            x = x.squeeze(2)
            
        x = self.enc(x)
        x = self.dec(x)

        # Remove dimension of size 1 at position 1 if applicable
        if x.size(1) == 1:
            x = x.squeeze(1)
        
        return x


class SetTransformer4D_SAB_PMA(SetTransformer_SAB_PMA):
    """
    Set Transformer variant with SAB + PMA for 4D input.

    Args:
        Same as SetTransformer_SAB_PMA but designed to handle 4D inputs.

    Handles input of shape [Batch, Timesteps, Sets, Features].
    """
    def forward(self, x):
        batch_size, timesteps, num_sets, feature_dim = x.shape
        x = x.view(batch_size * timesteps, num_sets, feature_dim)
        x = self.enc(x)
        x = self.dec(x)
        return x.view(batch_size, timesteps, -1)

       


class HierarchicalSetTransformer(nn.Module):
    """
    Hierarchical Set Transformer for processing nested set-to-sequence data.

    This model combines two hierarchical levels of set-to-sequence transformers:
    - The first Set Tansformer (`set_base`) processes individual sets at each timestep.
    - The second Set Tansformer (`sequence_base`) processes the sequence of transformed sets.

    Args:
        set_base_input_dim (int): Dimensionality of input features for the set base Set Tansformer.
        seq_base_input_dim (int): Dimensionality of input features for the sequence base Set Tansformer.
        output_dim (int): Dimensionality of output features for the penultimate layer.
        dim_hidden (int): Dimensionality of hidden layers.
        num_heads (int): Number of attention heads for the transformers.
        num_inds (int): Number of inducing points for ISAB layers.
        num_classes (int): Number of output classes for classification.
        ln (bool): Whether to use layer normalization.

    Attributes:
        set_base (SetTransformer4D_ISAB_PMA): Set Tansformer for processing sets at each timestep.
        sequence_base (SetTransformer4D_ISAB_PMA): Set Tansformer for processing sequences of sets.
        fc (nn.Linear): Final fully connected layer for final prediction.

    Forward Input:
        x (torch.Tensor): Input tensor of shape `[BatchSize, Timesteps, SetSize, FeatureDim]`.

    Forward Output:
        torch.Tensor: Output tensor of shape `[BatchSize, num_classes]` for classification or `[BatchSize]` for regression.
    """
    def __init__(self, set_input_dim=24, seq_input_dim=512,
                 output_dim=512, dim_hidden=256, num_heads=4, 
                 num_inds=1, num_classes=2, ln=False):
        super(HierarchicalSetTransformer, self).__init__()
        
        self.set_base = SetTransformer4D_ISAB_PMA(input_dim=set_input_dim,
                                                output_dim=output_dim, 
                                                dim_hidden=dim_hidden,
                                                num_heads=num_heads, 
                                                num_inds=num_inds,
                                                ln=ln)
        
        self.sequence_base = SetTransformer4D_ISAB_PMA(input_dim=seq_input_dim,
                                                     output_dim=output_dim, 
                                                     dim_hidden=dim_hidden, 
                                                     num_heads=num_heads,
                                                     num_inds=num_inds, 
                                                     ln=ln)
        
        self.fc = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the HierarchicalSetTransformer.

        Args:
            x (torch.Tensor): Input tensor of shape `[BatchSize, Timesteps, SetSize, FeatureDim]`.

        Returns:
            torch.Tensor: Output tensor of shape `[BatchSize, num_classes]` for classification or `[BatchSize]` for regression.
        """
        # Process sets at each timestep
        x = self.set_base(x)
        
        # Expand dimensions if the input becomes 3D after set processing
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # Process the sequence of sets
        x = self.sequence_base(x)
       
        # Squeeze dimensions if the output remains 3D
        if len(x.shape) == 3:
            x = x.squeeze(-2)
            
        # Final output
        return self.fc(x)