import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x).view(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.split(self.head_dim, dim=-1)
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.o_proj(values)

        return (output, attention) if return_attention else output


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attention=False):
        # Self-attention with optional attention return
        if return_attention:
            attn_out, attention = self.self_attn(x, mask=mask, return_attention=True)
            x = self.norm1(x + self.dropout(attn_out))
            linear_out = self.linear_net(x)
            x = self.norm2(x + self.dropout(linear_out))
            return x, attention
        else:
            attn_out = self.self_attn(x, mask=mask)
            x = self.norm1(x + self.dropout(attn_out))
            linear_out = self.linear_net(x)
            x = self.norm2(x + self.dropout(linear_out))
            return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None, return_attention=False):
        attention_maps = [] if return_attention else None
        
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, mask=mask, return_attention=True)
                attention_maps.append(attn)
            else:
                x = layer(x, mask=mask)
        
        return (x, attention_maps) if return_attention else x


class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        """
        General embedding class.
        
        Args:
            num_embeddings: Total number of embeddings (including padding).
            embedding_dim: Dimensionality of the embedding space.
            padding_idx: Index for padding token.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, indices):
        """
        Args:
            indices: Tensor of indices (e.g., positional or chronological values).
        """
        return self.embedding(indices)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        """
        Sinusoidal positional encoding.
        
        Args:
            d_model: Hidden dimensionality of the input.
            max_len: Maximum sequence length to encode.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, positions):
        """
        Args:
            positions: Tensor of positions [BatchSize, SeqLen].
        """
        if isinstance(positions, list):
            positions = torch.tensor(positions, dtype=torch.long)

        out = self.pe[:, positions]
        return out.squeeze(0) if out.size(0) == 1 else out


class TimestampTime2Vec(nn.Module):
    def __init__(self, d_model=512, min_year=2006, max_year=2019):
        """
        Timestamp Time2Vec encoding for DD-MM-YYYY format.
        
        Args:
            d_model: Dimensionality of the encoding (must be divisible by 2).
            min_year: Minimum year for normalization (default: 2006).
            max_year: Maximum year for normalization (default: 2019).
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be divisible by 2 for month/year encoding."

        self.month_dim = d_model // 2
        self.year_dim = d_model // 2
        self.min_year = min_year
        self.max_year = max_year

        # Parameters for month encoding
        self.month_freq = nn.Parameter(torch.ones(self.month_dim // 2) * (2 * torch.pi / 12))
        self.month_phase = nn.Parameter(torch.zeros(self.month_dim // 2))

        self.year_linear_weight = nn.Parameter(torch.randn(self.year_dim))  # Linear weight
        self.year_linear_bias = nn.Parameter(torch.randn(self.year_dim))  # Linear bias
        
        # Parameters for year encoding
        self.year_ffn = nn.Sequential(
            nn.Linear(1, self.year_dim),
            nn.ReLU(),
            nn.Linear(self.year_dim, self.year_dim),
        )

    def forward(self, timestamp):
        """
        Args:
            timestamp: Tuple of tensors (day, month, year) with shape [BatchSize, SeqLen].
        """
        _, month, year = timestamp

        month = month * (2 * torch.pi / 12)
        year = (year - self.min_year) / (self.max_year - self.min_year)
        
        month_sin = torch.sin(self.month_freq * month.unsqueeze(-1) + self.month_phase)
        month_cos = torch.cos(self.month_freq * month.unsqueeze(-1) + self.month_phase)
        month_encoding = torch.cat([month_sin, month_cos], dim=-1)

        year_input = year.unsqueeze(-1)
        year_encoding = self.year_ffn(year_input)
        
        year_linear = year_encoding * self.year_linear_weight + self.year_linear_bias  # Shape: [Batch, SeqLen, linear_term_dim]

        timestamp_encoding = torch.cat([month_encoding, year_linear], dim=-1)
        
        return timestamp_encoding


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim=24,
        model_dim=512,
        num_classes=1,
        num_heads=8,
        num_layers=6,
        dropout=0.0,
        input_dropout=0.0,
        positional_embedding_type="positional_encoding",
        temporal_embedding_type='positional_embedding',
        positional_embedding_dim=30,
        temporal_embedding_dim=32, 
        min_year=2006, 
        max_year=2019,
        pooling_method="mean"
    ):
        """
        Transformer Model with configurable positional and temporal embeddings.

        Args:
            input_dim (int): Dimensionality of input features.
            model_dim (int): Dimensionality of the model's hidden layers.
            num_classes (int): Number of output classes.
            num_heads (int): Number of attention heads in the multi-head attention layers.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate applied to the model.
            positional_embedding_type (str): Type of positional embedding ('positional_encoding', 'positional_embedding', None).
            temporal_embedding_type (str): Type of temporal embedding ('timestamp_time2vec', 'positional_embedding', None).
            pooling_method (str): Pooling method ('mean', 'last', 'cls').
        """
        super().__init__()
        self.pooling_method = pooling_method
        self.input_net = nn.Linear(input_dim, model_dim)
        
        # Learnable CLS token for cls pooling
        if pooling_method == 'cls':
            self.cls_token = torch.nn.Parameter(torch.randn(1, 1, model_dim))
        else:
            self.cls_token = None

        # Positional Embedding
        self.positional_embedding = None
        if positional_embedding_type == "positional_encoding":
            self.positional_embedding = PositionalEncoding(d_model=model_dim, max_len=positional_embedding_dim)
        elif positional_embedding_type == "positional_embedding":
            self.positional_embedding = PositionalEmbedding(
                num_embeddings=temporal_embedding_dim, embedding_dim=model_dim, padding_idx=temporal_embedding_dim-1
            )

        # Temporal Embedding
        self.temporal_embedding = None
        if temporal_embedding_type == "timestamp_time2vec":
            self.temporal_embedding = TimestampTime2Vec(d_model=model_dim, min_year=min_year, max_year=max_year)
        elif temporal_embedding_type == "positional_embedding":
            self.temporal_embedding = PositionalEmbedding(
                num_embeddings=temporal_embedding_dim, embedding_dim=model_dim, padding_idx=temporal_embedding_dim-1
            )

        # Transformer Encoder
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            input_dim=model_dim,
            dim_feedforward=2 * model_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Output layer
        self.output_net = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes),
        )

    def forward(self, x, positions=None, temporal_values=None, mask=None, return_attention=False):
        """
        Forward pass through the Transformer.

        Args:
            x (Tensor): Input features of shape [Batch, SeqLen, input_dim].
            positions (Tensor): Positional information for sequences.
            temporal_values (Tensor): Temporal information (e.g., years) for sequences.
            mask (Tensor): Optional attention mask of shape [Batch, SeqLen].
            return_attention (bool): If True, return attention weights from all layers.

        Returns:
            Tensor: Output predictions of shape [Batch, num_classes].
            (Optional) List[Tensor]: Attention weights if return_attention=True.
        """
        x = self.input_net(x.float())

        # Add positional embeddings
        if self.positional_embedding is not None and positions is not None:
            x = x + self.positional_embedding(positions)

        # Add temporal embeddings
        if self.temporal_embedding is not None and temporal_values is not None:
            x = x + self.temporal_embedding(temporal_values)

        # Add CLS token if using cls pooling
        if self.pooling_method == 'cls':
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            if mask is not None:
                cls_mask = torch.ones(batch_size, 1, 1, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=-1)
                
        # Transformer Encoder with optional attention return
        if return_attention:
            x, attention_maps = self.transformer(x, mask, return_attention=True)
        else:
            x = self.transformer(x, mask)
        
        x = self.output_net(x)
        
        # Pooling based on method
        if self.pooling_method == 'cls':
            output = x[:, 0]
        elif self.pooling_method == 'last':
            if mask is not None:
                mask_squeezed = mask.squeeze(1).squeeze(1)
                seq_lengths = mask_squeezed.sum(dim=1) - 1
                batch_indices = torch.arange(x.size(0), device=x.device)
                output = x[batch_indices, seq_lengths.long()]
            else:
                output = x[:, -1]
        elif self.pooling_method == 'mean':
            if mask is not None:
                mask_squeezed = mask.squeeze(1).squeeze(1)
                mask_expanded = mask_squeezed.unsqueeze(-1).expand_as(x)
                x_masked = x * mask_expanded.float()
                x_sum = x_masked.sum(dim=1)
                seq_lengths = mask_squeezed.sum(dim=1, keepdim=True)
                output = x_sum / seq_lengths.float()
            else:
                output = x.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling_method: {self.pooling_method}")
        
        return (output, attention_maps) if return_attention else output