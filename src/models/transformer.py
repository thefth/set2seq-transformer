# -*- coding: utf-8 -*-
import torch
import math



def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = torch.nn.functional.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = torch.nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.o_proj.weight)
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


class EncoderBlock(torch.nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        self.linear_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, dim_feedforward),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_feedforward, input_dim),
        )
        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.norm2 = torch.nn.LayerNorm(input_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        linear_out = self.linear_net(x)
        x = self.norm2(x + self.dropout(linear_out))
        return x


class TransformerEncoder(torch.nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = torch.nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        """
        General embedding class with optional CLS token support.
        
        Args:
            num_embeddings: Total number of embeddings (including padding).
            embedding_dim: Dimensionality of the embedding space.
            padding_idx: Index for padding token.
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, indices, include_cls_token=False):
        """
        Args:
            indices: Tensor of indices (e.g., positional or chronological values).
            include_cls_token: Whether to prepend a CLS token.
        """
        if include_cls_token:
            cls_token = torch.zeros(1, 1, device=indices.device, dtype=indices.dtype)
            indices = torch.cat([cls_token, indices + 1], dim=1)

        return self.embedding(indices)


class PositionalEncoding(torch.nn.Module):
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

    def forward(self, positions, include_cls_token=False):
        """
        Args:
            positions: Tensor of positions [BatchSize, SeqLen].
            include_cls_token: Whether to prepend a CLS token.
        """
        if include_cls_token:
            positions = torch.cat([torch.zeros(1, 1, device=positions.device), positions + 1], dim=1)

        if isinstance(positions, list):
            positions = torch.tensor(positions, dtype=torch.long)

        out = self.pe[:, positions]
        return out.squeeze(0) if out.size(0) == 1 else out


class TimestampTime2Vec(torch.nn.Module):
    def __init__(self, d_model=512):
        """
        Timestamp Time2Vec encoding for DD-MM-YYYY format.
        
        Args:
            d_model: Dimensionality of the encoding (must be divisible by 2).
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be divisible by 2 for month/year encoding."

        self.month_dim = d_model // 2
        self.year_dim = d_model // 2

        # Parameters for month encoding
        self.month_freq = torch.nn.Parameter(torch.ones(self.month_dim // 2) * (2 * torch.pi / 12))
        self.month_phase = torch.nn.Parameter(torch.zeros(self.month_dim // 2))

        self.year_linear_weight = torch.nn.Parameter(torch.randn(self.year_dim))  # Linear weight
        self.year_linear_bias = torch.nn.Parameter(torch.randn(self.year_dim))  # Linear bias
        
        
        # Parameters for year encoding
        self.year_ffn = torch.nn.Sequential(
            torch.nn.Linear(1, self.year_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.year_dim, self.year_dim),
        )

    def forward(self, timestamp, include_cls_token=False):
        """
        Args:
            timestamp: Tuple of tensors (day, month, year) with shape [BatchSize, SeqLen].
            include_cls_token: Whether to prepend a CLS token.
        """
        _, month, year = timestamp

        if include_cls_token:
            month = torch.cat([torch.zeros(1, 1, device=month.device), month], dim=1)
            year = torch.cat([torch.zeros(1, 1, device=year.device), year], dim=1)

        month = month * (2 * torch.pi / 12)
        
        # Normalize year (assuming range [min_year, max_year])
        min_year = 2006
        max_year = 2019
        
        year = (year - min_year) / (max_year - min_year)
        
        
        month_sin = torch.sin(self.month_freq * month.unsqueeze(-1) + self.month_phase)
        month_cos = torch.cos(self.month_freq * month.unsqueeze(-1) + self.month_phase)
        month_encoding = torch.cat([month_sin, month_cos], dim=-1)

        year_input = year.unsqueeze(-1)
        year_encoding = self.year_ffn(year_input)
        
        year_linear = year_encoding * self.year_linear_weight + self.year_linear_bias  # Shape: [Batch, SeqLen, linear_term_dim]

        timestamp_encoding = torch.cat([month_encoding, year_linear], dim=-1)
        
        return timestamp_encoding




class Transformer(torch.nn.Module):
    def __init__(
        self,
        input_dim=24,
        model_dim=512,
        num_classes=2,
        num_heads=32,
        num_layers=12,
        dropout=0.0,
        input_dropout=0.0,
        include_cls_token=False,
        positional_embedding_type="positional_encoding",
        temporal_embedding_type='timestamp_time2vec',
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
            temporal_embedding_type (str): Type of temporal embedding ('timestamp_time2vec', 'chronological_embedding', None).
        """
        super().__init__()
        self.input_net = torch.nn.Linear(input_dim, model_dim)

        # Positional Embedding
        self.positional_embedding = None
        if positional_embedding_type == "positional_encoding":
            self.positional_embedding = PositionalEncoding(d_model=model_dim)
        elif positional_embedding_type == "positional_embedding":
            self.positional_embedding = PositionalEmbedding(
                num_embeddings=32, embedding_dim=model_dim, padding_idx=31
            )

        # Temporal Embedding
        self.temporal_embedding = None
        if temporal_embedding_type == "timestamp_time2vec":
            self.temporal_embedding = TimestampTime2Vec(d_model=model_dim)
        elif temporal_embedding_type == "positional_embedding":
            self.temporal_embedding = PositionalEmbedding(
                num_embeddings=32, embedding_dim=model_dim, padding_idx=31
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
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.LayerNorm(model_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(model_dim, num_classes),
        )

    def forward(self, x, positions=None, temporal_values=None, mask=None):
        """
        Forward pass through the Transformer.

        Args:
            x (Tensor): Input features of shape [Batch, SeqLen, input_dim].
            positions (Tensor): Positional information for sequences.
            temporal_values (Tensor): Temporal information (e.g., years) for sequences.
            mask (Tensor): Optional attention mask of shape [Batch, SeqLen].

        Returns:
            Tensor: Output predictions of shape [Batch, num_classes].
        """
        x = self.input_net(x.float())  # Linear projection to model_dim

        # Add positional embeddings
        if self.positional_embedding is not None and positions is not None:
            x = x + self.positional_embedding(positions)

        # Add temporal embeddings
        if self.temporal_embedding is not None and temporal_values is not None:
            x = x + self.temporal_embedding(temporal_values)

        # Transformer Encoder
        x = self.transformer(x, mask)
        
        x = self.output_net(x)

        # Output layer
        return x[:,-int(30/len(set(positions[0]))):].mean(dim=1)

    @torch.no_grad()
    def get_attention_maps(self, x, positions=None, temporal_values=None, mask=None):
        """
        Extract attention maps from the Transformer.

        Args:
            x (Tensor): Input features of shape [Batch, SeqLen, input_dim].
            positions (Tensor): Positional information for sequences.
            temporal_values (Tensor): Temporal information (e.g., years) for sequences.
            mask (Tensor): Optional attention mask of shape [Batch, SeqLen].

        Returns:
            List[Tensor]: Attention maps for each Transformer layer.
        """
        x = self.input_net(x.float())

        if self.positional_embedding is not None and positions is not None:
            x = x + self.positional_embedding(positions)

        if self.temporal_embedding is not None and temporal_values is not None:
            x = x + self.temporal_embedding(temporal_values)

        return self.transformer.get_attention_maps(x, mask=mask)