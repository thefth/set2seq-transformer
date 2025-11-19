import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # Expand query to match the shape of keys
        query = query.unsqueeze(1).repeat(1, keys.size(1), 1)  # [batch_size, seq_len, hidden_size]
        
        # Compute attention scores
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys))).squeeze(-1)  # [batch_size, seq_len]
        
        # Compute attention weights
        weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]

        # Compute context vector
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # [batch_size, hidden_size]
        
        return context, weights


class LSTM(nn.Module):
    """
    A Bidirectional LSTM with Bahdanau Attention for sequence modeling.

    Args:
        input_dim (int): Dimensionality of the input features.
        hidden_dim (int): Dimensionality of the hidden state in the LSTM.
        num_layers (int): Number of layers in the LSTM.
        output_dim (int): Dimensionality of the output (e.g., number of classes for classification). Defaults to 2.
        device (str): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.

    Attributes:
        lstm (nn.LSTM): Bidirectional LSTM layer for sequence modeling.
        attention (BahdanauAttention): Attention mechanism to focus on relevant sequence parts.
        fc (nn.Linear): Fully connected layer for final prediction.

    Forward Input:
        x (torch.Tensor): Input tensor of shape `[BatchSize, SeqLen, InputDim]`.
        lengths (torch.Tensor, optional): Sequence lengths for each batch to handle variable-length inputs. Defaults to None.
        return_attention (bool): If True, return attention weights. Defaults to False.

    Forward Output:
        torch.Tensor: Output tensor of shape `[BatchSize, OutputDim]`.
        (Optional) torch.Tensor: Attention weights of shape `[BatchSize, SeqLen]` if return_attention=True.

    Methods:
        forward(x, lengths, return_attention): Computes the forward pass through the model.
        init_hidden(batch_size): Initializes the hidden and cell states for the LSTM.

    Example Usage:
        ```
        model = LSTM(input_dim=128, hidden_dim=64, num_layers=2, output_dim=10, device='cuda')
        x = torch.randn(32, 50, 128)  # Batch of 32 sequences, each of length 50, with 128 features
        lengths = torch.randint(10, 50, (32,))  # Variable sequence lengths
        
        # Without attention
        output = model(x, lengths)
        
        # With attention
        output, attn_weights = model(x, lengths, return_attention=True)
        ```
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=2, bidirectional=True, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.device = device

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                                  num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

        # Bahdanau Attention
        self.attention = BahdanauAttention(hidden_dim * 2)  # *2 for bidirectional

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths=None, return_attention=False):
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape `[BatchSize, SeqLen, InputDim]`.
            lengths (torch.Tensor, optional): Sequence lengths for variable-length inputs. Defaults to None.
            return_attention (bool): If True, return attention weights. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape `[BatchSize, OutputDim]`.
            (Optional) torch.Tensor: Attention weights of shape `[BatchSize, SeqLen]` if return_attention=True.
        """
        batch_size = x.size(0)

        # Initialize hidden and cell states
        hidden = self.init_hidden(batch_size)

        # Handle variable sequence lengths
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # LSTM forward pass with explicit hidden states
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)

        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Use the last hidden states for both directions
        last_hidden_forward = h_n[-2]
        last_hidden_backward = h_n[-1]
        last_hidden = torch.cat((last_hidden_forward, last_hidden_backward), dim=-1)

        # Apply attention
        context, attn_weights = self.attention(last_hidden, lstm_out)

        # Fully connected layer
        out = self.fc(context)

        if return_attention:
            return out, attn_weights
        return out

    def init_hidden(self, batch_size):
        """
        Initializes the hidden and cell states for the LSTM.

        Args:
            batch_size (int): Size of the batch.

        Returns:
            tuple: Initialized hidden state (h0) and cell state (c0), both of shape `[NumLayers*2, BatchSize, HiddenDim]`.
        """
        h0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_dim)).to(self.device)
        c0 = torch.zeros((self.num_layers * 2, batch_size, self.hidden_dim)).to(self.device)
        return h0, c0