import torch
import torch.nn as nn
from . import deep_sets
from . import set_transformer
from . import LSTM
from .transformer import Transformer


class Set2SeqTransformer(nn.Module):
    """
    Set2Seq Transformer combines a set-processing base (e.g., DeepSet or SetTransformer)
    with a sequence-processing Transformer-based model.
    """
    def __init__(self,
                 set_model_name='DeepSets',
                 set_input_dim=24,
                 set_dim_hidden=256,
                 set_output_dim=512,
                 set_num_heads=4,
                 set_num_inds=1,
                 set_ln=False,
                 sequence_model_name='Transformer',
                 sequence_input_dim=512,
                 sequence_model_dim=512,
                 sequence_num_classes=1,
                 sequence_num_heads=8,
                 sequence_num_layers=6,
                 sequence_dropout=0.,
                 sequence_input_dropout=0.,
                 positional_embedding_type='positional_encoding',
                 temporal_embedding_type='positional_embedding',
                 positional_embedding_dim=77,
                 temporal_embedding_dim=605, 
                 min_year=2006,              
                 max_year=2019,              
                 get_last_temporal_embedding=False,
                 variable_set_size=True,
                 pooling_method="mean"):
        """
        Args:
            set_model: The set-processing architecture to use (e.g., DeepSet, SetTransformer).
            sequence_input_dim: Input dimensionality for the sequence-processing Transformer.
            sequence_model_dim: Model dimensionality for the sequence-processing Transformer.
            positional_embedding_type: Type of positional embedding ('positional_encoding', 'positional_embedding', None).
            temporal_embedding_type: Type of temporal embedding ('timestamp_time2vec', 'positional_embedding', None).
        """
        super().__init__()
        
        # Configuration
        self.set_model_name = set_model_name
        self.sequence_model_name = sequence_model_name
        self.positional_embedding_type = positional_embedding_type
        self.temporal_embedding_type = temporal_embedding_type
        self.get_last_temporal_embedding = get_last_temporal_embedding
        self.variable_set_size = variable_set_size
        self.pooling_method = pooling_method

        # Set-based processing module
        if self.set_model_name == 'DeepSets':
            self.set_model = deep_sets.DeepSets(input_dim=set_input_dim,
                                               layer_norm=True)
        elif self.set_model_name == 'SetTransformer_SAB_PMA':
            self.set_model = set_transformer.SetTransformer4D_SAB_PMA(
                input_dim=set_input_dim,
                output_dim=set_output_dim,
                num_heads=set_num_heads,
                num_inds=set_num_inds,
                dim_hidden=set_dim_hidden,
                ln=set_ln
            )
        elif self.set_model_name == 'SetTransformer_ISAB_PMA':
            self.set_model = set_transformer.SetTransformer4D_ISAB_PMA(
                input_dim=set_input_dim,
                output_dim=set_output_dim,
                num_heads=set_num_heads,
                num_inds=set_num_inds,
                dim_hidden=set_dim_hidden,
                ln=set_ln
            )
        elif self.set_model_name == 'SetTransformer_ISAB_PMA_SAB':
            self.set_model = set_transformer.SetTransformer4D_ISAB_PMA_SAB(
                input_dim=set_input_dim,
                output_dim=set_output_dim,
                num_heads=set_num_heads,
                num_inds=set_num_inds,
                dim_hidden=set_dim_hidden,
                ln=set_ln
            )
        else:
            raise ValueError(f"Invalid set_model: {self.set_model_name}")

        # Sequence-based processing module
        if self.sequence_model_name == 'Transformer':
            self.sequence_model = Transformer(
                input_dim=sequence_input_dim,
                model_dim=sequence_model_dim,
                num_classes=sequence_num_classes,
                num_heads=sequence_num_heads,
                num_layers=sequence_num_layers,
                dropout=sequence_dropout,
                input_dropout=sequence_input_dropout,
                positional_embedding_type=positional_embedding_type,
                temporal_embedding_type=temporal_embedding_type,
                positional_embedding_dim=positional_embedding_dim,
                temporal_embedding_dim=temporal_embedding_dim,
                min_year=min_year,
                max_year=max_year,
                pooling_method=pooling_method
            )
        elif self.sequence_model_name == 'LSTM':
            self.sequence_model = LSTM(sequence_input_dim,
                                               sequence_model_dim,
                                               num_layers=2,
                                               output_dim=sequence_num_classes)
        else:
             raise ValueError(f"Invalid sequence_model: {sequence_model_name}")

    def forward(self, x, positions=None, temporal_values=None, mask=None, return_attention=False):
        """
        Forward pass through the Set2Seq Transformer.
    
        Args:
            x: Input features of shape [Batch, SeqLen, NumSets, FeatureDim] or list of lists.
            positions: Positional information for sequences.
            temporal_values: Temporal information for sequences (e.g., years).
            mask: Optional mask for the sequence model.
            return_attention: If True, return attention weights from both set and sequence models.
    
        Returns:
            torch.Tensor: Sequence-level predictions.
            (Optional) dict: Attention weights if return_attention=True
                - 'set_level': List of attention dicts (one per timestep) if SetTransformer
                - 'sequence_level': List of attention tensors
        """
        attention_maps = {'set_level': [], 'sequence_level': None} if return_attention else None
        
        # Process set-based features
        if not self.variable_set_size:
            # Fixed-size sets: [Batch, SeqLen, NumSets, FeatureDim]
            if return_attention and 'SetTransformer' in self.set_model_name:
                x, set_attn = self.set_model(x, return_attention=True)
                attention_maps['set_level'] = set_attn
            else:
                x = self.set_model(x)
        else:
            # Variable-size sets: list of lists
            output_sequence = []
            for sample in x:
                output_sample = []
                for id_, set_ in enumerate(sample):
                    set_tensor = torch.stack(set_).float().unsqueeze(0)
                    
                    if return_attention and 'SetTransformer' in self.set_model_name:
                        set_out, set_attn = self.set_model(set_tensor, return_attention=True)
                        output_sample.append(set_out)
                        attention_maps['set_level'].append(set_attn)
                    else:
                        output_sample.append(self.set_model(set_tensor))
                        
                output_sequence.append(torch.cat(output_sample))
            x = torch.stack(output_sequence)

        # Sequence-based processing
        if self.sequence_model_name == 'Transformer':
            # Process input features using the Transformer sequence base
            x = self.sequence_model.input_net(x.float())
    
            # Add positional embeddings
            if self.sequence_model.positional_embedding is not None and positions is not None:
                positional_embedding = self.sequence_model.positional_embedding(positions)
                x = x + positional_embedding
            
            # Add temporal embeddings
            if self.sequence_model.temporal_embedding is not None and temporal_values is not None:
                temporal_embedding = self.sequence_model.temporal_embedding(temporal_values)
                
                if len(temporal_embedding.shape) > 3:
                    if not self.get_last_temporal_embedding:
                        temporal_embedding = temporal_embedding.mean(2)
                    else:
                        temporal_embedding = temporal_embedding[:, :, -1]
                        
                x = x + temporal_embedding

            # Add CLS token if using cls pooling
            if self.pooling_method == 'cls':
                batch_size = x.size(0)
                cls_tokens = self.sequence_model.cls_token.expand(batch_size, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
                
                if mask is not None:
                    cls_mask = torch.ones(batch_size, 1, 1, 1, dtype=torch.bool, device=mask.device)
                    mask = torch.cat([cls_mask, mask], dim=-1)
                    
            # Pass through Transformer layers with optional attention extraction
            if return_attention:
                x, seq_attn = self.sequence_model.transformer(x, mask=mask, return_attention=True)
                attention_maps['sequence_level'] = seq_attn
            else:
                x = self.sequence_model.transformer(x, mask=mask)
            
            # Output predictions
            x = self.sequence_model.output_net(x)
            
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
                
        elif self.sequence_model_name == 'LSTM':
            # Extract sequence lengths from mask
            lengths = None
            if mask is not None:
                mask_squeezed = mask.squeeze(1).squeeze(1)
                lengths = mask_squeezed.sum(dim=1).cpu()
            
            # LSTM with optional attention extraction
            if return_attention:
                output, lstm_attn = self.sequence_model(x, lengths, return_attention=True)
                attention_maps['sequence_level'] = {'lstm_attention': lstm_attn}
            else:
                output = self.sequence_model(x, lengths)
        else:
            raise ValueError(f"Unsupported sequence base model: {self.sequence_model_name}")
        
        return (output, attention_maps) if return_attention else output