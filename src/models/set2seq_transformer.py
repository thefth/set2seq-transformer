# -*- coding: utf-8 -*-
import torch
from . import deep_sets
from . import set_transformer
from . import LSTM
from .transformer import Transformer


class Set2SeqTransformer(torch.nn.Module):
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
                 sequence_num_classes=2,
                 sequence_num_heads=16,
                 sequence_num_layers=12,
                 sequence_dropout=0.,
                 sequence_input_dropout=0.,
                 include_cls_token=False,
                 positional_embedding_type='positional_encoding',
                 temporal_embedding_type='timestamp_time2vec',
                 get_last_temporal_embedding=False):
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
        self.include_cls_token = include_cls_token
        self.get_last_temporal_embedding = get_last_temporal_embedding
        
        self.sequence_model_name = sequence_model_name

        # Set-based processing module
        if set_model_name == 'DeepSets':
            self.set_model = deep_sets.DeepSets(input_dim=set_input_dim,
                                               layer_norm=True)
        elif set_model_name == 'SetTransformer_SAB_PMA':
            self.set_model = set_transformer.SetTransformer4D_SAB_PMA(
                input_dim=set_input_dim,
                output_dim=set_output_dim,
                num_heads=set_num_heads,
                num_inds=set_num_inds,
                dim_hidden=set_dim_hidden,
                ln=set_ln,
                set2seq=True
            )
        elif set_model_name == 'SetTransformer_ISAB_PMA':
            self.set_model = set_transformer.SetTransformer4D_ISAB_PMA(
                input_dim=set_input_dim,
                output_dim=set_output_dim,
                num_heads=set_num_heads,
                num_inds=set_num_inds,
                dim_hidden=set_dim_hidden,
                ln=set_ln,
                set2seq=True
            )
        elif set_model_name == 'SetTransformer_ISAB_PMA_SAB':
            self.set_model = set_transformer.SetTransformer4D_ISAB_PMA_SAB(
                input_dim=set_input_dim,
                output_dim=set_output_dim,
                num_heads=set_num_heads,
                num_inds=set_num_inds,
                dim_hidden=set_dim_hidden,
                ln=set_ln,
                set2seq=True
            )
        else:
            raise ValueError(f"Invalid set_model: {set_model_name}")

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
                include_cls_token=include_cls_token,
                positional_embedding_type=positional_embedding_type,
                temporal_embedding_type=temporal_embedding_type,
            )
        elif self.sequence_model_name == 'LSTM':
            self.sequence_model = LSTM(sequence_input_dim,
                                               sequence_model_dim,
                                               num_layers=2,
                                               output_dim=sequence_num_classes)
        else:
             raise ValueError(f"Invalid sequence_model: {sequence_model_name}")

       

    def forward(self, x, positions=None, temporal_values=None, mask=None):
        """
        Forward pass through the Set2Seq Transformer.
    
        Args:
            x: Input features of shape [Batch, SeqLen, NumSets, FeatureDim].
            positions: Positional information for sequences.
            temporal_values: Temporal information for sequences (e.g., years).
            mask: Optional mask for the sequence model.
    
        Returns:
            torch.Tensor: Sequence-level predictions.
        """
        # Process set-based features
        x = self.set_model(x)  # Output shape: [Batch, SeqLen, ModelDim]
    
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
                
                if len(temporal_embedding.shape)>3:
                        
                    if not self.get_last_temporal_embedding:
                        temporal_embedding = temporal_embedding.mean(2)
                        
                    else:
                        temporal_embedding = temporal_embedding[:, :, -1]
                        
                x = x + temporal_embedding
    
            # Pass through Transformer layers
            x = self.sequence_model.transformer(x, mask=mask)
    
            # Output predictions
            x = self.sequence_model.output_net(x)
            return x[:, -1]  # Return the last timestep (or modify as needed)
    
        elif self.sequence_model_name == 'LSTM':
            # Directly pass the sequence to LSTM
            x = self.sequence_model(x)
            return x  # LSTM already outputs the sequence-level predictions
        else:
            raise ValueError(f"Unsupported sequence base model: {self.sequence_model}")