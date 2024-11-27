# -*- coding: utf-8 -*-
"""
Main script for training and evaluating models.
"""

import os
import numpy as np
import pickle
import argparse
import torch
import helpers
import utils
from dataloader import CustomDataset, collate_fn_


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Set2Seq Transformer models.")
    
    # Dataset and model configuration
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset file.")
    parser.add_argument('--features', type=str, required=True, help="Path to the features file.")
    parser.add_argument('--model', type=str, default='Set2SeqTransformer', help="Model to use.")
    parser.add_argument('--input_dim', type=int, default=24, help="Input dimensionality.")
    parser.add_argument('--set_model_name', type=str, default='DeepSets', help="Set base model.")
    parser.add_argument('--sequence_model_name', type=str, default='Transformer', help="Sequence base model.")
    parser.add_argument('--positional_embedding', type=str, default='positional_encoding', help="Positional embedding type.")
    parser.add_argument('--temporal_embedding', type=str, default='timestamp_time2vec', help="Temporal embedding type.")
    parser.add_argument('--set_aggregate', type=str, choices=['mean', 'max'], default=None,
                        help="Aggregation function to use (options: 'mean', 'max').")
    parser.add_argument('--temporality', action='store_true', default=True, help="Enable temporal dimension.")
    parser.add_argument('--setting', type=int, default=1, help="Setting for SWDF.")
    parser.add_argument('--use_timestamp', action='store_true', default=True, help="Use full timestamp.")
    
    
    # Training configuration
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs.")
    parser.add_argument('--early_stopping_patience', type=int, default=20, help="Early stopping patience.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use: 'cuda' or 'cpu'.")
    
    # Save paths
    parser.add_argument('--save', action='store_true', default=False, help="Whether to save the best model.")
    parser.add_argument('--save_path', type=str, default='.', help="Directory to save the model.")
    
    
    
    args = parser.parse_args()
    
    # Map the string input to actual numpy functions
    if args.set_aggregate == 'mean':
        args.set_aggregate = np.mean
    elif args.set_aggregate == 'max':
        args.set_aggregate = np.max
        
    # Generate save_path if saving is enabled
    if args.save:
        args.save_path = os.path.join(args.save_path, f"{args.model}_setting{args.setting}.pt")
    else:
        args.save_path = None
    
    return args

def load_data(args):
    print(f"Loading dataset from {args.data_path}...")
    data = pickle.load(open(args.data_path, 'rb'))
    features = pickle.load(open(args.features, 'rb'))
    
    
    def group_elements_in_dict_list(dict_list, k):
        grouped_dicts = []
        for original_dict in dict_list:
            grouped_dict = {}
            values = list(original_dict.values())
            for i in range(0, len(values), k):
                grouped_dict[i // k] = [item for sublist in values[i:i + k] for item in sublist]
            grouped_dicts.append(grouped_dict)
        return grouped_dicts
    
    # Preprocess and group elements
    setting = args.setting  # Define grouping setting
    X_train_grouped = group_elements_in_dict_list(data['train']['x'], setting)
    X_val_grouped = group_elements_in_dict_list(data['val']['x'], setting)
    X_test_grouped = group_elements_in_dict_list(data['test']['x'], setting)
    

    train_data = [X_train_grouped, data['train']['y'],\
                 data['train']['metadata'], data['train']['temporal_values']]
                         
    
    
                                
    val_data = [X_val_grouped, data['val']['y'],\
               data['val']['metadata'], data['val']['temporal_values']]
    
    
    test_data = [X_test_grouped, data['test']['y'],\
               data['test']['metadata'], data['test']['temporal_values']]
    
    return train_data, val_data, test_data, features



if __name__ == "__main__":
    
    args = parse_args()
    
    # Set device and seed
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    utils.set_seed(args.seed)
    print(f"Using device: {device}")
    
    # Load data
    train_data, val_data, test_data, features = load_data(args)
    print(f"Using setting: {args.setting}")
    
    # Correctly pass the arguments to CustomDataset
    train_dataset = CustomDataset(
        list_=train_data,
        features=features,
        set_aggregate=args.set_aggregate,
        temporality=args.temporality,
        model=args.model,
        setting=args.setting
    )
    
    val_dataset = CustomDataset(
        list_=val_data,
        features=features,
        set_aggregate=args.set_aggregate,
        temporality=args.temporality,
        model=args.model,
        setting=args.setting
    )
    
    test_dataset = CustomDataset(
        list_=test_data,
        features=features,
        set_aggregate=args.set_aggregate,
        temporality=args.temporality,
        model=args.model,
        setting=args.setting
    )

    
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                             shuffle=True, collate_fn=collate_fn_),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                           shuffle=False, collate_fn=collate_fn_),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                            shuffle=False, collate_fn=collate_fn_)
    }
    dataset_sizes = {split: len(dataset) for split, dataset in 
                     zip(['train', 'val', 'test'], 
                    [train_dataset, val_dataset, test_dataset])}
    
    # Initialize model, optimizer, and criterion
    print(f"Initializing model {args.model}...")
    model, criterion, optimizer = helpers.get_model(
        model_name=args.model,  
        input_dim=args.input_dim,
        device=device,
        lr=args.lr,
        task="swdf",
        set_model_name=args.set_model_name,
        sequence_model_name=args.sequence_model_name,
        positional_embedding_type=args.positional_embedding,
        temporal_embedding_type=args.temporal_embedding,
    )

    
    # Train the model
    print("Starting training...")
    best_model = helpers.train_model_mesogeos(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,  # Optional scheduler
        model_name=args.model,
        monitor_accuracy=False,  # Set based on task
        early_stopping_patience=args.early_stopping_patience,
        save_path=args.save_path,
        device=device,
        num_epochs=args.epochs,
        use_timestamp=args.use_timestamp,
    )
    
    print(f"Training complete. Best model saved to {args.save_path}.")
    
    # Evaluate on the test set
    print("Evaluating on the test set...")
    helpers.evaluate_model_mesogeos(
        model_name=args.model,
        model=best_model,
        test_loader=dataloaders['test'],
        device=device,
        criterion=criterion,
        use_timestamp=args.use_timestamp,
    )