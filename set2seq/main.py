"""
Main training script for Set2Seq Transformer.

Supports two datasets:
- Mesogeos: Wildfire forecasting (binary classification)
- WikiArt-Seq2Rank: Artist career prediction (learning-to-rank)

Usage:
    # Mesogeos
    python3 main.py --dataset mesogeos --model Set2SeqTransformer \
        --data_path ../datasets/mesogeos_dataset/mesogeos_dataset_swdf.pkl \
        --features ../datasets/mesogeos_dataset/mesogeos_dataset_swdf_features.pkl

    # WikiArt
    python3 main.py --dataset wikiart_seq2rank --model Set2SeqTransformer \
        --data_path ../datasets/wikiart_seq2rank_dataset/wikiart_seq2rank_dataset.pkl \
        --features ../datasets/wikiart_seq2rank_dataset/features_resnet34.pkl
"""
import os
import numpy as np
import argparse
import yaml
import torch
import helpers
import utils
from dataloader import (
    collate_fn_mesogeos,
    collate_fn_wikiart_flattened,
    collate_fn_wikiart_sets
)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate Set2Seq Transformer models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help="Path to YAML config file (overrides defaults)")
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, required=False, 
                       choices=['mesogeos', 'wikiart_seq2rank'],
                       help="Dataset to use: 'mesogeos' or 'wikiart_seq2rank'")
    parser.add_argument('--data_path', type=str, required=False,
                       help="Path to the dataset pickle file")
    parser.add_argument('--features', type=str, required=False,
                       help="Path to the features pickle file")
    
    # Model configuration
    parser.add_argument('--model', type=str, required=False,
                       choices=['Set2SeqTransformer', 'Transformer', 'LSTM',
                               'DeepSets', 'HierarchicalDeepSets',
                               'SetTransformer_SAB_PMA', 'SetTransformer_ISAB_PMA',
                               'SetTransformer_ISAB_PMA_SAB', 'HierarchicalSetTransformer'],
                       help="Model architecture to use")
    parser.add_argument('--input_dim', type=int, default=None,
                       help="Input dimensionality (auto-detected if not specified)")
    parser.add_argument('--num_classes', type=int, default=None,
                       help="Number of output classes (auto-detected if not specified)")
    
    # Set encoder hyperparameters
    parser.add_argument('--set_dim_hidden', type=int, default=256,
                       help="Hidden dimension for set encoder")
    parser.add_argument('--set_output_dim', type=int, default=512,
                       help="Output dimension for set encoder")
    parser.add_argument('--set_pool', type=str, default='max',
                       choices=['max', 'mean', 'sum'],
                       help="Pooling method for DeepSets")
    
    # Sequence encoder hyperparameters
    parser.add_argument('--sequence_model_dim', type=int, default=512,
                       help="Model dimension for sequence encoder")
    parser.add_argument('--sequence_num_heads', type=int, default=8,
                       help="Number of attention heads for Transformer")
    parser.add_argument('--sequence_num_layers', type=int, default=6,
                       help="Number of layers for sequence encoder")
    parser.add_argument('--sequence_dropout', type=float, default=0.0,
                       help="Dropout rate for sequence encoder")
    
    # LSTM specific
    parser.add_argument('--lstm_hidden_dim', type=int, default=512,
                       help="Hidden dimension for LSTM")
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                       help="Number of LSTM layers")
    
    # Set2SeqTransformer specific
    parser.add_argument('--set_model_name', type=str, default='DeepSets',
                       choices=['DeepSets', 'SetTransformer_SAB_PMA', 
                               'SetTransformer_ISAB_PMA', 'SetTransformer_ISAB_PMA_SAB'],
                       help="Set encoder for Set2SeqTransformer")
    parser.add_argument('--sequence_model_name', type=str, default='Transformer',
                       choices=['Transformer', 'LSTM'],
                       help="Sequence encoder for Set2SeqTransformer")
    parser.add_argument('--positional_embedding', type=str, default='positional_encoding',
                       choices=['positional_encoding', 'positional_embedding', 'none'],
                       help="Type of positional embedding")
    parser.add_argument('--temporal_embedding', type=str, default='timestamp_time2vec',
                       choices=['timestamp_time2vec', 'positional_embedding', 'none'],
                       help="Type of temporal embedding")
    parser.add_argument('--disable_temporal_embedding', action='store_true',
                       help="Disable temporal embeddings (ablation study)")
    
    # Dataset-specific options
    parser.add_argument('--set_aggregate', type=str, choices=['mean', 'max'], default=None,
                       help="Aggregation function for non-Set2Seq models")
    parser.add_argument('--temporality', action='store_true', default=True,
                       help="Use temporal information")
    parser.add_argument('--use_timestamp', action='store_true', default=True,
                       help="Use timestamp/temporal embeddings during training")
    parser.add_argument('--setting', type=int, default=1,
                       help="[Mesogeos] Grouping factor for timesteps (1=no grouping)")
    parser.add_argument('--wikiart_seq2rank_split', type=str, default='stratified_split',
                       help="[WikiArt] Split type: 'stratified_split' or 'time_series_split'")
    parser.add_argument('--wikiart_seq2rank_ranking', type=str, default='overall',
                       help="[WikiArt] Ranking type to use")
    
    # Training configuration
    parser.add_argument('--scheduler', type=str, default=None,
                       choices=[None, 'cosine', 'step', 'plateau'],
                       help="Learning rate scheduler")
    parser.add_argument('--lr', type=float, default=0.00001,
                       help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=256,
                       help="Batch size")
    parser.add_argument('--epochs', type=int, default=1,
                       help="Number of epochs")
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                       help="Early stopping patience")
    parser.add_argument('--monitor_metric', type=str, default='auto',
                       choices=['auto', 'loss', 'accuracy', 'kendall_tau'],
                       help="Metric to monitor for early stopping ('auto' chooses based on task)")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help="Device to use for training")
    
    # Output configuration
    parser.add_argument('--save', action='store_true',
                       help="Save the best model checkpoint")
    parser.add_argument('--save_path', type=str, default='.',
                       help="Directory to save the model checkpoint")
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update args with config values (CLI args override config)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Validate required arguments
    if args.dataset is None:
        parser.error("--dataset is required (either via CLI or config file)")
    if args.data_path is None:
        parser.error("--data_path is required (either via CLI or config file)")
    if args.features is None:
        parser.error("--features is required (either via CLI or config file)")
    
    # Post-process arguments
    args = post_process_args(args)
    
    return args


def post_process_args(args):
    """Post-process and validate arguments."""
    
    # Map aggregation string to function
    if args.set_aggregate == 'mean':
        args.set_aggregate = np.mean
    elif args.set_aggregate == 'max':
        args.set_aggregate = np.max
    
    # Auto-detect task based on dataset
    if args.dataset == 'mesogeos':
        args.task = 'swdf'
        args.num_classes = args.num_classes or 2
        args.input_dim = args.input_dim or 24
    elif args.dataset == 'wikiart_seq2rank':
        args.task = 'l2r'
        args.num_classes = args.num_classes or 1
        args.input_dim = args.input_dim or 512
    
    # Auto-select monitoring metric
    if args.monitor_metric == 'auto':
        if args.task == 'swdf':
            args.monitor_metric = 'loss'
        else:
            args.monitor_metric = 'kendall_tau'
    
    # Handle positional/temporal embedding 'none' option
    if args.positional_embedding == 'none':
        args.positional_embedding = None
    if args.temporal_embedding == 'none':
        args.temporal_embedding = None
    
    # Generate save path
    if args.save:
        filename = f"{args.model}_{args.dataset}_setting{args.setting}.pt"
        args.save_path = os.path.join(args.save_path, filename)
    else:
        args.save_path = None
    
    return args


# =============================================================================
# Data Loading
# =============================================================================

def load_data(args):
    """
    Load data based on dataset type.
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, collate_fn)
    """
    if args.dataset == 'mesogeos':
        train_dataset, val_dataset, test_dataset, min_year, max_year, max_seq_len = utils.load_mesogeos_data(
            args.data_path, args.features, args.setting, args.set_aggregate, args.model
        )
        collate_fn = collate_fn_mesogeos
        print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        print(f"Using setting: {args.setting}")
        
        args.positional_embedding_dim = max_seq_len
        args.temporal_embedding_dim = (max_year - min_year + 1) + 1
        args.min_year = min_year
        args.max_year = max_year
        
    elif args.dataset == 'wikiart_seq2rank':
        train_dataset, val_dataset, test_dataset, min_year, max_year, max_seq_len = utils.load_wikiart_data(
            args.data_path, args.features, args.wikiart_seq2rank_ranking, args.wikiart_seq2rank_split, args.model
        )
        use_sets = 'Set2Seq' in args.model
        collate_fn = collate_fn_wikiart_sets if use_sets else collate_fn_wikiart_flattened
        print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        print(f"Using ranking: {args.wikiart_seq2rank_ranking}")
        print(f"Mode: {'variable-size sets' if use_sets else 'flattened sequences'}")
        
        args.positional_embedding_dim = max_seq_len
        args.temporal_embedding_dim = (max_year - min_year + 1) + 1
        args.min_year = min_year
        args.max_year = max_year
        
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return train_dataset, val_dataset, test_dataset, collate_fn


# =============================================================================
# Main Training
# =============================================================================

def main():
    """Main training pipeline."""
    
    # Parse arguments
    args = parse_args()
    
    # Set device and seed
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    utils.set_seed(args.seed)
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    
    # Load data
    train_dataset, val_dataset, test_dataset, collate_fn = load_data(args)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }
    
    # Initialize model
    print(f"\nInitializing model: {args.model}")
    if args.model == 'Set2SeqTransformer':
        print(f"  Set encoder: {args.set_model_name}")
        print(f"  Sequence encoder: {args.sequence_model_name}")
        print(f"  Positional embedding: {args.positional_embedding}")
        print(f"  Temporal embedding: {args.temporal_embedding}")
    
    model, criterion, optimizer = helpers.get_model(args, device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize scheduler if specified
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = utils.CosineWarmupScheduler(optimizer, warmup=10, max_iters=args.epochs)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    
    # Train the model
    print(f"\n{'='*80}")
    print(f"Starting training...")
    print(f"{'='*80}\n")
    
    best_model, best_score, best_epoch = helpers.train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        model_name=args.model,
        task=args.task,
        device=device,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        save_path=args.save_path,
        monitor_metric=args.monitor_metric,
        use_timestamp=args.use_timestamp,
        disable_temporal_embedding=args.disable_temporal_embedding,
    )
    
    print(f"\n{'='*80}")
    print(f"Training complete!")
    print(f"Best model at epoch {best_epoch} with score: {best_score:.4f}")
    if args.save_path:
        print(f"Model saved to: {args.save_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()