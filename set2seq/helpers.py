"""
Helper functions for model initialization, training, and evaluation.

This module provides:
- Model factory for initializing different architectures
- Unified training loops for both datasets
- Metric computation utilities
- Timestamp processing functions
"""
import numpy as np
import torch
import torch.nn.functional as F
import copy
import models
from datetime import datetime
from sklearn.metrics import mean_absolute_error, precision_recall_curve, precision_recall_fscore_support, auc
from scipy import stats
import utils


# =============================================================================
# Timestamp Processing
# =============================================================================

def transform_batch_timestamps_to_tensors(batch_timestamps, device='cuda'):
    """
    Transforms a batch of timestamp strings into tensors of day, month, and year.

    Args:
        batch_timestamps (list): List of timestamp sequences (YYYY-MM-DD format).
        device (str): Device to place tensors on.

    Returns:
        tuple: (day_tensor, month_tensor, year_tensor) on the specified device.
    """
    batch_size = len(batch_timestamps)
    seq_len = len(batch_timestamps[0])
    timestamp_count = len(batch_timestamps[0][0])

    days = torch.zeros(batch_size, seq_len, timestamp_count, dtype=torch.long)
    months = torch.zeros(batch_size, seq_len, timestamp_count, dtype=torch.long)
    years = torch.zeros(batch_size, seq_len, timestamp_count, dtype=torch.long)

    for i, sample_timestamps in enumerate(batch_timestamps):
        for j, timestamp_list in enumerate(sample_timestamps):
            for k, timestamp in enumerate(timestamp_list):
                dt = datetime.strptime(timestamp, '%Y-%m-%d')
                days[i, j, k] = dt.day
                months[i, j, k] = dt.month
                years[i, j, k] = dt.year

    return days.to(device), months.to(device), years.to(device)


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_classification_metrics(targets, predictions, probabilities):
    """
    Compute precision, recall, F1 score, and PR-AUC for binary classification.

    Args:
        targets (list): Ground truth labels.
        predictions (list): Predicted labels.
        probabilities (list): Predicted probabilities.

    Returns:
        dict: Dictionary containing precision, recall, F1 score, and PR-AUC.
    """
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        targets, predictions, average='binary', zero_division=0
    )
    
    if len(probabilities) > 0 and len(targets) > 0:
        precision_curve, recall_curve, _ = precision_recall_curve(targets, probabilities)
        pr_auc = auc(recall_curve, precision_curve)
    else:
        pr_auc = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "pr_auc": pr_auc,
    }


def compute_ranking_metrics(targets, predictions):
    """
    Compute ranking metrics: MAE, Kendall's Tau.

    Args:
        targets (list): Ground truth rankings.
        predictions (list): Predicted rankings.

    Returns:
        dict: Dictionary containing ranking metrics.
    """
    mae = mean_absolute_error(targets, predictions)
    kendall_tau = stats.kendalltau(targets, predictions)[0]
    
    return {
        "mae": mae,
        "kendall_tau": kendall_tau
    }


# =============================================================================
# Model Factory
# =============================================================================

def get_model(args, device=torch.device("cpu")):
    """
    Factory function to create models based on configuration.

    Args:
        args: Argument namespace with model configuration
        device (torch.device): Device to place the model on.

    Returns:
        tuple: (model, criterion, optimizer)
    """
    # Initialize model based on architecture type
    if args.model == "DeepSets":
        model = models.DeepSets(input_dim=args.input_dim, 
                                dim_hidden=args.set_dim_hidden,
                                output_dim=args.set_output_dim, 
                                pool=args.set_pool,
                                num_classes=args.num_classes)
        
    elif args.model == "HierarchicalDeepSets":
        model = models.HierarchicalDeepSets(set_input_dim=args.input_dim, 
                                            dim_hidden=args.set_dim_hidden,
                                            pool=args.set_pool, 
                                            num_classes=args.num_classes)
        
    elif args.model == "SetTransformer_SAB_PMA":
        model = models.SetTransformer_SAB_PMA(input_dim=args.input_dim, 
                                              num_classes=args.num_classes)
        
    elif args.model == "SetTransformer_ISAB_PMA":
        model = models.SetTransformer_ISAB_PMA(input_dim=args.input_dim, 
                                               num_classes=args.num_classes)
        
    elif args.model == "SetTransformer_ISAB_PMA_SAB":
        model = models.SetTransformer_ISAB_PMA_SAB(input_dim=args.input_dim, 
                                                   num_classes=args.num_classes)
        
    elif args.model == "HierarchicalSetTransformer":
        model = models.HierarchicalSetTransformer(set_input_dim=args.input_dim, 
                                                  num_classes=args.num_classes)
        
    elif args.model == "LSTM":
        model = models.LSTM(input_dim=args.input_dim, 
                            hidden_dim=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, 
                            output_dim=args.num_classes)
        
    elif args.model == "Transformer":
        model = models.Transformer(input_dim=args.input_dim, 
                                   model_dim=args.sequence_model_dim,
                                   num_heads=args.sequence_num_heads, 
                                   num_layers=args.sequence_num_layers,
                                   dropout=args.sequence_dropout, 
                                   num_classes=args.num_classes,
                                   positional_embedding_dim=args.positional_embedding_dim,  
                                   temporal_embedding_dim=args.temporal_embedding_dim,
                                   min_year=args.min_year,
                                   max_year=args.max_year)
        
    elif args.model == "Set2SeqTransformer":
        model = models.Set2SeqTransformer(
            set_input_dim=args.input_dim,
            set_dim_hidden=args.set_dim_hidden,
            set_output_dim=args.set_output_dim,
            sequence_input_dim=args.set_output_dim,
            sequence_model_dim=args.sequence_model_dim,
            sequence_num_classes=args.num_classes,
            sequence_num_heads=args.sequence_num_heads,
            sequence_num_layers=args.sequence_num_layers,
            sequence_dropout=args.sequence_dropout,
            set_model_name=args.set_model_name,
            sequence_model_name=args.sequence_model_name,
            positional_embedding_type=args.positional_embedding,
            temporal_embedding_type=args.temporal_embedding,
            positional_embedding_dim=args.positional_embedding_dim,
            temporal_embedding_dim=args.temporal_embedding_dim,    
            min_year=args.min_year,                                
            max_year=args.max_year                                 
        )
    else:
        raise ValueError(f"Unknown model: '{model_name}'")

    # Move model to device
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Select loss function based on task
    if args.task == "l2r":
        criterion = torch.nn.MSELoss()
    elif args.task == "swdf":
        criterion = torch.nn.NLLLoss(reduction='none')
    else:
        raise ValueError(f"Unknown task: '{args.task}'. Choose 'l2r' or 'swdf'.")

    return model, criterion, optimizer


# =============================================================================
# Unified Training Function
# =============================================================================

def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler=None,
    model_name="Transformer",
    task="swdf",
    device=torch.device("cpu"),
    epochs=25,
    early_stopping_patience=10,
    save_path=None,
    monitor_metric="loss",
    use_timestamp=True,
    disable_temporal_embedding=False,
):
    """
    Unified training loop for all models and datasets.

    Args:
        model (torch.nn.Module): Model to train.
        dataloaders (dict): Dict with 'train', 'val' dataloaders.
        dataset_sizes (dict): Dict with dataset sizes for each split.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler (optional).
        model_name (str): Name of the model.
        task (str): 'swdf' for classification, 'l2r' for ranking.
        device (torch.device): Device for training.
        epochs (int): Number of training epochs.
        early_stopping_patience (int): Patience for early stopping.
        save_path (str): Path to save best model.
        monitor_metric (str): Metric to monitor - 'loss', 'accuracy', or 'kendall_tau'.
        use_timestamp (bool): Whether to use temporal embeddings.
        disable_temporal_embedding (bool): Disable temporal embeddings (for ablation).

    Returns:
        tuple: (best_model, best_score, best_epoch)
    """
    since = datetime.now()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0.0 if monitor_metric != "loss" else float('inf')
    
    # Initialize early stopping
    monitor_accuracy = (monitor_metric != "loss")
    early_stopping = utils.EarlyStopping(
        path=save_path,
        accuracy=monitor_accuracy,
        patience=early_stopping_patience,
        verbose=True
    )

    for epoch in range(epochs):
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] Epoch {epoch}/{epochs - 1}')
        print('-' * 10)

        # Each epoch has training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_outputs = []
            all_targets = []
            all_probabilities = []

            # Iterate over data
            for batch_data in dataloaders[phase]:
                # Unpack batch (format depends on dataset/collate function)
                inputs = batch_data[0]
                labels = batch_data[1]
                positions = batch_data[2] if len(batch_data) > 2 else None
                temporal_values = batch_data[3] if len(batch_data) > 3 else None
                mask = batch_data[5] if len(batch_data) > 5 else None
                
                if task == 'swdf':
                    burned_area = torch.tensor([np.log(i) for i in batch_data[4]]).to(device)
                else:
                    artists_names = batch_data[4] # Only used for logging - if needed

                # Handle different input types
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(device)
                elif isinstance(inputs, list):
                    # For Set2SeqTransformer with variable-size sets
                    pass  # inputs stay as list
                
                # Handle labels
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                elif isinstance(labels, list):
                    labels = torch.tensor(labels).to(device)
                else:
                    labels = torch.from_numpy(np.asarray(labels)).to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    # Prepare model inputs based on model type
                    if 'Set2Seq' in model_name:
                        # Set2SeqTransformer: needs positions and temporal values
                        if positions is not None:
                            positions_tensor = torch.stack(positions).to(device) if isinstance(positions[0], torch.Tensor) else torch.tensor(positions).to(device)
                        else:
                            positions_tensor = None
                        
                        if use_timestamp and temporal_values is not None and not disable_temporal_embedding:
                            if isinstance(temporal_values[0], list):
                                # Mesogeos: convert timestamp strings
                                days, months, years = transform_batch_timestamps_to_tensors(temporal_values, device)
                                temporal_tensor = (days, months, years)
                            else:
                                # WikiArt: year values
                                temporal_tensor = torch.stack(temporal_values).long().to(device) if isinstance(temporal_values[0], torch.Tensor) else torch.tensor(temporal_values).long().to(device)
                        else:
                            temporal_tensor = None
                            
                        # Move mask to device if present
                        if mask is not None:
                            mask_tensor = mask.to(device).unsqueeze(1).unsqueeze(2)
                        else:
                            mask_tensor = None
                        
                        outputs = model(inputs, positions_tensor, temporal_tensor, mask_tensor)
                                            
                    elif model_name == 'Transformer':
                        # Sequence models: may need positions/temporal
                        positions_tensor = torch.tensor(positions).to(device) if positions is not None else None
                        
                        if use_timestamp and temporal_values is not None:
                            if isinstance(temporal_values[0], list):
                                days, months, years = transform_batch_timestamps_to_tensors(temporal_values, device)
                                temporal_tensor = (days, months, years)
                            else:
                                temporal_tensor = torch.tensor(temporal_values).long().to(device)
                        else:
                            temporal_tensor = None
                        
                        # Move mask to device if present
                        if mask is not None:
                            mask_tensor = mask.to(device).unsqueeze(1).unsqueeze(2)
                        else:
                            mask_tensor = None
                        
                        outputs = model(inputs, positions_tensor, temporal_tensor, mask_tensor)
                    
                    elif model_name == 'LSTM':
                        # LSTM: needs lengths parameter for packed sequences
                        lengths = None
                        if mask is not None:
                            mask_squeezed = mask.squeeze(1).squeeze(1) if len(mask.shape) > 2 else mask
                            lengths = mask_squeezed.sum(dim=1).cpu()
                        
                        outputs = model(inputs, lengths)
                        
                    else:
                        # Simple models (DeepSets, SetTransformer, etc.)
                        outputs = model(inputs)

                    # Compute loss based on task
                    if task == 'swdf':
                        # Classification: NLLLoss expects log probabilities
                        log_probs = F.log_softmax(outputs, dim=1)
                        loss = criterion(log_probs, labels)
                        loss = torch.mean(loss * burned_area)
                        probs = torch.exp(log_probs)
                        predictions = probs.argmax(dim=-1)
                        probabilities = probs[:,1]
                    else:
                        # Ranking/Regression: MSELoss
                        if labels.dim() == 1:
                            labels = labels.unsqueeze(1).float()
                        loss = criterion(outputs, labels)

                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                batch_size = inputs.size(0) if isinstance(inputs, torch.Tensor) else len(inputs)
                running_loss += loss.item() * batch_size
                
                # Collect outputs and targets
                if task == 'swdf':
                    all_outputs.extend(predictions.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().detach().numpy())
                else:
                    # Flatten outputs for ranking
                    if len(outputs.size()) == 0:
                        all_outputs.extend(outputs.unsqueeze(0).cpu().detach().numpy())
                    elif outputs.shape[-1] != 1:
                        all_outputs.extend(outputs.squeeze().cpu().detach().numpy())
                    else:
                        all_outputs.extend(outputs.cpu().detach().numpy())
                    
                    target_vals = labels.squeeze().cpu().detach().numpy() if labels.dim() > 1 else labels.cpu().detach().numpy()
                    if target_vals.ndim == 0:
                        all_targets.append(target_vals.item())
                    else:
                        all_targets.extend(target_vals)
            # Compute epoch metrics
            epoch_loss = running_loss / dataset_sizes[phase]
            
            # Flatten nested lists
            if task == 'l2r':
                all_outputs = [float(i) if not isinstance(i, (list, np.ndarray)) else float(i[0]) for i in all_outputs]
                all_targets = [float(i) if not isinstance(i, (list, np.ndarray)) else float(i[0]) for i in all_targets]

            # Compute task-specific metrics
            if task == 'swdf':
                metrics = compute_classification_metrics(all_targets, all_outputs, all_probabilities)
                print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] {phase} '
                      f'Loss: {epoch_loss:.5f} '
                      f'Precision: {metrics["precision"]:.5f} '
                      f'Recall: {metrics["recall"]:.5f} '
                      f'F1: {metrics["f1_score"]:.5f} '
                      f'PR-AUC: {metrics["pr_auc"]:.5f}')
                current_metric = metrics["f1_score"] if monitor_metric == "accuracy" else epoch_loss
            else:
                metrics = compute_ranking_metrics(all_targets, all_outputs)
                print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] {phase} '
                      f'Loss: {epoch_loss:.5f} '
                      f'MAE: {metrics["mae"]:.5f} '
                      f'Kendall\'s Tau: {metrics["kendall_tau"]:.5f} ')
                current_metric = metrics["kendall_tau"] if monitor_metric == "kendall_tau" else epoch_loss

            # Early stopping (only on validation set)
            if phase == 'val':
                if scheduler:
                    scheduler.step(current_metric)
                
                early_stopping(
                    epoch=epoch,
                    current_score=current_metric,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=scheduler
                )
                
                if early_stopping.early_stop:
                    print(f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] Early stopping")
                    time_elapsed = datetime.now() - since
                    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
                          f'Training complete in {time_elapsed.seconds // 60}m {time_elapsed.seconds % 60}s')
                    print(f'Best validation score: {early_stopping.best_score:.4f}')
                    model.load_state_dict(best_model_wts)
                    return model, early_stopping.best_score, epoch

            # Update best model
            if phase == 'val':
                is_better = (current_metric > best_score) if monitor_metric != "loss" else (current_metric < best_score)
                if is_better:
                    best_score = current_metric
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # Training complete
    time_elapsed = datetime.now() - since
    print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}] '
          f'Training complete in {time_elapsed.seconds // 60}m {time_elapsed.seconds % 60}s')
    print(f'Best validation score: {best_score:.4f}')

    model.load_state_dict(best_model_wts)
    return model, best_score, epochs - 1