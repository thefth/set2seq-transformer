# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
import utils
import copy
import models

from datetime import datetime
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, auc

def transform_batch_timestamps_to_tensors(batch_timestamps, device='cuda'):
    """
    Transforms a batch of timestamp strings into tensors of day, month, and year.

    Args:
        batch_timestamps (list): List of timestamp sequences (YYYY-MM-DD format).

    Returns:
        tuple: day_tensor, month_tensor, year_tensor on the specified device.
    """
    batch_size = len(batch_timestamps)
    seq_len = len(batch_timestamps[0])
    timestamp_count = len(batch_timestamps[0][0])

    days, months, years = (
        torch.zeros(batch_size, seq_len, timestamp_count, dtype=torch.long),
        torch.zeros(batch_size, seq_len, timestamp_count, dtype=torch.long),
        torch.zeros(batch_size, seq_len, timestamp_count, dtype=torch.long),
    )

    for i, sample_timestamps in enumerate(batch_timestamps):
        for j, timestamp_list in enumerate(sample_timestamps):
            for k, timestamp in enumerate(timestamp_list):
                dt = datetime.strptime(timestamp, '%Y-%m-%d')
                days[i, j, k] = dt.day
                months[i, j, k] = dt.month
                years[i, j, k] = dt.year

    return days.to(device), months.to(device), years.to(device)


def compute_metrics(targets, predictions, probabilities):
    """
    Compute precision, recall, F1 score, and PR-AUC.

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




def get_model(
    model_name="Transformer",
    input_dim=24,
    num_classes=2,
    lr=0.0001,
    task="swdf",
    set_model_name="SmallDeepSets",
    sequence_model_name="Transformer",
    positional_embedding_type="positional_encoding",
    temporal_embedding_type="timestamp_time2vec",
    device=torch.device("cpu"),
):
    """
    Returns the model, criterion, and optimizer based on the given configurations.

    Args:
        model_name (str): Name of the model to use.
        inputs_dim (int): Input dimensionality for the model.
        lr (float): Learning rate.
        task (str): Task type, either "l2r" or "binary_classification".
        set_model_name (str): Base model for set-to-sequence models.
        te (str): Temporal encoding type.
        device (torch.device): Device to place the model on.

    Returns:
        model (torch.nn.Module): The initialized model.
        criterion: Loss function.
        optimizer: Optimizer for the model.
    """
    # Model initialization based on the model name
    if model_name == "DeepSets":
        model = models.DeepSets(input_dim=input_dim,
                                num_classes=num_classes)
    elif model_name == "HierarchicalDeepSets":
        model = models.HierarchicalDeepSets(set_input_dim=input_dim, 
                                            num_classes=num_classes)
    elif model_name == "SetTransformer_SAB_PMA":
        model = models.SetTransformer_SAB_PMA(input_dim=input_dim, 
                                              num_classes=num_classes)
    elif model_name == "SetTransformer_ISAB_PMA":
        model = models.SetTransformer_ISAB_PMA(input_dim=input_dim, 
                                               num_classes=num_classes)
    elif model_name == "SetTransformer_ISAB_PMA_SAB":
        model = models.SetTransformer_ISAB_PMA_SAB(input_dim=input_dim, 
                                                   num_classes=num_classes)
    elif model_name == "HierarchicalSetTransformer":
        model = models.HierarchicalSetTransformer(set_input_dim=input_dim, 
                                                  num_classes=num_classes)
    elif model_name == "LSTM":
        model = models.LSTM(input_dim=input_dim, 
                            hidden_dim=512, 
                            num_layers=2, 
                            output_dim=2,
                            )
    elif model_name == "Transformer":
        model = models.Transformer(input_dim=input_dim, 
                                   num_classes=num_classes)
    elif model_name == "Set2SeqTransformer":
        model = models.Set2SeqTransformer(set_input_dim=input_dim,
                                          sequence_num_classes=num_classes,
                                          set_model_name=set_model_name,
                                          sequence_model_name=sequence_model_name,
                                          positional_embedding_type=positional_embedding_type,
                                          temporal_embedding_type=temporal_embedding_type)
    else:
        raise ValueError(f"Model '{model_name}' is not recognized. Check the model name.")

    # Move model to the specified device
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss function based on the task
    if task == "l2r":
        criterion = torch.nn.MSELoss()
    elif task == "swdf":
        criterion = torch.nn.NLLLoss(reduction='none')
    else:
        raise ValueError(f"Task '{task}' is not recognized. Choose 'l2r' or 'swdf'.")

    return model, criterion, optimizer



def train_model_mesogeos(
    model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
    model_name, device, num_epochs=25, early_stopping_patience=10, save_path=None,
    monitor_accuracy=False, use_timestamp=True
):
    """
    Train the model with early stopping and checkpointing.

    Args:
        model (torch.nn.Module): The model to train.
        dataloaders (dict): Dictionary containing 'train' and 'val' dataloaders.
        dataset_sizes (dict): Dictionary with dataset sizes for 'train' and 'val'.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        model_name (str): Model name (e.g., 'Transformer').
        device (torch.device): Device to run the training on.
        num_epochs (int): Number of epochs.
        early_stopping_patience (int): Early stopping patience.
        save_path (str): Path to save the best model.
        monitor_accuracy (bool): Whether to monitor accuracy for early stopping.
        use_timestamp (bool): Whether to use timestamps.

    Returns:
        torch.nn.Module: The trained model.
    """
    since = datetime.now()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = np.Inf

    early_stopping = utils.EarlyStopping(
        path=save_path, accuracy=monitor_accuracy, patience=early_stopping_patience, verbose=True
    )

    for epoch in range(num_epochs):
        print(f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, outputs_, outputs_prob_, targets_ = 0.0, [], [], []

            for inputs, labels, burned_area, positions, timestamps in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.long().to(device)
                burned_area = torch.tensor([np.log(i) for i in burned_area]).to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if model_name=='Transformer':
                        
                        outputs = model(inputs, positions)
                        
                    elif model_name=='Set2SeqTransformer':
                                                
                        if use_timestamp:
                            temporal_values = torch.stack(transform_batch_timestamps_to_tensors(timestamps,
                                                                                                device=device))
                            
                        if len(inputs.shape)==3:
                            
                            inputs = inputs.unsqueeze(-2)
                            
                        outputs = model(inputs, positions, temporal_values)
                        
                        
                    else:
                        outputs = model(inputs.float())
                    loss = criterion(F.log_softmax(outputs, dim=1), labels)
                    weighted_loss = torch.mean(loss * burned_area)

                    if phase == 'train':
                        weighted_loss.backward()
                        optimizer.step()

                running_loss += weighted_loss.item() * inputs.size(0)
                outputs_prob = torch.exp(outputs)
                predictions = outputs_prob.argmax(dim=1)
                outputs_.extend(predictions.cpu().tolist())
                targets_.extend(labels.cpu().tolist())
                outputs_prob_.extend(outputs_prob[:, 1].cpu().tolist())

            epoch_loss = running_loss / dataset_sizes[phase]
            metrics = compute_metrics(targets_, outputs_, outputs_prob_)
            print(f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] {phase} Loss: {epoch_loss:.3f} Precision: {metrics['precision']:.3f} Recall: {metrics['recall']:.3f} F1: {metrics['f1_score']:.3f} PR-AUC: {metrics['pr_auc']:.3f}")

            if phase == 'val':
                
                early_stopping(epoch=epoch, current_score=epoch_loss, model=model, optimizer=optimizer, lr_scheduler=scheduler)
                if early_stopping.early_stop:
                    print(f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] Early stopping")
                    model.load_state_dict(best_model_wts)
                    return model

            if phase == 'val' and epoch_loss < best_metric:
                best_metric = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = datetime.now() - since
    print(f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] Training complete in {time_elapsed.seconds // 60}m {time_elapsed.seconds % 60}s")
    print(f"Best val Loss: {best_metric:.4f}")

    model.load_state_dict(best_model_wts)
    return model


def evaluate_model_mesogeos(model_name, model, test_loader, device, criterion, use_timestamp=False):
    """
    Evaluate the model on the test set.

    Args:
        model_name (str): Name of the model (e.g., 'Transformer').
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device for evaluation.
        criterion: Loss function.
        use_timestamp (bool): Whether to use timestamps.

    Returns:
        dict: Dictionary containing test loss and evaluation metrics (precision, recall, F1, PR-AUC).
    """
    model.eval()
    running_loss, outputs_, outputs_prob_, targets_ = 0.0, [], [], []

    with torch.no_grad():
        for inputs, labels, burned_area, positions, timestamps in test_loader:
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            burned_area = torch.tensor([np.log(i) for i in burned_area]).to(device)

            if model_name=='Transformer':
                        
                outputs = model(inputs, positions)
                
            elif model_name=='Set2SeqTransformer':
                
                
                if use_timestamp:
                    temporal_values = torch.stack(transform_batch_timestamps_to_tensors(timestamps,
                                                                                        device=device))
                    
                if len(inputs.shape)==3:
                    
                    inputs = inputs.unsqueeze(-2)
                    
                outputs = model(inputs, positions, temporal_values)
                
                
            else:
                outputs = model(inputs.float())
            loss = criterion(F.log_softmax(outputs, dim=1), labels)
            weighted_loss = torch.mean(loss * burned_area)

            running_loss += weighted_loss.item() * inputs.size(0)
            outputs_prob = torch.exp(outputs)
            predictions = outputs_prob.argmax(dim=1)
            outputs_.extend(predictions.cpu().tolist())
            targets_.extend(labels.cpu().tolist())
            outputs_prob_.extend(outputs_prob[:, 1].cpu().tolist())

    test_loss = running_loss / len(test_loader.dataset)
    metrics = compute_metrics(targets_, outputs_, outputs_prob_)
    metrics["loss"] = test_loss

    print(f"[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] Test Loss: {test_loss:.3f} Precision: {metrics['precision']:.3f} Recall: {metrics['recall']:.3f} F1: {metrics['f1_score']:.3f} PR-AUC: {metrics['pr_auc']:.3f}")

    return metrics