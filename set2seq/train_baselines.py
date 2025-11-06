"""
Train simple baseline models (Linear Regression, XGBoost) with aggregation.
Matches the original WikiArt baseline implementation.

Example usage:
    python3 train_baselines.py --set_aggregate=mean --model=xgb
"""

import argparse
import pickle
import numpy as np
import torch
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

import sys
sys.path.append('.')

from baseline_utils import SimpleBaselineDataset, simple_collate_fn, aggregate_features


def evaluate_predictions(y_true, y_pred, phase='Test'):
    """Compute and print evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    tau = stats.kendalltau(y_pred, y_true)[0]
    
    print(f"{phase} - MSE: {mse:.5f}, MAE: {mae:.5f}, Kendall's Tau: {tau:.5f})
    
    return mse, mae, tau


def main():
    parser = argparse.ArgumentParser(description='Train baseline models')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                        default='../datasets/wikiart_seq2rank/wikiart_seq2rank.pkl',
                        help='Path to dataset pickle file')
    parser.add_argument('--features_path', type=str,
                        default='../datasets/wikiart_seq2rank/wikiart_seq2rank_features_resnet34.pkl',
                        help='Path to precomputed features')
    
    # Aggregation arguments  
    parser.add_argument('--set_aggregate', type=str, default='mean',
                        choices=['mean', 'max'],
                        help='Set aggregation function')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='xgb',
                        choices=['lr', 'xgb'],
                        help='Model type: lr (Linear Regression) or xgb (XGBoost)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Settings
    parser.add_argument('--setting', type=str, default='stratified_split',
                        choices=['stratified_split', 'time_series_split'],
                        help='Data split setting')
    parser.add_argument('--ranking', type=str, default='overall',
                        help='Ranking type')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load data
    print(f"Loading data from {args.data_path}")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    data = data[args.setting]
    
    # Remove empty sequences
    data = {k: v for k, v in data.items() if len(v['sequence']) > 0}
    
    # Load features
    print(f"Loading features from {args.features_path}")
    with open(args.features_path, 'rb') as f:
        features = pickle.load(f)
    
    # Handle different feature key formats
    if '/' in next(iter(features.keys())):
        features = {k.split('/')[-1]: v for k, v in features.items()}
    
    print(f"Setting: {args.setting}, Ranking: {args.ranking}")
    
    # Extract data splits
    X_train = [v['sequence'] for v in data.values() if 'train' in v['rankings'][args.ranking]]
    y_train = [v['rankings'][args.ranking]['train'] for v in data.values() if 'train' in v['rankings'][args.ranking]]
    names_train = [k for k, v in data.items() if 'train' in v['rankings'][args.ranking]]
    
    X_val = [v['sequence'] for v in data.values() if 'val' in v['rankings'][args.ranking]]
    y_val = [v['rankings'][args.ranking]['val'] for v in data.values() if 'val' in v['rankings'][args.ranking]]
    names_val = [k for k, v in data.items() if 'val' in v['rankings'][args.ranking]]
    
    X_test = [v['sequence'] for v in data.values() if 'test' in v['rankings'][args.ranking]]
    y_test = [v['rankings'][args.ranking]['test'] for v in data.values() if 'test' in v['rankings'][args.ranking]]
    names_test = [k for k, v in data.items() if 'test' in v['rankings'][args.ranking]]
    
    # Normalize labels
    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()
    y_val = scaler.transform(np.array(y_val).reshape(-1, 1)).ravel()
    y_test = scaler.transform(np.array(y_test).reshape(-1, 1)).ravel()
    
    # Aggregate features
    print(f"Aggregating features using {args.set_aggregate}...")
    X_train_agg = aggregate_features(X_train, features, args.set_aggregate)
    X_val_agg = aggregate_features(X_val, features, args.set_aggregate)
    X_test_agg = aggregate_features(X_test, features, args.set_aggregate)
    
    print(f"Train: {X_train_agg.shape}, Val: {X_val_agg.shape}, Test: {X_test_agg.shape}")
    
    # Train model
    print(f"\nTraining {args.model.upper()} model...")
    if args.model == 'lr':
        model = LinearRegression()
    elif args.model == 'xgb':
        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=args.seed)
    
    model.fit(X_train_agg, y_train)
    
    # Predict
    preds_train = model.predict(X_train_agg)
    preds_val = model.predict(X_val_agg)
    preds_test = model.predict(X_test_agg)
    
    # Evaluate
    print("\nResults:")
    evaluate_predictions(y_train, preds_train, 'Train')
    evaluate_predictions(y_val, preds_val, 'Val')
    evaluate_predictions(y_test, preds_test, 'Test')


if __name__ == '__main__':
    main()