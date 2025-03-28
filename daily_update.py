# This script performs daily updates for stock prediction models. It processes new data files, updates the dataset, 
# re-trains models for specified tickers, and generates predictions for the next day and the next 5 days. 
# The script also logs predictions and provides trading strategy recommendations.
# Functions:
#     - update_dataset_from_file(data, file_path): Processes a single update file and merges it with the existing dataset.
#     - update_dataset(update_dir=None): Updates the dataset with new data from all files in the specified directory.
#     - train_ticker_model(ticker_data, ticker_name, seq_length=SEQ_LENGTH): Trains a model for a specific ticker and saves it.
#     - retrain_models(data): Retrains models for all target tickers using the updated dataset.
#     - make_predictions(data): Generates predictions for all tickers using the latest models.
#     - daily_update(update_dir=None, skip_retrain=False, clear_update_files=True): Performs a complete daily update cycle.
# Command-line Arguments:
#     --update-dir: Directory containing update files (default: NEW_DATA_DIR).
#     --skip-retrain: If specified, skips model retraining.
#     --keep-files: If specified, keeps update files after processing.
# Usage:
#     Run the script with optional arguments to update the dataset, retrain models, and generate predictions.
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os
import json
import argparse
import datetime
import glob
import warnings
warnings.filterwarnings('ignore')

# Set fixed random seeds for reproducibility 
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Import existing functions to ensure consistency
from step2_find_most_predictable_stocks import TimeSeriesPredictor, create_features, prepare_data_sequences
from step3_make_prediction_for_next_day import predict_multi_horizon

# Define the tickers for daily updates
PREDICTABLE_TICKERS = ['fccl', 'mlcf']

# File paths
SOURCE_FILE = 'data/daily_stocks.pkl'
MODEL_DIR = 'model'
HISTORY_DIR = 'model_history'
SEQ_LENGTH = 5
NEW_DATA_DIR = 'data/daily/new'  # Directory for new update files

# Ensure necessary directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs(NEW_DATA_DIR, exist_ok=True)  # Ensure the new updates directory exists

def update_dataset_from_file(data, file_path):
    """Process a single update file and merge with existing data"""
    print(f"Processing file: {file_path}")
    
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as file:
                new_data = pd.DataFrame(json.loads(file.read()).get("data", {}))
        elif file_path.endswith('.csv'):
            new_data = pd.read_csv(file_path)
        else:
            print(f"Skipping unsupported file format: {file_path}")
            return data
            
        # Process new data
        new_data["date"] = pd.to_datetime(new_data["date"], errors='coerce')
        
        # Ensure the new data has ticker information
        if "ticker" not in new_data.columns:
            if "ticker_name" in new_data.columns:
                new_data["ticker"] = new_data["ticker_name"]
            else:
                print(f"Warning: File {file_path} missing ticker column, skipping")
                return data
        
        # Set ticker_name if it doesn't exist
        if "ticker_name" not in new_data.columns:
            new_data["ticker_name"] = new_data["ticker"]
        
        # Filter for our tickers of interest
        new_data = new_data[new_data["ticker_name"].isin(PREDICTABLE_TICKERS)]
        
        if len(new_data) == 0:
            print(f"No relevant data found in {file_path}")
            return data
        
        # Merge with existing data
        combined_data = pd.concat([data, new_data], ignore_index=True)
        
        # Remove duplicates based on ticker and date
        combined_data = combined_data.drop_duplicates(subset=["ticker_name", "date"])
        
        # Re-sort the data
        combined_data = combined_data.sort_values(["ticker_name", "date"]).reset_index(drop=True)
        
        print(f"Added {len(new_data)} new rows from {file_path}")
        return combined_data
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return data

def update_dataset(update_dir=None):
    """
    Update the dataset with new data from all files in the specified directory
    
    Args:
        update_dir: Directory containing JSON or CSV files with new data
            If None, uses the default NEW_DATA_DIR
            If no files found, will use existing data without modifications
    
    Returns:
        Updated DataFrame
    """
    print("Checking for dataset updates...")
    
    # Use default directory if none specified
    if update_dir is None:
        update_dir = NEW_DATA_DIR
    
    # Load existing dataset
    if os.path.exists(SOURCE_FILE):
        data = pd.read_pickle(SOURCE_FILE)
        print(f"Loaded existing dataset with {len(data)} rows")
    else:
        raise FileNotFoundError(f"Base dataset not found at {SOURCE_FILE}")
    
    # Get all update files
    update_files = glob.glob(os.path.join(update_dir, "*.json"))
    update_files.extend(glob.glob(os.path.join(update_dir, "*.csv")))
    
    if update_files:
        print(f"Found {len(update_files)} update files")
        
        # Create a copy of the data to avoid modifying the original
        updated_data = data.copy()
        
        # Process each file
        for file_path in sorted(update_files):  # Sort to ensure consistent processing order
            updated_data = update_dataset_from_file(updated_data, file_path)
            
        print(f"Processed all update files. Dataset now has {len(updated_data)} rows.")
        
        # Save the updated dataset only if there were actual updates
        updated_data.to_pickle(SOURCE_FILE)
        print(f"Updated dataset saved to {SOURCE_FILE}")
        return updated_data
    else:
        print(f"No update files found in {update_dir}")
        print("Using existing dataset without modifications.")
        # Return the data directly without any modifications
        return data

def train_ticker_model(ticker_data, ticker_name, seq_length=SEQ_LENGTH):
    """Train a model for a specific ticker and save it"""
    print(f"\nTraining model for {ticker_name}...")
    
    # Sort by date
    ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
    
    # Split into train and validation (keep the most recent data for validation)
    train_size = int(len(ticker_data) * 0.8)
    train_data = ticker_data.iloc[:train_size]
    val_data = ticker_data.iloc[train_size:]
    
    # Prepare the data
    X_train, y_train, train_scaler = prepare_data_sequences(train_data, seq_length)
    X_val, y_val, _ = prepare_data_sequences(val_data, seq_length)
    
    # Check class distribution
    up_ratio = y_train[1].float().mean().item()
    print(f"Up direction ratio in training: {up_ratio:.4f}")
    
    # Skip training if data is heavily imbalanced (all one class)
    if up_ratio == 0.0 or up_ratio == 1.0:
        print(f"Warning: Only one class present for {ticker_name}. Skipping training.")
        return None
    
    # Initialize model
    input_size = X_train.shape[2]
    model = TimeSeriesPredictor(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.5)
    
    # Loss functions with class weighting if needed
    regression_loss = nn.MSELoss()
    if up_ratio > 0.6 or up_ratio < 0.4:
        # Cap weights to avoid extreme values
        up_ratio = min(max(up_ratio, 0.01), 0.99)
        weight = torch.tensor([1/(1-up_ratio), 1/up_ratio])
        classification_loss = nn.CrossEntropyLoss(weight=weight)
        print("Using weighted loss due to class imbalance")
    else:
        classification_loss = nn.CrossEntropyLoss()
    
    # Train the model
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    train_dataset = TensorDataset(X_train, y_train[0], y_train[1])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    best_val_acc = 0
    best_model = None
    patience = 0
    max_patience = 10
    
    for epoch in range(50):
        # Training
        model.train()
        total_loss = 0
        for batch_X, batch_y_reg, batch_y_cls in train_loader:
            reg_out, cls_out = model(batch_X)
            reg_loss = regression_loss(reg_out, batch_y_reg)
            cls_loss = classification_loss(cls_out, batch_y_cls)
            combined_loss = 0.2 * reg_loss + 0.8 * cls_loss
            
            optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += combined_loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            _, val_cls_out = model(X_val)
            val_cls_preds = torch.argmax(val_cls_out, dim=1)
            val_accuracy = (val_cls_preds == y_val[1]).float().mean().item()
            
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_model = model.state_dict()
                patience = 0
            else:
                patience += 1
        
        print(f"Epoch {epoch+1}/50: loss={total_loss/len(train_loader):.4f}, val_acc={val_accuracy:.4f}")
        
        # Early stopping
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Store the best model
    if best_model:
        # Make model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save current best model
        model_path = os.path.join(MODEL_DIR, f'best_{ticker_name}_model.pth')
        
        # Archive old model if it exists
        if os.path.exists(model_path):
            # Create a timestamped copy in the history directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(HISTORY_DIR, f'best_{ticker_name}_model_{timestamp}.pth')
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)
            
            # Copy the old model to the archive
            import shutil
            shutil.copy2(model_path, archive_path)
            print(f"Archived previous model to {archive_path}")
        
        # Save new model
        model.load_state_dict(best_model)
        torch.save(best_model, model_path)
        print(f"Saved best model to {model_path} with validation accuracy: {best_val_acc:.4f}")
        
        return model
    else:
        print(f"Warning: No best model found for {ticker_name}")
        return None

def retrain_models(data):
    """Retrain models for all target tickers"""
    print("\nRetraining models with updated data...")
    
    # Create features
    featured_data = create_features(data)
    
    # Train models for each ticker
    trained_models = {}
    for ticker in PREDICTABLE_TICKERS:
        ticker_data = featured_data[featured_data['ticker_name'] == ticker].copy()
        
        # Skip tickers with insufficient data
        if len(ticker_data) < 100:
            print(f"Skipping {ticker}: insufficient data ({len(ticker_data)} data points)")
            continue
        
        # Train model
        model = train_ticker_model(ticker_data, ticker)
        if model:
            trained_models[ticker] = model
    
    return trained_models

def make_predictions(data):
    """Make predictions for all tickers using the latest models"""
    print("\nGenerating predictions using updated models...")
    
    # This code now exactly matches the approach in step3_make_prediction_for_next_day.py
    # Filter data for the selected tickers first
    filtered_data = data[data['ticker_name'].isin(PREDICTABLE_TICKERS)]
    
    # Create features with the filtered data
    featured_data = create_features(filtered_data)
    
    # Generate predictions
    predictions = {}
    for ticker in PREDICTABLE_TICKERS:
        try:
            predictions[ticker] = predict_multi_horizon(ticker, featured_data, SEQ_LENGTH, MODEL_DIR)
        except Exception as e:
            print(f"Error generating prediction for {ticker}: {e}")
    
    # Display predictions
    print("\n=== Updated Multi-Horizon Predictions ===")
    print("{:<8} {:<8} {:<8} {:<15}".format("Ticker", "Next Day", "Next 5 Days", "Bullish Signals"))
    print("-" * 40)
    for ticker, prediction in predictions.items():
        bullish_ratio = f"{prediction.get('bullish_signals', 0)}/{prediction.get('total_signals', 0)}"
        print("{:<8} {:<8} {:<8} {:<15}".format(
            ticker, 
            prediction.get('next_day', 'N/A'), 
            prediction.get('next_5day', 'N/A'),
            bullish_ratio
        ))
    
    # Trading strategy suggestion
    print("\n=== Updated Trading Strategy Recommendation ===")
    for ticker, prediction in predictions.items():
        next_day = prediction.get('next_day', 'N/A')
        next_5day = prediction.get('next_5day', 'N/A')
        
        if next_day == 'Up' and next_5day == 'Up':
            strategy = "STRONG BUY: Positive outlook for both short and medium term"
        elif next_day == 'Up' and next_5day == 'Down':
            strategy = "SHORT-TERM BUY: Consider buying for day trading, but be cautious about holding"
        elif next_day == 'Down' and next_5day == 'Up':
            strategy = "WATCH: Consider buying on the dip for medium-term gains"
        elif next_day == 'Down' and next_5day == 'Down':
            strategy = "AVOID/SELL: Negative outlook for both short and medium term"
        else:
            strategy = "INSUFFICIENT DATA"
        
        print(f"{ticker.upper()}: {strategy}")
    
    # Save predictions to CSV
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    prediction_log = []
    for ticker, prediction in predictions.items():
        prediction_log.append({
            'date': today,
            'ticker': ticker,
            'next_day': prediction.get('next_day', 'N/A'),
            'next_5day': prediction.get('next_5day', 'N/A'),
            'bullish_signals': prediction.get('bullish_signals', 0),
            'total_signals': prediction.get('total_signals', 0)
        })
    
    # Create or update prediction log
    log_file = 'results/prediction_log.csv'
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        updated_df = pd.concat([log_df, pd.DataFrame(prediction_log)], ignore_index=True)
    else:
        updated_df = pd.DataFrame(prediction_log)
    
    updated_df.to_csv(log_file, index=False)
    print(f"\nPredictions logged to {log_file}")
    
    return predictions

def daily_update(update_dir=None, skip_retrain=False, clear_update_files=True):
    """
    Perform a complete daily update cycle
    
    Args:
        update_dir: Directory containing update files (default: NEW_DATA_DIR)
        skip_retrain: If True, skip retraining models
        clear_update_files: If True, remove update files after processing
        
    Returns:
        Dictionary with predictions
    """
    # Step 1: Check for dataset updates
    updated_data = update_dataset(update_dir)
    
    # Check if there were any actual updates
    data_was_updated = len(glob.glob(os.path.join(update_dir or NEW_DATA_DIR, "*.json"))) > 0 or \
                       len(glob.glob(os.path.join(update_dir or NEW_DATA_DIR, "*.csv"))) > 0
    
    # Step 2: Retrain models if needed and not skipped
    if data_was_updated and not skip_retrain:
        print("\nRetraining models with updated data...")
        retrain_models(updated_data)
    else:
        if skip_retrain:
            print("\nSkipping model retraining, using existing models...")
        else:
            print("\nNo new data found, using existing models without retraining...")
    
    # Step 3: Make predictions
    predictions = make_predictions(updated_data)
    
    # Step 4: Clean up processed files (only if there were files to process)
    if data_was_updated and clear_update_files and update_dir is not None:
        processed_files = glob.glob(os.path.join(update_dir, "*.json"))
        processed_files.extend(glob.glob(os.path.join(update_dir, "*.csv")))
        
        if processed_files:
            # Create a backup directory
            backup_dir = os.path.join(update_dir, "processed", datetime.datetime.now().strftime("%Y%m%d"))
            os.makedirs(backup_dir, exist_ok=True)
            
            # Move files to backup
            for file_path in processed_files:
                import shutil
                target_path = os.path.join(backup_dir, os.path.basename(file_path))
                shutil.move(file_path, target_path)
            
            print(f"\nMoved {len(processed_files)} processed files to {backup_dir}")
    
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update stock prediction models with latest data')
    parser.add_argument('--update-dir', type=str, help='Directory with update files', default=NEW_DATA_DIR)
    parser.add_argument('--skip-retrain', action='store_true', help='Skip model retraining')
    parser.add_argument('--keep-files', action='store_true', help='Keep update files after processing')
    args = parser.parse_args()
    
    daily_update(args.update_dir, args.skip_retrain, not args.keep_files)