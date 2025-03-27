"""
--
-- Find the Most Predictable Stocks --
--
This script performs the following tasks:
1. **Data Loading and Preprocessing**:
    - Loads stock market data from a specified source file.
    - Preprocesses the data to ensure it is clean and ready for feature engineering.
2. **Feature Engineering**:
    - Creates a variety of features for stock prediction, including:
      - Return-based features (e.g., returns, log returns, volatility).
      - Technical indicators (e.g., RSI, moving averages, MACD, Bollinger Bands).
      - Volume indicators (e.g., volume change, relative volume).
      - Price patterns (e.g., high-low difference, close-open difference).
      - Lagged features to avoid future data leakage.
    - Ensures all features are properly normalized and handles missing values.
3. **Model Definition**:
    - Defines a `TimeSeriesPredictor` class, which is a PyTorch-based RNN model.
    - Supports both LSTM and GRU architectures with dual output heads for regression and classification tasks.
4. **Data Preparation**:
    - Prepares data sequences for time series modeling.
    - Scales features using MinMaxScaler and creates input-output pairs for training.
5. **Cross-Validation**:
    - Implements time series cross-validation using `TimeSeriesSplit`.
    - Trains and evaluates the model on multiple folds to ensure robustness.
    - Handles imbalanced classes by applying class weighting in the loss function.
6. **Model Training and Evaluation**:
    - Trains the model using a combination of regression and classification losses.
    - Implements early stopping to prevent overfitting.
    - Evaluates the model on test data using metrics such as accuracy, precision, recall, and F1 score.
7. **Ticker Analysis**:
    - Processes each stock ticker individually.
    - Skips tickers with insufficient data or suspicious patterns (e.g., stale data).
    - Identifies the most predictable tickers based on cross-validation results.
8. **Results Storage**:
    - Aggregates results across all tickers.
    - Filters out suspicious results and identifies the most predictable tickers.
    - Saves the results to a CSV file for further analysis.
**Main Functions**:
- `create_features(data)`: Generates features for stock prediction while avoiding future data leakage.
- `prepare_data_sequences(data, seq_length)`: Prepares normalized data sequences for time series modeling.
- `cross_validate_ticker(ticker_data, ticker_name, n_splits, seq_length)`: Performs cross-validation for a specific stock ticker.
- `TimeSeriesPredictor`: PyTorch-based RNN model for time series prediction.
**Execution**:
- The script processes all tickers in the dataset, performs cross-validation, and identifies the most predictable stocks.
- Results are saved to a CSV file for further analysis.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import TimeSeriesSplit
import os
import warnings
warnings.filterwarnings('ignore')

from src.data_processing import load_data, preprocess_data
from src.feature_engineering import compute_rsi, compute_moving_average, compute_bollinger_bands

source_file = 'data/daily_stocks.pkl'

# LSTM model for time series prediction
class TimeSeriesPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.5, use_gru=False):
        super(TimeSeriesPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Choose between LSTM and GRU
        if use_gru:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                             num_layers=num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Dual output heads
        self.fc_regression = nn.Linear(hidden_size, 1)
        self.fc_classification = nn.Linear(hidden_size, 2)
        
    def forward(self, x, output_type='both'):
        if isinstance(self.rnn, nn.LSTM):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, h0)
        
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        last_out = self.batch_norm(last_out)
        
        if output_type == 'regression':
            return self.fc_regression(last_out)
        elif output_type == 'classification':
            return self.fc_classification(last_out)
        else:
            return self.fc_regression(last_out), self.fc_classification(last_out)

# Feature engineering that avoids future data leakage
def create_features(data):
    df = data.copy().sort_values('date')  # Ensure chronological order
    
    # 1. Return-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['returns_volatility'] = df['returns'].rolling(window=5).std()
    
    # 2. Technical indicators
    df['rsi'] = compute_rsi(df['close'], window=14)
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'close_to_ma_{window}'] = (df['close'] - df[f'ma_{window}']) / df[f'ma_{window}']
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_upper'], df['bb_lower'] = compute_bollinger_bands(df['close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # 3. Volume indicators
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['rel_volume'] = df['volume'] / df['volume_ma_5']
    
    # 4. Price patterns
    df['high_low_diff'] = (df['high'] - df['low']) / df['low']
    df['close_open_diff'] = (df['close'] - df['open']) / df['open']
    
    # 5. Lagged features (use only past data)
    for i in range(1, 5):
        df[f'return_lag_{i}'] = df['returns'].shift(i)
    
    # 6. Target variables (carefully separate to avoid leakage)
    df['next_return'] = df['returns'].shift(-1)  # Next day's return
    df['next_direction'] = (df['next_return'] > 0).astype(int)
    
    # Fill NaNs
    for col in df.columns:
        if col not in ['ticker', 'date', 'ticker_name', 'next_return', 'next_direction']:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    return df.dropna()

# Prepare sequences with proper data normalization
def prepare_data_sequences(data, seq_length=5):
    data = data.copy()
    
    # Features to use (excluding targets, dates, and identifiers)
    feature_cols = [col for col in data.columns if col not in 
                   ['date', 'ticker', 'ticker_name', 'next_return', 'next_direction']]
    
    # Ensure all columns are numeric
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Handle invalid values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=feature_cols, inplace=True)
    
    if data.empty:
        raise ValueError("Dataset is empty after cleaning.")
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data[feature_cols])
    
    X, y_returns, y_directions = [], [], []
    
    # Create sequences
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y_returns.append(data['next_return'].iloc[i+seq_length])
        y_directions.append(data['next_direction'].iloc[i+seq_length])
    
    # Convert to tensors
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y_returns = torch.tensor(np.array(y_returns), dtype=torch.float32).view(-1, 1)
    y_directions = torch.tensor(np.array(y_directions), dtype=torch.long)
    
    return X, (y_returns, y_directions), scaler

# Time series cross-validation and training
def cross_validate_ticker(ticker_data, ticker_name, n_splits=5, seq_length=5):
    # Sort by date to ensure proper time-series split
    ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
    
    # Use TimeSeriesSplit for proper time series validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Store results for each fold
    fold_results = []
    
    # Track the best model across all folds
    best_overall_model = None
    best_overall_acc = 0
    best_overall_fold = -1
    input_size = None
    
    # Examine class distribution
    class_distribution = ticker_data['next_direction'].value_counts(normalize=True)
    print(f"Class distribution: {class_distribution.to_dict()}")
    
    # Check if heavily imbalanced
    is_imbalanced = abs(class_distribution.get(1, 0) - 0.5) > 0.3
    if is_imbalanced:
        print(f"Warning: Highly imbalanced classes for {ticker_name}.")
    
    fold = 0
    for train_idx, test_idx in tscv.split(ticker_data):
        fold += 1
        print(f"Fold {fold}/{n_splits}")
        
        # Further split train into train/val
        train_size = int(len(train_idx) * 0.8)
        train_data = ticker_data.iloc[train_idx[:train_size]]
        val_data = ticker_data.iloc[train_idx[train_size:]]
        test_data = ticker_data.iloc[test_idx]
        
        # Skip folds with insufficient data
        min_required = seq_length + 10  # Need at least sequence length + some samples
        if len(train_data) < min_required or len(val_data) < min_required or len(test_data) < min_required:
            print(f"Insufficient data in fold {fold}. Skipping.")
            continue
        
        try:
            # Prepare data
            X_train, y_train, train_scaler = prepare_data_sequences(train_data, seq_length)
            X_val, y_val, _ = prepare_data_sequences(val_data, seq_length)
            X_test, y_test, _ = prepare_data_sequences(test_data, seq_length)
            
            # Store input_size for model creation
            input_size = X_train.shape[2]
            
            # Check class balance in training set
            up_ratio = y_train[1].float().mean().item()
            print(f"Up direction ratio in training: {up_ratio:.4f}")
            
            # Initialize model with parameters matching the prediction script
            model = TimeSeriesPredictor(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.5)
            
            # Loss functions with class weighting if needed
            regression_loss = nn.MSELoss()
            if up_ratio == 0.0 or up_ratio == 1.0:
                print("Warning: Only one class present. Skipping fold.")
                continue
                
            elif up_ratio > 0.6 or up_ratio < 0.4:
                # Cap weights to avoid extreme values
                up_ratio = min(max(up_ratio, 0.01), 0.99)
                weight = torch.tensor([1/(1-up_ratio), 1/up_ratio])
                classification_loss = nn.CrossEntropyLoss(weight=weight)
            else:
                classification_loss = nn.CrossEntropyLoss()
            
            # Train the model
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            train_dataset = TensorDataset(X_train, y_train[0], y_train[1])
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            best_val_acc = 0
            best_fold_model = None
            patience = 0
            max_patience = 10
            
            for epoch in range(50):
                # Training
                model.train()
                for batch_X, batch_y_reg, batch_y_cls in train_loader:
                    reg_out, cls_out = model(batch_X)
                    reg_loss = regression_loss(reg_out, batch_y_reg)
                    cls_loss = classification_loss(cls_out, batch_y_cls)
                    combined_loss = 0.2 * reg_loss + 0.8 * cls_loss
                    
                    optimizer.zero_grad()
                    combined_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    _, val_cls_out = model(X_val)
                    val_cls_preds = torch.argmax(val_cls_out, dim=1)
                    val_accuracy = accuracy_score(y_val[1].numpy(), val_cls_preds.numpy())
                    
                    if val_accuracy > best_val_acc:
                        best_val_acc = val_accuracy
                        patience = 0
                        # Save the best model weights for this fold
                        best_fold_model = model.state_dict().copy()
                    else:
                        patience += 1
                
                # Early stopping
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Test evaluation
            model.eval()
            with torch.no_grad():
                _, test_cls_out = model(X_test)
                test_cls_preds = torch.argmax(test_cls_out, dim=1)
                
                # Calculate metrics
                test_accuracy = accuracy_score(y_test[1].numpy(), test_cls_preds.numpy())
                test_precision = precision_score(y_test[1].numpy(), test_cls_preds.numpy(), zero_division=0)
                test_recall = recall_score(y_test[1].numpy(), test_cls_preds.numpy(), zero_division=0)
                test_f1 = f1_score(y_test[1].numpy(), test_cls_preds.numpy(), zero_division=0)
                
                # Class-specific metrics to detect if model only predicts one class
                class_counts = np.bincount(test_cls_preds.numpy())
                if len(class_counts) < 2 or class_counts.min() < 3:
                    print(f"Warning: Model predicts mostly one class. Class counts: {class_counts}")
                
                print(f"Fold {fold} accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
                
                # Save fold results
                fold_results.append({
                    'fold': fold,
                    'accuracy': test_accuracy,
                    'precision': test_precision,
                    'recall': test_recall,
                    'f1': test_f1
                })
                
                # Update best overall model if this fold is better
                if test_accuracy > best_overall_acc and best_fold_model is not None:
                    best_overall_acc = test_accuracy
                    best_overall_model = best_fold_model
                    best_overall_fold = fold
                    
        except Exception as e:
            print(f"Error in fold {fold}: {e}")
    
    # Save the best model across all folds
    if best_overall_model is not None:
        print(f"Best model was from fold {best_overall_fold} with accuracy {best_overall_acc:.4f}")
        
        # Create the model directory if it doesn't exist
        os.makedirs('model', exist_ok=True)
        
        # Save the model
        model_path = os.path.join('model', f'best_{ticker_name}_model.pth')
        torch.save(best_overall_model, model_path)
        print(f"Saved best model for {ticker_name} to {model_path}")
    else:
        print(f"No suitable model found for {ticker_name}")
    
    # Aggregate results across folds
    if fold_results:
        df_results = pd.DataFrame(fold_results)
        avg_accuracy = df_results['accuracy'].mean()
        avg_f1 = df_results['f1'].mean()
        std_accuracy = df_results['accuracy'].std()
        
        print(f"\nCross-validation results for {ticker_name}:")
        print(f"Average accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Average F1 score: {avg_f1:.4f}")
        
        return {
            'ticker': ticker_name,
            'accuracy': avg_accuracy,
            'accuracy_std': std_accuracy,
            'f1': avg_f1,
            'n_folds': len(fold_results),
            'data_points': len(ticker_data)
        }
    else:
        print(f"No valid results for {ticker_name}")
        return None

# Main execution
if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    
    print("Loading data...")
    data = load_data(source_file)
    processed_data = preprocess_data(data)
    
    print("Engineering features...")
    featured_data = create_features(processed_data)
    
    # Find tickers with sufficient data
    tickers = featured_data['ticker_name'].unique()
    print(f"Found {len(tickers)} unique tickers")
    
    # Results storage
    ticker_results = []
    
    # Process each ticker
    for i, ticker in enumerate(tickers):
        print(f"\n--- Processing ticker {ticker} ({i+1}/{len(tickers)}) ---")
        ticker_data = featured_data[featured_data['ticker_name'] == ticker].copy()
        
        # Skip tickers with insufficient data
        if len(ticker_data) < 100:
            print(f"Skipping {ticker}: insufficient data ({len(ticker_data)} data points)")
            continue
        
        # Check for suspicious patterns in the data
        price_changes = ticker_data['returns'].abs().describe()
        if price_changes['mean'] < 0.0001 or price_changes['std'] < 0.0001:
            print(f"Warning: {ticker} shows suspiciously low price movement. May be stale data.")
        
        # Run cross-validation
        print(f"Cross-validating model for {ticker}...")
        result = cross_validate_ticker(ticker_data, ticker, n_splits=3)
        if result:
            ticker_results.append(result)
    
    # Summarize results
    if ticker_results:
        results_df = pd.DataFrame(ticker_results)
        
        # Filter out suspicious results
        filtered_df = results_df[
            (results_df['accuracy'] < 0.95) &  # Extremely high accuracy is suspicious
            (results_df['accuracy'] > 0.45)     # Too low is worse than random
        ]
        
        print("\n=== Most Predictable Tickers (by Accuracy) ===")
        sorted_df = filtered_df.sort_values('accuracy', ascending=False)
        print(sorted_df[['ticker', 'accuracy', 'accuracy_std', 'f1', 'data_points']].head(10).to_string(index=False))
        
        # Save results
        sorted_df.to_csv('results/ticker_predictability.csv', index=False)
    else:
        print("No ticker had sufficient data for training.")
    
    print("Analysis complete!")