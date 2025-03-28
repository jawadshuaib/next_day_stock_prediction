
"""
--
-- Make Predictions for Next Day and Next 5 Days
--
This script generates multi-horizon stock predictions (next day and next 5 days) for a list of predefined tickers.
It uses trained models and feature engineering to analyze stock data and provide predictions along with trading strategy recommendations.
Functions:
- add_5day_horizon(data):
    Adds a 5-day prediction horizon to the dataset by calculating future returns and direction.
- load_model(ticker, input_size, model_dir='model'):
    Loads a trained PyTorch model for a specific ticker from the specified directory.
- predict_multi_horizon(ticker, data, seq_length, model_dir='model'):
    Generates predictions for the next day and next 5 days for a given ticker using the trained model and heuristic analysis.
Main Execution:
- Loads stock data from a pickle file.
- Ensures the required tickers are present in the dataset.
- Applies feature engineering using the original feature engineering function.
- Generates predictions for each ticker in the predefined list.
- Displays predictions and provides trading strategy recommendations based on the predictions.
Constants:
- PREDICTABLE_TICKERS: List of tickers for which predictions are generated.
- source_file: Path to the stock data file.
- model_dir: Directory containing the trained models.
- seq_length: Sequence length used during training.
Usage:
Run the script directly to generate predictions and trading strategy recommendations for the predefined tickers.
Ensure the required data file and trained models are available in the specified paths.
"""
# Add at the top of the file (after the imports)
# Set fixed random seeds for reproducibility
import random
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import os
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Import the original feature engineering function to ensure compatibility with trained models
from step2_find_most_predictable_stocks import TimeSeriesPredictor, create_features, prepare_data_sequences

# Define the tickers for prediction
PREDICTABLE_TICKERS = ['fccl', 'mlcf']

# File paths
source_file = 'data/daily_stocks.pkl'
model_dir = 'model'
seq_length = 5  # Sequence length used during training

# Ensure the model directory exists
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory '{model_dir}' does not exist. Train the models first.")

# Function to add 5-day horizon targets to existing features
def add_5day_horizon(data):
    """Add 5-day prediction horizon to data that already has features computed"""
    df = data.copy()
    
    # Calculate 5-day horizon
    future_price = df['close'].shift(-5)
    df['next_5day_return'] = (future_price - df['close']) / df['close']
    df['next_5day_direction'] = (df['next_5day_return'] > 0).astype(int)
    
    return df

# Function to load the trained model
def load_model(ticker, input_size, model_dir='model'):
    model_path = os.path.join(model_dir, f'best_{ticker}_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file for ticker '{ticker}' not found at '{model_path}'.")
    
    # Ensure the model architecture matches the one used during training
    model = TimeSeriesPredictor(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to make multi-horizon predictions
def predict_multi_horizon(ticker, data, seq_length, model_dir='model'):
    print(f"\nGenerating predictions for {ticker}...")
    
    # Filter data for the specific ticker
    ticker_data = data[data['ticker_name'] == ticker].copy()
    
    # Ensure there is enough data for sequence preparation
    if len(ticker_data) < seq_length + 5:  # Need 5 more days for 5-day prediction
        raise ValueError(f"Not enough data for ticker '{ticker}' to generate a prediction. "
                         f"At least {seq_length + 5} rows are required.")
    
    # Prepare the data for next-day prediction using original functions
    X, _, _ = prepare_data_sequences(ticker_data, seq_length)
    
    # Load the trained model
    input_size = X.shape[2]
    model = load_model(ticker, input_size, model_dir)
    
    # Use the last sequence for next-day prediction
    last_sequence = X[-1].unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        # For compatibility with the existing model, which returns (reg_out, cls_out)
        _, cls_out = model(last_sequence)
        next_day_pred = torch.argmax(cls_out, dim=1).item()
    
    # For 5-day prediction, analyze recent patterns and momentum
    # This is a simple heuristic approach since we don't have a trained 5-day model
    
    # Method 1: Trend analysis
    recent_returns = ticker_data['returns'].iloc[-10:].mean()
    recent_ma_trend = (ticker_data['ma_5'].iloc[-1] > ticker_data['ma_10'].iloc[-1])
    
    # Method 2: RSI analysis
    rsi_value = ticker_data['rsi'].iloc[-1]
    rsi_trend = ticker_data['rsi'].iloc[-5:].mean() > ticker_data['rsi'].iloc[-10:-5].mean()
    
    # Method 3: MACD analysis
    macd_signal = ticker_data['macd'].iloc[-1] > ticker_data['macd_signal'].iloc[-1]
    macd_trend = ticker_data['macd_hist'].iloc[-5:].mean() > 0
    
    # Combine signals (simple voting mechanism)
    signals = [
        recent_returns > 0,  # Positive recent returns
        recent_ma_trend,     # 5-day MA above 10-day MA
        rsi_trend,           # Rising RSI
        macd_signal,         # MACD above signal line
        macd_trend           # Positive MACD histogram
    ]
    
    # Count bullish signals
    bullish_count = sum(1 for signal in signals if signal)
    next_5day_pred = 1 if bullish_count >= 3 else 0  # Majority vote
    
    # Map predictions to directions
    next_day_direction = "Up" if next_day_pred == 1 else "Down"
    next_5day_direction = "Up" if next_5day_pred == 1 else "Down"
    
    print(f"Prediction for {ticker}:")
    print(f"  Next day: {next_day_direction}")
    print(f"  Next 5 days: {next_5day_direction}")
    
    return {
        'next_day': next_day_direction,
        'next_5day': next_5day_direction,
        'bullish_signals': bullish_count,
        'total_signals': len(signals)
    }

# Main execution
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data = pd.read_pickle(source_file)
    
    # Ensure the data contains the required tickers
    available_tickers = data['ticker_name'].unique()
    missing_tickers = [ticker for ticker in PREDICTABLE_TICKERS if ticker not in available_tickers]
    if missing_tickers:
        raise ValueError(f"The following tickers are missing from the data: {missing_tickers}")
    
    # Filter data for the selected tickers
    data = data[data['ticker_name'].isin(PREDICTABLE_TICKERS)]
    
    # Feature engineering using the original function for compatibility
    print("Engineering features...")
    featured_data = create_features(data)
    
    # Generate predictions for all selected tickers
    predictions = {}
    for ticker in PREDICTABLE_TICKERS:
        try:
            predictions[ticker] = predict_multi_horizon(ticker, featured_data, seq_length, model_dir)
        except Exception as e:
            print(f"Error generating prediction for {ticker}: {e}")
    
    # Display predictions
    print("\n=== Multi-Horizon Predictions ===")
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
    print("\n=== Trading Strategy Recommendation ===")
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