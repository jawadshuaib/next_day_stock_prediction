
# This script implements a momentum-based stock prediction model using machine learning.
# It calculates momentum features, trains models for each stock, performs cross-validation,
# backtests the strategy, and predicts the next day's stock movement.
# Modules and Functions:
# ----------------------
# 1. calculate_momentum_features(df):
#   - Calculates momentum-based features for each stock, such as rate of change (ROC),
#     moving averages, volume changes, and target labels for prediction.
# 2. prepare_training_data(df, window_size=5):
#   - Prepares training data by creating sequences of features and targets for a given window size.
# 3. cross_validate_model(X, y, n_splits=5):
#   - Performs time-series cross-validation to evaluate model performance using metrics like
#     accuracy, precision, recall, and F1 score.
# 4. backtest_strategy(df, features, model, scaler):
#   - Backtests the trained model on historical data to calculate performance metrics and
#     potential returns compared to a buy-and-hold strategy.
# 5. train_models(df):
#   - Trains a machine learning model for each stock, performs cross-validation, and backtests
#     the strategy. Saves the trained models, scalers, and performance metrics.
# 6. predict_next_day(df, models, scalers):
#   - Predicts the next day's stock movement for each stock using the trained models and scalers.
# Main Execution:
# ---------------
# - Loads stock data from a pickle file.
# - Calculates momentum features for the dataset.
# - Trains models for each stock with cross-validation and backtesting.
# - Saves trained models, scalers, and performance metrics to disk.
# - Predicts the next day's stock movement and saves predictions to a file.
# - Displays a summary of the latest predictions and model performance.
# ------
# - Ensure the stock data is available in the specified file path.
# - Run the script to train models, backtest strategies, and generate predictions.
# - Review the saved predictions and performance metrics for insights.
# Note:
# -----
# - The script assumes the input data contains columns like 'ticker_name', 'date', 'close', and 'volume'.
# - Missing or invalid data is handled by dropping rows with NaN or infinity values.
# - The backtesting strategy uses a rolling window approach for training and testing.
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_momentum_features(df):
    """
    Calculate various momentum features for each stock
    """
    df = df.copy().sort_values(['ticker_name', 'date'])
    
    # Group by ticker to calculate features within each stock
    result_dfs = []
    
    for ticker, group in df.groupby('ticker_name'):
        group = group.sort_values('date')
        
        # Rate of Change (ROC) for different lookback periods
        group['roc_1'] = group['close'].pct_change(1)
        group['roc_2'] = group['close'].pct_change(2)
        group['roc_3'] = group['close'].pct_change(3)
        group['roc_5'] = group['close'].pct_change(5)
        
        # Moving averages
        group['ma_5'] = group['close'].rolling(window=5).mean()
        
        # Price relative to moving average
        group['close_vs_ma5'] = group['close'] / group['ma_5'] - 1
        
        # Volume changes
        if 'volume' in group.columns:
            group['vol_change'] = group['volume'].pct_change(1)
            group['vol_ma5'] = group['volume'].rolling(window=5).mean()
            group['vol_ratio'] = group['volume'] / group['vol_ma5']
        
        # Create target: whether next day's close is higher than current close
        group['next_day_close'] = group['close'].shift(-1)
        group['target'] = (group['next_day_close'] > group['close']).astype(int)
        
        # Replace infinity values with NaN
        group.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        result_dfs.append(group)
    
    return pd.concat(result_dfs, ignore_index=True)

def prepare_training_data(df, window_size=5):
    """
    Prepare training data by creating sequences of 'window_size' days
    """
    features = ['roc_1', 'roc_2', 'roc_3', 'roc_5', 'close_vs_ma5']
    
    if 'vol_change' in df.columns:
        features.extend(['vol_change', 'vol_ratio'])
    
    # Replace any infinity values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with NaN (typically the first 5 rows due to rolling calculations)
    df = df.dropna(subset=features + ['target'])
    
    X = df[features].values
    y = df['target'].values
    
    return X, y, features, df

def cross_validate_model(X, y, n_splits=5):
    """
    Perform time series cross-validation and return performance metrics
    """
    cv_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Time series split ensures that training data comes before test data
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        cv_scores['accuracy'].append(accuracy_score(y_test, y_pred))
        cv_scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        cv_scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        cv_scores['f1'].append(f1_score(y_test, y_pred, zero_division=0))
    
    # Calculate average scores
    avg_scores = {key: np.mean(values) for key, values in cv_scores.items()}
    
    return avg_scores

def backtest_strategy(df, features, model, scaler):
    """
    Backtest the model on historical data and calculate returns
    """
    backtest_df = df.copy()
    backtest_df['predicted_target'] = None
    backtest_df['correct_prediction'] = None
    
    # Use a rolling window approach for backtesting
    window_size = 60  # Initial training window size
    
    if len(df) <= window_size:
        print("Not enough data for backtesting")
        return None
    
    # Sort by date for proper backtesting
    backtest_df = backtest_df.sort_values('date').reset_index(drop=True)
    
    # Store predictions and actual outcomes
    all_predictions = []
    all_actuals = []
    
    for i in range(window_size, len(backtest_df) - 1):  # -1 to ensure we have tomorrow's price
        # Train on window_size days
        train_data = backtest_df.iloc[i-window_size:i]
        test_point = backtest_df.iloc[i:i+1]
        
        # Skip if we have NaN values
        if train_data[features + ['target']].isna().any().any() or test_point[features].isna().any().any():
            continue
        
        # Prepare training data
        X_train = train_data[features].values
        y_train = train_data['target'].values
        
        # Skip if we have infinity values
        if np.isinf(X_train).any() or np.isnan(X_train).any():
            continue
            
        # Scale and train
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        
        # Prepare test data and predict
        X_test = test_point[features].values
        X_test_scaled = scaler.transform(X_test)
        prediction = model.predict(X_test_scaled)[0]
        
        # Store prediction and actual outcome
        backtest_df.loc[backtest_df.index[i], 'predicted_target'] = prediction
        actual = backtest_df.loc[backtest_df.index[i], 'target']
        backtest_df.loc[backtest_df.index[i], 'correct_prediction'] = int(prediction == actual)
        
        all_predictions.append(prediction)
        all_actuals.append(actual)
    
    # Calculate performance metrics if we have any predictions
    if all_predictions:
        metrics = {
            'accuracy': accuracy_score(all_actuals, all_predictions),
            'precision': precision_score(all_actuals, all_predictions, zero_division=0),
            'recall': recall_score(all_actuals, all_predictions, zero_division=0),
            'f1': f1_score(all_actuals, all_predictions, zero_division=0),
            'win_rate': backtest_df['correct_prediction'].mean()
        }
        
        # Calculate potential returns (simplified)
        backtest_df['pct_change'] = backtest_df['close'].pct_change()
        backtest_df['strategy_return'] = None
        
        # For each prediction, calculate the return if we followed it
        for i in range(window_size, len(backtest_df) - 1):
            if backtest_df.loc[backtest_df.index[i], 'predicted_target'] is not None:
                # If predicted up, we go long, otherwise short
                direction = 1 if backtest_df.loc[backtest_df.index[i], 'predicted_target'] == 1 else -1
                next_return = backtest_df.loc[backtest_df.index[i+1], 'pct_change']
                if not np.isnan(next_return):
                    backtest_df.loc[backtest_df.index[i], 'strategy_return'] = direction * next_return
        
        # Calculate cumulative returns
        metrics['cumulative_return'] = backtest_df['strategy_return'].sum()
        # Buy and hold comparison
        metrics['buy_and_hold_return'] = backtest_df['pct_change'].sum()
        
        return metrics, backtest_df
    else:
        return None, backtest_df

def train_models(df):
    """
    Train a model for each stock with cross-validation
    """
    models = {}
    scalers = {}
    performance = {}
    backtest_results = {}
    
    for ticker, group in df.groupby('ticker_name'):
        print(f"Training model for {ticker}...")
        
        # Prepare data for this ticker
        X, y, features, ticker_df = prepare_training_data(group)
        
        # Skip if insufficient data
        if len(X) < 60:  # Need at least 60 days for proper validation
            print(f"Skipping {ticker} - insufficient data")
            continue
        
        # Make sure there are no infinity or NaN values before scaling
        if np.isnan(X).any() or np.isinf(X).any():
            print(f"Skipping {ticker} - data contains NaN or infinity values")
            continue
        
        # Perform cross-validation
        print("  Performing time series cross-validation...")
        cv_scores = cross_validate_model(X, y, n_splits=5)
        print(f"  CV Scores: Accuracy: {cv_scores['accuracy']:.4f}, Precision: {cv_scores['precision']:.4f}, Recall: {cv_scores['recall']:.4f}, F1: {cv_scores['f1']:.4f}")
        performance[ticker] = cv_scores
        
        # Train final model on all data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_scaled, y)
        
        # Run backtesting
        print("  Running backtest...")
        backtest_metrics, _ = backtest_strategy(ticker_df, features, 
            RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42), StandardScaler())
        
        if backtest_metrics:
            print(f"  Backtest: Win Rate: {backtest_metrics['win_rate']:.4f}, Return: {backtest_metrics['cumulative_return']:.4f}, Buy&Hold: {backtest_metrics['buy_and_hold_return']:.4f}")
            backtest_results[ticker] = backtest_metrics
        
        # Save model and scaler
        models[ticker] = model
        scalers[ticker] = scaler
        
        # Print feature importance
        importances = dict(zip(features, model.feature_importances_))
        print(f"  Top features: {sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    return models, scalers, performance, backtest_results

def predict_next_day(df, models, scalers):
    """
    Predict the next day's movement for each stock
    """
    df = df.copy()
    df['prediction'] = None
    df['prediction_confidence'] = None
    df['prediction_date'] = None
    
    for ticker, group in df.groupby('ticker_name'):
        if ticker not in models:
            print(f"No model available for {ticker}")
            continue
            
        # Get the model and scaler
        model = models[ticker]
        scaler = scalers[ticker]
        
        # Get the features for the latest available data point
        group = group.sort_values('date')
        latest_data = group.iloc[-1:].copy()
        
        # Extract features
        features = ['roc_1', 'roc_2', 'roc_3', 'roc_5', 'close_vs_ma5']
        if 'vol_change' in group.columns:
            features.extend(['vol_change', 'vol_ratio'])
        
        # Replace infinity values with NaN
        latest_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Skip if we're missing any features
        if latest_data[features].isna().any(axis=1).iloc[0]:
            print(f"Missing data for {ticker} prediction")
            continue
            
        # Scale features
        X = scaler.transform(latest_data[features].values)
        
        # Make prediction
        pred_proba = model.predict_proba(X)[0]
        pred_class = model.predict(X)[0]
        confidence = pred_proba[pred_class]
        
        # Get the date for which we're making a prediction
        last_date = latest_data['date'].iloc[0]
        next_date = last_date + timedelta(days=1)
        
        # Store prediction
        prediction = 'Up' if pred_class == 1 else 'Down'
        idx = latest_data.index[0]
        df.loc[idx, 'prediction'] = prediction
        df.loc[idx, 'prediction_confidence'] = confidence
        df.loc[idx, 'prediction_date'] = next_date
        
        print(f"{ticker}: Predict {prediction} for {next_date.strftime('%Y-%m-%d')} with {confidence:.2f} confidence")
    
    return df

if __name__ == "__main__":
    data_file = "data/daily_stocks.pkl"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found at {data_file}")

    # Load consolidated stock data
    data = pd.read_pickle(data_file)

    # Ensure we have a 'close' column
    if "close" not in data.columns:
        raise ValueError("No 'close' column found in the data.")

    # Calculate momentum features
    data = calculate_momentum_features(data)
    
    # Train models with cross-validation and backtesting
    models, scalers, validation_metrics, backtest_results = train_models(data)
    
    # Save models, scalers and performance metrics
    os.makedirs("models", exist_ok=True)
    with open("models/momentum_models.pkl", "wb") as f:
        pickle.dump(models, f)
    with open("models/momentum_scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    with open("models/validation_metrics.pkl", "wb") as f:
        pickle.dump(validation_metrics, f)
    with open("models/backtest_results.pkl", "wb") as f:
        pickle.dump(backtest_results, f)
    
    # Predict the next day's movement
    predictions = predict_next_day(data, models, scalers)
    
    # Save predictions
    predictions.to_pickle("data/momentum_predictions.pkl")
    print("\nMomentum predictions saved to data/momentum_predictions.pkl")
    
    # Display summary of latest predictions
    latest_predictions = predictions.sort_values('date').groupby('ticker_name').tail(1)
    
    print("\nLatest predictions for each ticker:")
    for _, row in latest_predictions.iterrows():
        if row['prediction'] is not None:
            print(f"{row['ticker_name']}: {row['prediction']} for {row['prediction_date'].strftime('%Y-%m-%d')} (confidence: {row['prediction_confidence']:.2f})")
    
    # Display validation and backtest summary
    print("\nModel Performance Summary:")
    print("==========================")
    print("{:<10} {:<12} {:<12} {:<12} {:<12} {:<12}".format("Ticker", "CV Accuracy", "Win Rate", "Strategy %", "Buy&Hold %", "Confidence"))
    for ticker in models.keys():
        if ticker in validation_metrics and ticker in backtest_results:
            cv_acc = validation_metrics[ticker]['accuracy']
            winrate = backtest_results[ticker]['win_rate']
            strat_ret = backtest_results[ticker]['cumulative_return']
            bh_ret = backtest_results[ticker]['buy_and_hold_return']
            
            # Find the corresponding confidence for latest prediction
            confidence = None
            for _, row in latest_predictions.iterrows():
                if row['ticker_name'] == ticker and row['prediction'] is not None:
                    confidence = row['prediction_confidence']
                    break
                
            conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
            
            print("{:<10} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12}".format(
                ticker, cv_acc, winrate, strat_ret, bh_ret, conf_str))