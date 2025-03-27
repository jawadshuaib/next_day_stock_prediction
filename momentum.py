import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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
    
    return X, y, features

def train_models(df):
    """
    Train a model for each stock
    """
    models = {}
    scalers = {}
    
    for ticker, group in df.groupby('ticker_name'):
        print(f"Training model for {ticker}...")
        
        # Prepare data for this ticker
        X, y, features = prepare_training_data(group)
        
        # Skip if insufficient data
        if len(X) < 30:  # Require at least 30 data points
            print(f"Skipping {ticker} - insufficient data")
            continue
        
        # Make sure there are no infinity or NaN values before scaling
        if np.isnan(X).any() or np.isinf(X).any():
            print(f"Skipping {ticker} - data contains NaN or infinity values")
            continue
            
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train a Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5, 
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Save model and scaler
        models[ticker] = model
        scalers[ticker] = scaler
        
        # Calculate accuracy on training set
        train_accuracy = model.score(X_scaled, y)
        print(f"  Training accuracy: {train_accuracy:.4f}")
        
        # Print feature importance
        importances = dict(zip(features, model.feature_importances_))
        print(f"  Top features: {sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    return models, scalers

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
    
    # Train models for each ticker
    models, scalers = train_models(data)
    
    # Save models and scalers for future use
    os.makedirs("models", exist_ok=True)
    with open("models/momentum_models.pkl", "wb") as f:
        pickle.dump(models, f)
    with open("models/momentum_scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    
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