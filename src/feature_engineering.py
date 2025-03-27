import pandas as pd
import numpy as np

def compute_rsi(prices, window=14):
    delta = prices.diff().fillna(0)
    gains = delta.clip(lower=0)
    losses = -1 * delta.clip(upper=0)
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()
    avg_loss = avg_loss.replace(0, 1e-10)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_moving_average(prices, window=5):
    return prices.rolling(window).mean()

def compute_bollinger_bands(prices, window=20, num_std_dev=2):
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def add_features(data):
    data['RSI'] = compute_rsi(data['close'])
    data['MA_5'] = compute_moving_average(data['close'], window=5)
    data['Upper_Band'], data['Lower_Band'] = compute_bollinger_bands(data['close'])
    data['lagged_close_1'] = data['close'].shift(1)
    data['lagged_close_2'] = data['close'].shift(2)
    data['lagged_close_3'] = data['close'].shift(3)
    data['lagged_close_4'] = data['close'].shift(4)
    data['lagged_close_5'] = data['close'].shift(5)
    return data.dropna()