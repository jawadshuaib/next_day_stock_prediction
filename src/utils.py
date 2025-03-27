def calculate_rsi(prices, window=14):
    delta = prices.diff().fillna(0)
    gains = delta.clip(lower=0)
    losses = -1 * delta.clip(upper=0)
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()
    avg_loss = avg_loss.replace(0, 1e-10)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_vwap(data):
    cumulative_price_volume = (data['close'] * data['volume']).cumsum()
    cumulative_volume = data['volume'].cumsum()
    return cumulative_price_volume / cumulative_volume

def create_lagged_features(data, n_lags=5):
    for lag in range(1, n_lags + 1):
        data[f'lagged_close_{lag}'] = data['close'].shift(lag)
    return data

def preprocess_data(data):
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data['RSI'] = calculate_rsi(data['close'])
    data['VWAP'] = calculate_vwap(data)
    data = create_lagged_features(data)
    return data.dropna()