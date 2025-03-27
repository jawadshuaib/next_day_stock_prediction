import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.feature_engineering import compute_rsi

class StockPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(StockPricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(data, epochs=100, batch_size=32):
    features = data[['open', 'high', 'low', 'close', 'volume', 'RSI', 'lagged_return_1', 'lagged_return_2']].values
    labels = data['close'].shift(-1).dropna().values
    features = features[:-1]  # Align features with labels

    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = StockPricePredictor(input_size=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = torch.tensor(X_train[i:i + batch_size], dtype=torch.float32)
            y_batch = torch.tensor(y_train[i:i + batch_size], dtype=torch.float32).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = model.loss_function(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    validate_model(model, X_val, y_val)
    return model

def validate_model(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        val_outputs = model(X_val_tensor)
        val_loss = model.loss_function(val_outputs, y_val_tensor)
        print(f'Validation Loss: {val_loss.item():.4f}')

def save_model(model, path='model/saved_model.pth'):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

if __name__ == "__main__":
    data = pd.read_pickle("../../data/processed/train.pkl")
    data["RSI"] = compute_rsi(data["close"])
    data["lagged_return_1"] = data["returns"].shift(1).fillna(0)
    data["lagged_return_2"] = data["returns"].shift(2).fillna(0)
    
    train_model(data)
    save_model(model)