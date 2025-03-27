import torch
import torch.nn as nn

class StockPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(StockPricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the last time step's output
        return out

def build_model(input_size):
    model = StockPricePredictor(input_size)
    return model

def compile_model(model):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer