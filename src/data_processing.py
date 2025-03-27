import pandas as pd

def load_data(filepath):
    """Load stock data from a pickle file."""
    data = pd.read_pickle(filepath)
    return data

def preprocess_data(data):
    """Preprocess the stock data by cleaning and preparing it for feature extraction."""
    # Drop rows with missing values
    data = data.dropna()
    
    # Reset index to ensure 'date' is a column
    data = data.reset_index()
    
    # Convert 'date' to datetime format
    data['date'] = pd.to_datetime(data['date'])
    
    # Sort data by date
    data = data.sort_values(by='date')
    
    return data

def save_processed_data(data, output_filepath):
    """Save the processed data to a pickle file."""
    data.to_pickle(output_filepath)