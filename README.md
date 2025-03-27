# Stock Price Movement Prediction System

## Purpose

This project implements a sophisticated machine learning system to predict daily stock price movements (up/down) using historical price data, technical indicators, and deep learning models. The system aims to identify the most predictable stocks and generate actionable trading signals based on next-day and 5-day forecasts.

---

## System Architecture

### Data Flow Pipeline

1. **Raw Data Processing** (`step1_process_data.py`)

   - Consolidates JSON stock data files into a unified dataset
   - Handles data cleaning and normalization
   - Creates a standardized dataset for further analysis

2. **Predictability Analysis** (`step2_find_most_predictable_stocks.py`)

   - Analyzes each stock's historical data to determine predictability
   - Uses time-series cross-validation to evaluate model performance
   - Ranks stocks by prediction accuracy and F1 score
   - Trains and saves models for the most predictable stocks

3. **Prediction Generation** (`step3_make_prediction_for_next_day.py`)

   - Uses trained models to predict next-day price movements
   - Implements heuristic methods for 5-day predictions
   - Combines multiple technical signals for robust forecasting
   - Generates trading recommendations based on prediction patterns

4. **Ongoing Maintenance** (`daily_update.py`)
   - Processes new data files as they become available
   - Updates the master dataset with new market information
   - Retrains models when necessary to maintain accuracy
   - Generates fresh predictions and logs results for tracking

---

## Detailed File Breakdown

### `step1_process_data.py`

This script handles the initial data processing pipeline:

- **Data Loading**: Reads individual stock data from JSON files
- **Ticker Extraction**: Identifies stock symbols from filenames
- **Data Transformation**: Converts date strings to datetime objects
- **Data Cleaning**: Removes invalid entries, handles missing values
- **Data Standardization**: Creates a consistent format across all stocks
- **Data Storage**: Saves the processed dataset for further analysis

### `step2_find_most_predictable_stocks.py`

This script identifies which stocks exhibit patterns that can be reliably predicted:

- **Feature Engineering**:

  - Return-based features (returns, log returns, volatility)
  - Technical indicators (RSI, MACD, Bollinger Bands)
  - Moving averages with multiple timeframes (5, 10, 20, 50 days)
  - Volume indicators (volume change, relative volume)
  - Price patterns (high-low difference, close-open difference)
  - Lagged features to avoid future data leakage

- **Model Architecture** (`TimeSeriesPredictor` class):

  - Recurrent Neural Network with configurable LSTM/GRU layers
  - Dual output heads for regression (return prediction) and classification (direction prediction)
  - Dropout regularization (0.5) and batch normalization for improved generalization
  - Configurable hidden layer size (128 neurons) and layer count (2 layers)

- **Training Process**:

  - Time-series cross-validation with multiple folds
  - Weighted loss function to handle class imbalance
  - Early stopping to prevent overfitting
  - Model performance evaluation using accuracy, precision, recall, and F1 score
  - Model archiving and versioning

- **Stock Selection**:
  - Filters stocks with suspicious patterns or insufficient data
  - Ranks stocks by prediction accuracy and F1 score
  - Identifies stocks with consistently high performance across validation folds

### `step3_make_prediction_for_next_day.py`

This script generates actionable predictions using the trained models:

- **Model Loading**: Loads the best-performing model for each target stock
- **Sequence Preparation**: Creates input sequences of the proper length (5 days)
- **Next-Day Prediction**: Uses the trained model to predict price direction
- **5-Day Prediction**: Uses technical indicators to estimate medium-term movement:
  - Trend analysis (recent returns, moving average trends)
  - RSI analysis (current value and trend)
  - MACD analysis (signal line crosses and histogram trends)
- **Signal Aggregation**: Combines multiple technical signals into a cohesive prediction
- **Trading Strategy Generation**: Suggests appropriate actions based on prediction patterns:
  - Strong Buy: Both short and medium-term positive outlook
  - Short-Term Buy: Positive short-term, negative medium-term outlook
  - Watch: Negative short-term, positive medium-term outlook
  - Avoid/Sell: Both short and medium-term negative outlook

### `daily_update.py`

This script handles the day-to-day operation of the prediction system:

- **Data Update Handling**:

  - Processes new JSON/CSV files with recent market data
  - Merges new data with the existing dataset
  - Handles data consistency and deduplication
  - Preserves dataset integrity when no updates are available

- **Model Management**:

  - Conditional retraining based on data updates
  - Model versioning with timestamp-based archiving
  - Performance tracking across retraining cycles

- **Prediction Pipeline**:

  - Generates consistent predictions using the same algorithm as `step3_make_prediction_for_next_day.py`
  - Maintains prediction logs for tracking performance over time
  - Provides clear trading recommendations

- **File Management**:
  - Organizes processed files into dated directories
  - Maintains backup copies of previous model versions
  - Creates necessary directory structure automatically

---

## Machine Learning Implementation Details

### Neural Network Architecture

- **Model Type**: Recurrent Neural Network (RNN)
- **Cell Types**: Configurable between LSTM (default) and GRU
- **Layer Structure**:
  - **Input layer**: Variable size based on feature count
  - **Hidden layers**: 2 layers with 128 neurons each
  - **Output layers**:
    - Regression head: 1 neuron for return prediction
    - Classification head: 2 neurons for up/down prediction
- **Regularization Techniques**:
  - Dropout (0.5) to prevent overfitting
  - Batch normalization to improve training stability
  - L2 regularization via weight decay (1e-5)
  - Gradient clipping (max_norm=1.0) to prevent exploding gradients

### Training Methodology

- **Loss Functions**:
  - Regression: Mean Squared Error (MSE)
  - Classification: Cross-Entropy Loss with class weighting
  - Combined loss: `0.2 * regression_loss + 0.8 * classification_loss`
- **Optimizer**: Adam with learning rate `0.001` and weight decay `1e-5`
- **Batch Size**: 32 samples per batch
- **Sequence Length**: 5 days of historical data for each prediction
- **Early Stopping**: Patience of 10 epochs with validation accuracy monitoring
- **Class Imbalance Handling**: Dynamic class weighting based on training data distribution

### Performance Metrics

- **Primary Metrics**:

  - **Accuracy**: Percentage of correctly predicted movements
  - **F1 Score**: Harmonic mean of precision and recall
  - **Standard Deviation of Accuracy**: Measure of prediction consistency

- **Filtering Criteria**:
  - Extremely high accuracy (>95%) is considered suspicious
  - Very low accuracy (<45%) is considered worse than random
  - High standard deviation indicates inconsistent performance

---

## Performance Analysis

Based on cross-validation results, the most predictable stocks are:

|  Ticker  | Accuracy | Accuracy Std | F1 Score | Data Points |
| :------: | :------: | :----------: | :------: | :---------: |
| **bop**  |  76.54%  |   0.187538   | 0.856274 |    1239     |
| **ubl**  |  71.27%  |   0.073579   | 0.154960 |    1238     |
| **mlcf** |  70.18%  |   0.052871   | 0.815882 |    1239     |

This suggests that **bop** and **mlcf** are highly predictable (high accuracy and F1 score), while **ubl** shows good accuracy but poor F1 score (indicating potential class imbalance issues).

---

## Setup Instructions

1. Clone the repository:

   ```
   git clone git@github.com:jawadshuaib/next_day_stock_prediction.git
   cd next_day_stock_prediction
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Usage Instructions

### Initial Setup

1. Install dependencies from `requirements.txt`
2. Prepare your historical stock data in JSON format
3. Configure the tickers you want to track in `PREDICTABLE_TICKERS`

### Complete Model Training Pipeline

1. **Process Raw Data**:
   `python step1_process_data.py`

Consolidate JSON files into a unified dataset

2. **Find Predictable Stocks**:
   `python step2_find_most_predictable_stocks.py`

Identify which stocks show predictable patterns and train models

3. **Generate Predictions**:
   `python step3_make_prediction_for_next_day.py`

Obtain actionable trading signals

### Daily Operation

For ongoing use, simply run `daily_update.py` with appropriate arguments:

- **Default operation**: Process any new data files, retrain models if needed, and generate predictions

`python daily_update.py`

- **--skip-retrain**: Skip model retraining and use existing models
- **--update-dir PATH**: Specify a custom directory for new data files
- **--keep-files**: Keep the update files in place (don't move to processed folder)

---

## Limitations and Future Improvements

- **Model Architecture**: Explore transformer-based models or hybrid CNN-RNN architectures
- **Feature Engineering**: Incorporate sentiment analysis from news or social media
- **Performance Metrics**: Implement financial metrics like risk-adjusted returns
- **Portfolio Optimization**: Extend predictions to guide portfolio allocation decisions
- **Transfer Learning**: Explore transfer learning between related stocks or sectors

---

## Dependencies

- `numpy==1.21.0`
- `pandas==1.3.0`
- `scikit-learn==0.24.2`
- `torch==1.9.0`
- `matplotlib==3.4.2`
- `seaborn==0.11.1`

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
