"""
US Stock Data Loader
Replaces Chinese stock data loading with US stock data
"""

import pandas as pd
import numpy as np
import os
import pickle
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_us_stock_data(ticker, start_date='2020-01-01', end_date=None):
    """
    Load US stock data using yfinance
    
    Parameters:
    -----------
    ticker : str
        US stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format. If None, uses current date
    
    Returns:
    --------
    pd.DataFrame
        Stock data with columns: Open, High, Low, Close, Volume, Adj Close
    """
    try:
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"Warning: No data found for {ticker}")
            return None
        
        # Rename columns to match Chinese stock data format
        data = data.rename(columns={
            'Open': 'openPrice',
            'High': 'highestPrice', 
            'Low': 'lowestPrice',
            'Close': 'closePrice',
            'Volume': 'turnoverVol',
            'Adj Close': 'adjClose'
        })
        
        # Add additional columns that were in Chinese data
        data['preClosePrice'] = data['closePrice'].shift(1)
        data['pctChg'] = ((data['closePrice'] - data['preClosePrice']) / data['preClosePrice'] * 100).round(2)
        data['turnoverValue'] = data['closePrice'] * data['turnoverVol']
        
        # Add technical indicators
        data['MA5'] = data['closePrice'].rolling(window=5).mean()
        data['MA10'] = data['closePrice'].rolling(window=10).mean()
        data['MA20'] = data['closePrice'].rolling(window=20).mean()
        data['MA50'] = data['closePrice'].rolling(window=50).mean()
        
        # RSI
        delta = data['closePrice'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['closePrice'].ewm(span=12).mean()
        exp2 = data['closePrice'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger Bands
        data['BB_Middle'] = data['closePrice'].rolling(window=20).mean()
        bb_std = data['closePrice'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['closePrice'] - data['BB_Lower']) / data['BB_Width']
        
        # Volume indicators
        data['Volume_MA5'] = data['turnoverVol'].rolling(window=5).mean()
        data['Volume_MA10'] = data['turnoverVol'].rolling(window=10).mean()
        data['Volume_Ratio'] = data['turnoverVol'] / data['Volume_MA10']
        
        # Volatility
        data['Volatility'] = data['pctChg'].rolling(window=20).std()
        
        # Price momentum
        data['Momentum_5'] = data['closePrice'] / data['closePrice'].shift(5) - 1
        data['Momentum_10'] = data['closePrice'] / data['closePrice'].shift(10) - 1
        data['Momentum_20'] = data['closePrice'] / data['closePrice'].shift(20) - 1
        
        # Price change
        data['Price_Change_1'] = data['closePrice'] - data['closePrice'].shift(1)
        data['Price_Change_5'] = data['closePrice'] - data['closePrice'].shift(5)
        data['Price_Change_10'] = data['closePrice'] - data['closePrice'].shift(10)
        
        # Log returns
        data['Log_Return_1'] = np.log(data['closePrice'] / data['closePrice'].shift(1))
        data['Log_Return_5'] = np.log(data['closePrice'] / data['closePrice'].shift(5))
        data['Log_Return_10'] = np.log(data['closePrice'] / data['closePrice'].shift(10))
        
        # Fill NaN values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ Loaded {len(data)} days of data for {ticker}")
        return data
        
    except Exception as e:
        print(f"❌ Error loading data for {ticker}: {str(e)}")
        return None

def load_us_stock_single_score():
    """
    Load US stock single score data (replaces load_us_stock_single_score)
    
    Returns:
    --------
    tuple
        (scores, feature_names)
    """
    # US stock feature names (replacing Chinese multi-factor features)
    us_stock_features = [
        'openPrice', 'highestPrice', 'lowestPrice', 'closePrice', 'preClosePrice',
        'turnoverVol', 'turnoverValue', 'pctChg', 'adjClose',
        'MA5', 'MA10', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
        'Volume_MA5', 'Volume_MA10', 'Volume_Ratio', 'Volatility',
        'Momentum_5', 'Momentum_10', 'Momentum_20',
        'Price_Change_1', 'Price_Change_5', 'Price_Change_10',
        'Log_Return_1', 'Log_Return_5', 'Log_Return_10'
    ]
    
    # Generate mock scores for demonstration
    # In practice, these would be calculated from actual US stock data
    np.random.seed(42)
    scores = np.random.randn(len(us_stock_features))
    
    return scores, us_stock_features

def form_us_sector_types():
    """
    Form US sector types (replaces form_us_sector_types)
    
    Returns:
    --------
    list
        List of lists containing feature names for each sector
    """
    # Quality factors - describing financial health, efficiency, profitability
    quality_factors = [
        'MA5', 'MA10', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
        'Volatility', 'Momentum_5', 'Momentum_10', 'Momentum_20'
    ]
    
    # Return and risk factors - describing returns and risk metrics
    return_risk_factors = [
        'pctChg', 'Log_Return_1', 'Log_Return_5', 'Log_Return_10',
        'Price_Change_1', 'Price_Change_5', 'Price_Change_10',
        'Volatility', 'RSI', 'MACD', 'MACD_Histogram'
    ]
    
    # Valuation factors - describing market value, P/E, P/B ratios
    valuation_factors = [
        'closePrice', 'adjClose', 'MA5', 'MA10', 'MA20', 'MA50',
        'BB_Position', 'Momentum_5', 'Momentum_10', 'Momentum_20'
    ]
    
    # Sentiment factors - describing market psychology, volume, trends
    sentiment_factors = [
        'turnoverVol', 'turnoverValue', 'Volume_Ratio', 'Volume_MA5', 'Volume_MA10',
        'pctChg', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Position', 'Volatility'
    ]
    
    # Technical indicators - moving averages, oscillators, momentum
    technical_indicators = [
        'MA5', 'MA10', 'MA20', 'MA50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
        'Momentum_5', 'Momentum_10', 'Momentum_20'
    ]
    
    # Momentum factors - describing price momentum and trends
    momentum_factors = [
        'Momentum_5', 'Momentum_10', 'Momentum_20',
        'Price_Change_1', 'Price_Change_5', 'Price_Change_10',
        'Log_Return_1', 'Log_Return_5', 'Log_Return_10',
        'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI'
    ]
    
    # Growth factors - describing growth rates and trends
    growth_factors = [
        'Momentum_5', 'Momentum_10', 'Momentum_20',
        'Price_Change_1', 'Price_Change_5', 'Price_Change_10',
        'Log_Return_1', 'Log_Return_5', 'Log_Return_10',
        'pctChg', 'Volatility'
    ]
    
    sector_types = [
        quality_factors,
        return_risk_factors,
        valuation_factors,
        sentiment_factors,
        technical_indicators,
        momentum_factors,
        growth_factors
    ]
    
    return sector_types

def form_us_stock_dataset(feature_col, label_length=3, start_date='2020-01-01', end_date=None):
    """
    Form US stock dataset (replaces form_us_stock_dataset)
    
    Parameters:
    -----------
    feature_col : list
        List of feature column names
    label_length : int
        Length of label sequence
    start_date : str
        Start date for data
    end_date : str
        End date for data
    
    Returns:
    --------
    tuple
        (dataset, labels, lengths, col_nan_record)
    """
    # US stock tickers to use
    us_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
    
    all_data = []
    all_labels = []
    all_lengths = []
    
    for ticker in us_tickers:
        data = load_us_stock_data(ticker, start_date, end_date)
        if data is not None:
            # Select features
            feature_data = data[feature_col].values
            
            # Create labels (simple trend labels)
            close_prices = data['closePrice'].values
            labels = np.zeros(len(close_prices))
            
            for i in range(label_length, len(close_prices)):
                future_prices = close_prices[i:i+label_length]
                current_price = close_prices[i-1]
                
                if np.mean(future_prices) > current_price * 1.02:
                    labels[i] = 1  # Up trend
                elif np.mean(future_prices) < current_price * 0.98:
                    labels[i] = -1  # Down trend
                else:
                    labels[i] = 0  # Sideways
            
            all_data.append(feature_data)
            all_labels.append(labels)
            all_lengths.append(len(feature_data))
    
    if not all_data:
        print("❌ No US stock data loaded")
        return None, None, None, None
    
    dataset = np.vstack(all_data)
    labels = np.hstack(all_labels)
    lengths = all_lengths
    
    # Check for NaN values
    col_nan_record = []
    for i in range(dataset.shape[1]):
        nan_count = np.isnan(dataset[:, i]).sum()
        col_nan_record.append(nan_count)
    
    print(f"✅ Formed US stock dataset with {len(us_tickers)} stocks")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Features: {len(feature_col)}")
    print(f"   Lengths: {lengths}")
    
    return dataset, labels, lengths, col_nan_record

def create_us_stock_data_directory():
    """
    Create US stock data directory structure
    """
    data_dir = 'data/us_stocks'
    os.makedirs(data_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['us_stocks_by_sector', 'processed', 'models', 'results']
    for subdir in subdirs:
        os.makedirs(os.path.join(data_dir, subdir), exist_ok=True)
    
    print(f"✅ Created US stock data directory: {data_dir}")

def main():
    """
    Main function to demonstrate US stock data loading
    """
    print("US Stock Data Loader Demo")
    print("=" * 50)
    
    # Create data directory
    create_us_stock_data_directory()
    
    # Load sample US stock data
    ticker = 'AAPL'
    data = load_us_stock_data(ticker, start_date='2023-01-01')
    
    if data is not None:
        print(f"\nSample data for {ticker}:")
        print(data.head())
        print(f"\nData shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
    
    # Load US stock single score
    scores, features = load_us_stock_single_score()
    print(f"\nUS Stock Features: {len(features)}")
    print(f"Sample features: {features[:10]}")
    
    # Form US sector types
    sector_types = form_us_sector_types()
    print(f"\nUS Sector Types: {len(sector_types)}")
    for i, sector in enumerate(sector_types):
        print(f"  Sector {i+1}: {len(sector)} features")

if __name__ == "__main__":
    main()
