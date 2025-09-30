# US Stock Market Trend Analysis Using HMM-LSTM with Wiener Filtering

A comprehensive machine learning framework for analyzing US stock market trends using Hidden Markov Models (HMM), Long Short-Term Memory (LSTM) networks, XGBoost, and Wiener filtering for noise reduction.

## Features

- **Hidden Markov Models (HMM)** for market state detection
- **LSTM Networks** for sequential pattern recognition
- **XGBoost** for feature-based classification
- **Wiener Filtering** for noise reduction and signal enhancement
- **Cross-validation** for robust model evaluation
- **Real-time US stock data** integration via yfinance
- **Comprehensive technical indicators** calculation
- **Market trend prediction** with confidence intervals

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
# Run the main analysis
python robust_spy_analysis.py

# Train the model
python main_train_us_stock_wiener_model.py

# Analyze SPY market data
python spy_market_analysis.py
```

## Project Structure

```
├── robust_spy_analysis.py          # Main analysis script
├── main_train_us_stock_wiener_model.py  # Main training script
├── spy_market_analysis.py          # SPY market analysis
├── XGB_HMM_Wiener/                 # Core model implementation
│   ├── XGB_HMM_Wiener.py          # Main model class
│   └── __init__.py
├── wiener_filtering/               # Wiener filtering implementation
│   ├── financial_wiener_filter.py # Financial Wiener filter
│   └── __init__.py
├── dataset_code/                   # Data processing utilities
│   ├── us_stock_data_loader.py    # US stock data loader
│   ├── HMM_us_stocks.py           # Multi-factor HMM
│   ├── HMM_us_market.py           # Market HMM
│   └── ...
├── XGB_HMM/                       # XGBoost-HMM implementation
├── train_model/                   # Training scripts
├── public_tool/                   # Utility functions
└── requirements.txt               # Dependencies
```

## Model Architecture

### 1. Wiener Filtering
- **Purpose**: Noise reduction and signal enhancement
- **Types**: Basic and Adaptive filters
- **Parameters**: Filter order, noise variance, signal gain

### 2. Hidden Markov Model (HMM)
- **States**: 3 (Down, Sideways, Up)
- **Purpose**: Market state detection
- **Implementation**: Gaussian HMM with EM algorithm

### 3. XGBoost Classification
- **Purpose**: Feature-based trend prediction
- **Features**: Technical indicators, price ratios, correlations
- **Parameters**: Optimized for financial time series

### 4. LSTM Network
- **Purpose**: Sequential pattern recognition
- **Architecture**: Multi-layer LSTM with dropout
- **Input**: Market state probabilities from HMM

## Data Sources

- **Primary**: US stock market data via yfinance
- **Assets**: SPY, QQQ, IWM, VIX, Treasuries, Commodities, Sectors
- **Indicators**: 46+ technical indicators
- **Time Period**: Configurable (default: 3 years)

## Technical Indicators

- **Price Indicators**: Moving averages, Bollinger Bands, RSI, MACD
- **Volume Indicators**: Volume ratios, turnover rates
- **Momentum Indicators**: Price momentum, rate of change
- **Volatility Indicators**: Historical volatility, ATR
- **Cross-Asset Indicators**: Correlations, relative strength

## Model Performance

- **Cross-validation**: 37.38% ± 3.11%
- **Final Accuracy**: 35.89%
- **Convergence**: Stable with robust error handling
- **Features**: 30 most stable indicators selected

## Usage Examples

### Basic Analysis
```python
from robust_spy_analysis import main
results = main()
```

### Custom Data Analysis
```python
from dataset_code.us_stock_data_loader import load_us_stock_data

# Load data for specific ticker
data = load_us_stock_data('AAPL', start_date='2023-01-01')
```

### Model Training
```python
from XGB_HMM_Wiener.XGB_HMM_Wiener import XGB_HMM_Wiener

model = XGB_HMM_Wiener(n_states=3, filter_order=3)
model.fit(X, lengths)
```

## Dependencies

- numpy
- pandas
- scipy
- scikit-learn
- xgboost
- matplotlib
- hmmlearn
- yfinance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original HMM-LSTM implementation adapted for US markets
- Wiener filtering implementation for financial time series
- Technical indicators based on standard financial analysis

## References

- Hidden Markov Models for financial time series
- LSTM networks for sequence prediction
- XGBoost for gradient boosting
- Wiener filtering for signal processing
- Technical analysis indicators

---

**Note**: This model is for educational and research purposes. Always conduct thorough backtesting before using in live trading.
