# MAESTRO Predictive Analytics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/github/stars/invecta/predictive-analytics?style=social)](https://github.com/invecta/predictive-analytics)

**Comprehensive Trading Indicator and Market Analysis Framework**

A professional Python implementation of the MAESTRO trading indicator with advanced technical analysis capabilities, macroeconomic integration, and predictive market outlook generation.

## 🎯 What This Project Does

This project provides:
- **MAESTRO Trading Indicator**: Python conversion of the Pine Script indicator with Fibonacci retracements, ZigZag analysis, and multi-timeframe support
- **Technical Analysis Framework**: Comprehensive trend analysis, momentum indicators, and support/resistance identification
- **Market Outlook System**: Short, medium, and long-term market predictions combining technical and fundamental analysis
- **Risk Assessment**: Automated risk evaluation and strategy recommendations

## 🚀 Quick Start

See [QUICKSTART.md](QUICKSTART.md) for a 5-minute getting started guide.

## 📋 Overview

This is a Python conversion of the MAESTRO Pine Script indicator for TradingView, enhanced with additional predictive analytics capabilities.

## Features

- **Fibonacci Retracement Levels**: Calculates FIB 0%, 23.6%, 78.6%, and 100% levels
- **Thrive EMA**: Custom exponential moving average with multiple iterations
- **ZigZag Indicator**: BHS ZigZag implementation for identifying pivot points
- **Multi-Timeframe Analysis**: Analyzes both M1 (primary) and M5 (secondary) timeframes
- **Trading Signals**: Generates long and short entry signals based on multiple conditions

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import pandas as pd
from maestro import MAESTRO

# Load your OHLCV data (must have datetime index)
df = pd.read_csv('your_data.csv', index_col='datetime', parse_dates=True)

# Initialize MAESTRO
maestro = MAESTRO(
    fib_period=1440,
    secondary_timeframe='5T',  # 5 minutes
    zigzag_period=777,
    show_signals=True
)

# Calculate indicators and signals
result_df, m5_df = maestro.calculate_signals(df)

# Plot results
maestro.plot(result_df)

# Access signals
long_signals = result_df[result_df['long_signal'] == True]
short_signals = result_df[result_df['short_signal'] == True]
```

### Run Example

```bash
python example_usage.py
```

## Parameters

- `fib_period` (int): Period for Fibonacci calculations (default: 1440 bars)
- `secondary_timeframe` (str): Secondary timeframe for M5 analysis (default: "5T")
- `zigzag_period` (int): Period for ZigZag calculation (default: 777)
- `zigzag_dev_step` (int): Deviation step for ZigZag, 0 = no filter (default: 0)
- `zigzag_lookback_multiplier` (int): Lookback multiplier for ZigZag (default: 10)
- `volume_factor` (float): Volume factor for Thrive EMA (default: 0.62)
- `show_signals` (bool): Show entry/exit signals (default: True)
- `show_m5_levels` (bool): Show M5 levels (default: True)
- `show_zigzag` (bool): Show ZigZag indicator (default: True)

## Output Columns

The `calculate_signals()` method returns a DataFrame with the following columns:

### M1 (Primary Timeframe) Indicators
- `m1_lowest_low`: Lowest low over FIB period
- `m1_highest_high`: Highest high over FIB period
- `m1_fib236`: Fibonacci 23.6% level
- `m1_fib786`: Fibonacci 78.6% level
- `m1_ma`: Thrive EMA
- `binary_ls`: Binary flag (1 = long, -1 = short)
- `m1_hit_fib0`: Boolean indicating FIB 0% hit
- `m1_hit_fib100`: Boolean indicating FIB 100% hit
- `m1_long_entry`: M1 long entry condition
- `m1_short_entry`: M1 short entry condition

### M5 (Secondary Timeframe) Indicators
- `m5_lowest_low`: M5 lowest low
- `m5_highest_high`: M5 highest high
- `m5_fib236`: M5 Fibonacci 23.6% level
- `m5_fib786`: M5 Fibonacci 78.6% level
- `m5_ma`: M5 Thrive EMA
- `m5_long_entry`: M5 long entry condition
- `m5_short_entry`: M5 short entry condition
- `m5_hit_fib0`, `m5_hit_fib100`, `m5_hit_fib236`, `m5_hit_fib786`: FIB hit indicators

### Signals
- `long_signal`: Final long signal (M1 + M5 conditions met)
- `short_signal`: Final short signal (M1 + M5 conditions met)

## Signal Logic

### Long Signal
- M1: Close crosses above Thrive EMA AND binary_ls == 1
- M5: Close > M5 Thrive EMA AND Close > M5 FIB 23.6%
- Both conditions must be met

### Short Signal
- M1: Close crosses below Thrive EMA AND binary_ls == -1
- M5: Close < M5 Thrive EMA AND Close < M5 FIB 78.6%
- Both conditions must be met

## Notes

- The indicator requires OHLCV data with a datetime index
- The secondary timeframe is automatically resampled from the primary data
- ZigZag calculation uses conservative lookback limits for safety
- All calculations are vectorized using pandas/numpy for performance

## 📚 Additional Features

### Market Outlook Analysis

Generate comprehensive market outlooks combining technical and fundamental analysis:

```python
from market_outlook_analysis import future_stock_market_outlook

# Generate outlook report
outlook = future_stock_market_outlook(
    price_data=df,
    macro_environment={
        'inflation_rate': 2.5,
        'interest_rates': 2.0,
        'gdp_growth': 2.8,
        'market_sentiment': 'NEUTRAL'
    }
)
print(outlook)
```

### Technical Analysis Framework

Use the comprehensive technical analysis framework:

```python
from technical_analysis_framework import TechnicalMarketAnalysis

analyzer = TechnicalMarketAnalysis(maestro_indicator=maestro)
assessment = analyzer.comprehensive_market_assessment(df, macro_data)
report = analyzer.generate_market_outlook_report(df, macro_data)
```

## 📖 Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: Quick start guide
- **[FUTURE_MARKET_OUTLOOK_FRAMEWORK.md](FUTURE_MARKET_OUTLOOK_FRAMEWORK.md)**: Comprehensive analysis framework documentation
- **[example_usage.py](example_usage.py)**: Complete usage examples

## 🛠️ Installation Options

### Option 1: Using pip (Recommended)

```bash
pip install -r requirements.txt
```

### Option 2: Install as Package

```bash
pip install -e .
```

## 📊 Example Output

The framework generates comprehensive analysis including:

- **Short-term outlook** (1-5 days): Direction, key levels, reasoning
- **Medium-term outlook** (1-3 months): Trend strength, confidence level
- **Long-term outlook** (3-12 months): Macroeconomic factors, strategic view
- **Risk assessment**: Identified risks and mitigation strategies
- **Strategy recommendations**: Trading approach based on analysis

## 🔧 Differences from Pine Script Version

- Visualization uses matplotlib instead of TradingView's built-in plotting
- Multi-timeframe data is handled via pandas resampling
- Enhanced with macroeconomic analysis capabilities
- Additional risk assessment and strategy recommendation features
- Some Pine Script-specific features (like labels) are converted to DataFrame columns
- Alert functionality would need to be implemented separately based on your needs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice. Always conduct your own research and consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.

## 👤 Author

**Hani Hindaoui**
- Email: hindaouihani@gmail.com
- GitHub: [@invecta](https://github.com/invecta)

## 🙏 Acknowledgments

- Original MAESTRO Pine Script indicator
- TradingView community
- Python data science community

---

⭐ If you find this project useful, please consider giving it a star!

