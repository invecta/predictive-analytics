# Quick Start Guide - MAESTRO Predictive Analytics

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Basic Usage

```python
import pandas as pd
from maestro import MAESTRO
from market_outlook_analysis import future_stock_market_outlook

# Load your data (CSV with columns: datetime, open, high, low, close, volume)
df = pd.read_csv('your_data.csv', index_col='datetime', parse_dates=True)

# Initialize MAESTRO
maestro = MAESTRO(
    fib_period=1440,
    secondary_timeframe='5T',
    show_signals=True
)

# Calculate indicators
result_df, m5_df = maestro.calculate_signals(df)

# Generate market outlook
outlook = future_stock_market_outlook(df)
print(outlook)
```

### Step 3: Visualize Results

```python
# Plot the indicators
maestro.plot(result_df, show_plot=True)

# Access signals
long_signals = result_df[result_df['long_signal'] == True]
short_signals = result_df[result_df['short_signal'] == True]

print(f"Found {len(long_signals)} long signals")
print(f"Found {len(short_signals)} short signals")
```

## 📊 Example with Sample Data

```python
from example_usage import main
main()
```

## 🔍 Key Features

- **Multi-Timeframe Analysis**: M1 and M5 timeframe integration
- **Fibonacci Levels**: Automatic calculation of key retracement levels
- **ZigZag Indicator**: Pivot point identification
- **Market Outlook**: Short, medium, and long-term predictions
- **Risk Assessment**: Comprehensive risk analysis

## 📈 Output Example

```
SHORT-TERM OUTLOOK (1-5 days):
  Direction: BULLISH
  Key Levels: Support $150.00, Resistance $155.00

MEDIUM-TERM OUTLOOK (1-3 months):
  Direction: BULLISH
  Confidence: HIGH
  Trend Strength: STRONG

LONG-TERM OUTLOOK (3-12 months):
  Direction: BULLISH
  Confidence: MODERATE
```

## 🛠️ Troubleshooting

**Issue**: Import errors
- **Solution**: Ensure all dependencies are installed: `pip install pandas numpy matplotlib`

**Issue**: Data format errors
- **Solution**: Ensure your CSV has columns: datetime, open, high, low, close, volume

**Issue**: Plotting not working
- **Solution**: Install matplotlib: `pip install matplotlib`

## 📚 Next Steps

1. Read the full [README.md](README.md)
2. Explore [FUTURE_MARKET_OUTLOOK_FRAMEWORK.md](FUTURE_MARKET_OUTLOOK_FRAMEWORK.md)
3. Run `python example_usage.py` for a complete example

