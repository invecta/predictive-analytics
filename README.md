# MAESTRO Trading Indicator - Python Version

This is a Python conversion of the MAESTRO Pine Script indicator for TradingView.

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

## Differences from Pine Script Version

- Visualization uses matplotlib instead of TradingView's built-in plotting
- Multi-timeframe data is handled via pandas resampling
- Some Pine Script-specific features (like labels) are converted to DataFrame columns
- Alert functionality would need to be implemented separately based on your needs

