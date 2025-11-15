"""
Example usage of MAESTRO Trading Indicator
"""

import pandas as pd
import numpy as np
from maestro import MAESTRO

# Generate sample data (replace this with your actual data loading)
def generate_sample_data(n_bars=5000):
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='1T')
    
    # Generate random walk price data
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.randn(n_bars) * 0.001
    prices = base_price + np.cumsum(returns)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'open': prices + np.random.randn(n_bars) * 0.1,
        'high': prices + np.abs(np.random.randn(n_bars) * 0.2),
        'low': prices - np.abs(np.random.randn(n_bars) * 0.2),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_bars)
    }, index=dates)
    
    # Ensure high >= close >= low and high >= open >= low
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    return df


def main():
    print("MAESTRO Trading Indicator - Example Usage")
    print("=" * 50)
    
    # Load or generate data
    print("\n1. Loading data...")
    # Option 1: Use sample data
    df = generate_sample_data(5000)
    
    # Option 2: Load from CSV (uncomment and modify as needed)
    # df = pd.read_csv('your_data.csv', index_col='datetime', parse_dates=True)
    # Ensure columns: 'open', 'high', 'low', 'close', 'volume'
    
    print(f"   Loaded {len(df)} bars")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # Initialize MAESTRO indicator
    print("\n2. Initializing MAESTRO indicator...")
    maestro = MAESTRO(
        fib_period=1440,              # Fibonacci period (1440 bars)
        secondary_timeframe='5T',     # 5 minutes for M5 analysis
        zigzag_period=777,            # ZigZag period
        zigzag_dev_step=0,            # ZigZag deviation step (0 = no filter)
        zigzag_lookback_multiplier=10, # ZigZag lookback multiplier
        volume_factor=0.62,           # Thrive EMA volume factor
        show_signals=True,             # Show entry/exit signals
        show_m5_levels=True,          # Show M5 levels
        show_zigzag=True              # Show ZigZag indicator
    )
    
    # Calculate indicators and signals
    print("\n3. Calculating indicators and signals...")
    result_df, m5_df = maestro.calculate_signals(df)
    
    print("   ✓ M1 indicators calculated")
    print("   ✓ M5 indicators calculated")
    print("   ✓ Signals generated")
    
    # Display results
    print("\n4. Results Summary:")
    print("-" * 50)
    
    # Count signals
    long_signals = result_df[result_df['long_signal'] == True]
    short_signals = result_df[result_df['short_signal'] == True]
    
    print(f"   Long Signals:  {len(long_signals)}")
    print(f"   Short Signals: {len(short_signals)}")
    
    # Show recent signals
    if len(long_signals) > 0:
        print("\n   Recent Long Signals:")
        recent_long = long_signals.tail(5)
        for idx, row in recent_long.iterrows():
            print(f"     {idx}: Price = {row['close']:.2f}, MA = {row['m1_ma']:.2f}")
    
    if len(short_signals) > 0:
        print("\n   Recent Short Signals:")
        recent_short = short_signals.tail(5)
        for idx, row in recent_short.iterrows():
            print(f"     {idx}: Price = {row['close']:.2f}, MA = {row['m1_ma']:.2f}")
    
    # Display current levels
    print("\n5. Current Levels (last bar):")
    print("-" * 50)
    last_bar = result_df.iloc[-1]
    print(f"   Close Price:     {last_bar['close']:.2f}")
    print(f"   M1 THRIVE EMA:   {last_bar['m1_ma']:.2f}")
    print(f"   FIB 0% (Low):    {last_bar['m1_lowest_low']:.2f}")
    print(f"   FIB 23.6%:       {last_bar['m1_fib236']:.2f}")
    print(f"   FIB 78.6%:       {last_bar['m1_fib786']:.2f}")
    print(f"   FIB 100% (High): {last_bar['m1_highest_high']:.2f}")
    print(f"   Binary LS:       {last_bar['binary_ls']}")
    
    # Plot results
    print("\n6. Generating plot...")
    try:
        maestro.plot(result_df, show_plot=True)
        print("   ✓ Plot displayed")
    except Exception as e:
        print(f"   ✗ Plot failed: {e}")
    
    # Save results to CSV
    print("\n7. Saving results...")
    output_file = 'maestro_results.csv'
    result_df.to_csv(output_file)
    print(f"   ✓ Results saved to {output_file}")
    
    # Save signals separately
    if len(long_signals) > 0 or len(short_signals) > 0:
        signals_df = pd.concat([
            long_signals[['close', 'm1_ma', 'm1_fib236', 'm1_fib786', 'binary_ls']].assign(signal='LONG'),
            short_signals[['close', 'm1_ma', 'm1_fib236', 'm1_fib786', 'binary_ls']].assign(signal='SHORT')
        ]).sort_index()
        signals_file = 'maestro_signals.csv'
        signals_df.to_csv(signals_file)
        print(f"   ✓ Signals saved to {signals_file}")
    
    print("\n" + "=" * 50)
    print("Analysis complete!")


if __name__ == "__main__":
    main()

