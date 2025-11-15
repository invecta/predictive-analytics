"""
MAESTRO Trading Indicator - Python Conversion
Original Pine Script converted to Python
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class MAESTRO:
    """
    MAESTRO Trading Indicator
    
    Features:
    - Fibonacci retracement levels (0, 23.6%, 78.6%, 100%)
    - Thrive EMA (custom exponential moving average)
    - ZigZag indicator
    - Multi-timeframe analysis (M1 and M5)
    - Entry/exit signals
    """
    
    def __init__(
        self,
        fib_period: int = 1440,
        secondary_timeframe: str = "5T",  # 5 minutes
        zigzag_period: int = 777,
        zigzag_dev_step: int = 0,
        zigzag_lookback_multiplier: int = 10,
        volume_factor: float = 0.62,
        show_signals: bool = True,
        show_m5_levels: bool = True,
        show_zigzag: bool = True
    ):
        """
        Initialize MAESTRO indicator
        
        Parameters:
        -----------
        fib_period : int
            Period for Fibonacci calculations (default: 1440 bars)
        secondary_timeframe : str
            Secondary timeframe for M5 analysis (default: "5T" for 5 minutes)
        zigzag_period : int
            Period for ZigZag calculation
        zigzag_dev_step : int
            Deviation step for ZigZag (0 = no deviation filter)
        zigzag_lookback_multiplier : int
            Lookback multiplier for ZigZag
        volume_factor : float
            Volume factor for Thrive EMA (default: 0.62)
        show_signals : bool
            Whether to show entry/exit signals
        show_m5_levels : bool
            Whether to show M5 levels
        show_zigzag : bool
            Whether to show ZigZag
        """
        self.fib_period = fib_period
        self.secondary_timeframe = secondary_timeframe
        self.zigzag_period = zigzag_period
        self.zigzag_dev_step = zigzag_dev_step
        self.zigzag_lookback_multiplier = zigzag_lookback_multiplier
        self.volume_factor = volume_factor
        self.show_signals = show_signals
        self.show_m5_levels = show_m5_levels
        self.show_zigzag = show_zigzag
        
        # State variables
        self.binary_ls = 0  # Binary flag for long/short
        
    def thrive_ema(self, src: pd.Series, periods: int) -> pd.Series:
        """
        Calculate Thrive EMA (custom exponential moving average)
        
        Parameters:
        -----------
        src : pd.Series
            Source data (typically close prices)
        periods : int
            Period for EMA calculation
            
        Returns:
        --------
        pd.Series
            Thrive EMA values
        """
        e1 = src.ewm(span=periods, adjust=False).mean()
        e2 = e1.ewm(span=periods, adjust=False).mean()
        e3 = e2.ewm(span=periods, adjust=False).mean()
        e4 = e3.ewm(span=periods, adjust=False).mean()
        e5 = e4.ewm(span=periods, adjust=False).mean()
        e6 = e5.ewm(span=periods, adjust=False).mean()
        
        a = self.volume_factor
        c1 = -a * a * a
        c2 = 3 * a * a + 3 * a * a * a
        c3 = -6 * a * a - 3 * a - 3 * a * a * a
        c4 = 1 + 3 * a + a * a * a + 3 * a * a
        
        return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    
    def calculate_zigzag(
        self,
        high: pd.Series,
        low: pd.Series,
        bar_index: pd.Series,
        mintick: float = 0.01
    ) -> Tuple[List[int], List[int], List[float], List[float]]:
        """
        Calculate ZigZag indicator (BHS ZigZag logic)
        
        Parameters:
        -----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        bar_index : pd.Series
            Bar indices
        mintick : float
            Minimum tick size
            
        Returns:
        --------
        Tuple[List[int], List[int], List[float], List[float]]
            (high_bars, low_bars, high_values, low_values)
        """
        # Safety limit for lookback
        max_safe_lookback = 15
        effective_period_zz = min(self.zigzag_period, max_safe_lookback - 3)
        max_lookback_with_period = max(0, max_safe_lookback - effective_period_zz)
        
        # Convert to numpy arrays for easier indexing
        high_arr = high.values
        low_arr = low.values
        bar_index_arr = bar_index.values
        
        # Temporary arrays for extrema
        temp_high_values = []
        temp_low_values = []
        temp_high_bars = []
        temp_low_bars = []
        
        # Get current bar index (last bar)
        current_bar_idx = len(bar_index_arr) - 1
        calculated_lookback = effective_period_zz * self.zigzag_lookback_multiplier
        lookback_bars = min(current_bar_idx, min(calculated_lookback, max_lookback_with_period))
        start_bar = max(0, current_bar_idx - lookback_bars)
        
        mem_low = 0.0
        mem_high = 0.0
        
        # PASS 1: Find all extrema
        for bar_idx in range(start_bar, current_bar_idx + 1):
            lookback = current_bar_idx - bar_idx
            
            # Safety checks
            if lookback > max_lookback_with_period or lookback < 0:
                continue
            if lookback > current_bar_idx or lookback >= max_safe_lookback or lookback < 0:
                continue
            if lookback + effective_period_zz > max_safe_lookback:
                continue
            if lookback >= max_safe_lookback:
                continue
            
            # Find lowest value in period window
            lowest_value = low_arr[lookback]
            is_lowest = True
            
            for j in range(effective_period_zz):
                check_lb = lookback + j
                if check_lb >= max_safe_lookback or check_lb > current_bar_idx or check_lb < 0:
                    break
                if check_lb >= max_safe_lookback:
                    break
                if low_arr[check_lb] < lowest_value:
                    is_lowest = False
                    break
            
            # If it's a local minimum and different from previous
            if is_lowest and lowest_value != mem_low:
                mem_low = lowest_value
                if low_arr[lookback] - lowest_value <= self.zigzag_dev_step * mintick:
                    temp_low_values.append(lowest_value)
                    temp_low_bars.append(bar_idx)
            
            # Find highest value in period window
            if lookback >= max_safe_lookback:
                continue
            
            highest_value = high_arr[lookback]
            is_highest = True
            
            for j in range(effective_period_zz):
                check_lb = lookback + j
                if check_lb >= max_safe_lookback or check_lb > current_bar_idx or check_lb < 0:
                    break
                if check_lb >= max_safe_lookback:
                    break
                if high_arr[check_lb] > highest_value:
                    is_highest = False
                    break
            
            # If it's a local maximum and different from previous
            if is_highest and highest_value != mem_high:
                mem_high = highest_value
                if highest_value - high_arr[lookback] <= self.zigzag_dev_step * mintick:
                    temp_high_values.append(highest_value)
                    temp_high_bars.append(bar_idx)
        
        # PASS 2: Force alternation (MQL4 exact logic)
        zz_high_bars = []
        zz_low_bars = []
        zz_high_values = []
        zz_low_values = []
        
        last_high = -1.0
        last_high_idx = -1
        last_low = -1.0
        last_low_idx = -1
        last_type = 0  # 0=start, 1=last was high, -1=last was low
        
        idx_high = 0
        idx_low = 0
        size_high = len(temp_high_bars)
        size_low = len(temp_low_bars)
        
        # Process all pivots in chronological order
        while idx_high < size_high or idx_low < size_low:
            next_high_bar = temp_high_bars[idx_high] if idx_high < size_high else 999999
            next_low_bar = temp_low_bars[idx_low] if idx_low < size_low else 999999
            
            # Process next chronological pivot
            if next_high_bar < next_low_bar:
                # It's a HIGH
                val = temp_high_values[idx_high]
                bar_num = next_high_bar
                
                if last_type == -1 or last_type == 0:
                    # Last was a low (or start) → add this high
                    zz_high_bars.append(bar_num)
                    zz_high_values.append(val)
                    last_high = val
                    last_high_idx = len(zz_high_bars) - 1
                    last_type = 1
                elif last_type == 1:
                    # Last was also a high → keep the HIGHEST
                    if val > last_high:
                        zz_high_bars[last_high_idx] = bar_num
                        zz_high_values[last_high_idx] = val
                        last_high = val
                
                idx_high += 1
            else:
                # It's a LOW
                val = temp_low_values[idx_low]
                bar_num = next_low_bar
                
                if last_type == 1 or last_type == 0:
                    # Last was a high (or start) → add this low
                    zz_low_bars.append(bar_num)
                    zz_low_values.append(val)
                    last_low = val
                    last_low_idx = len(zz_low_bars) - 1
                    last_type = -1
                elif last_type == -1:
                    # Last was also a low → keep the LOWEST
                    if val < last_low:
                        zz_low_bars[last_low_idx] = bar_num
                        zz_low_values[last_low_idx] = val
                        last_low = val
                
                idx_low += 1
        
        return zz_high_bars, zz_low_bars, zz_high_values, zz_low_values
    
    def calculate_fib_levels(
        self,
        high: pd.Series,
        low: pd.Series,
        period: Optional[int] = None
    ) -> Dict[str, pd.Series]:
        """
        Calculate Fibonacci retracement levels
        
        Parameters:
        -----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        period : int, optional
            Period for calculation (default: self.fib_period)
            
        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary with FIB levels: 'lowest_low', 'highest_high', 'fib236', 'fib786'
        """
        if period is None:
            period = self.fib_period
        
        lowest_low = low.rolling(window=period, min_periods=1).min()
        highest_high = high.rolling(window=period, min_periods=1).max()
        
        fib236 = lowest_low + 0.236 * (highest_high - lowest_low)
        fib786 = lowest_low + 0.786 * (highest_high - lowest_low)
        
        return {
            'lowest_low': lowest_low,
            'highest_high': highest_high,
            'fib236': fib236,
            'fib786': fib786
        }
    
    def detect_fib_hits(
        self,
        price: pd.Series,
        fib_level: pd.Series,
        tolerance: float = 0.01,
        use_high: bool = False,
        use_low: bool = False
    ) -> pd.Series:
        """
        Detect when price hits a Fibonacci level
        
        Parameters:
        -----------
        price : pd.Series
            Price series (close, high, or low)
        fib_level : pd.Series
            Fibonacci level to check
        tolerance : float
            Tolerance for hit detection
        use_high : bool
            If True, check if high crosses the level
        use_low : bool
            If True, check if low crosses the level
            
        Returns:
        --------
        pd.Series
            Boolean series indicating hits
        """
        if use_high and use_low:
            # Check if level is between high and low
            return (price <= fib_level + tolerance) & (price >= fib_level - tolerance)
        else:
            return (price - fib_level).abs() <= tolerance
    
    def calculate_signals(
        self,
        df: pd.DataFrame,
        m5_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate trading signals
        
        Parameters:
        -----------
        df : pd.DataFrame
            Main timeframe DataFrame with columns: 'open', 'high', 'low', 'close'
        m5_df : pd.DataFrame, optional
            M5 timeframe DataFrame (if None, will be resampled from df)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all indicators and signals added
        """
        # Make a copy to avoid modifying original
        result_df = df.copy()
        
        # Ensure datetime index
        if not isinstance(result_df.index, pd.DatetimeIndex):
            if 'datetime' in result_df.columns:
                result_df.set_index('datetime', inplace=True)
            else:
                raise ValueError("DataFrame must have a datetime index or 'datetime' column")
        
        # Calculate M1 (primary) levels
        fib_levels_m1 = self.calculate_fib_levels(result_df['high'], result_df['low'])
        result_df['m1_lowest_low'] = fib_levels_m1['lowest_low']
        result_df['m1_highest_high'] = fib_levels_m1['highest_high']
        result_df['m1_fib236'] = fib_levels_m1['fib236']
        result_df['m1_fib786'] = fib_levels_m1['fib786']
        result_df['m1_ma'] = self.thrive_ema(result_df['close'], self.fib_period)
        
        # Calculate binary flag (long/short)
        result_df['binary_ls'] = 0
        crossover = (result_df['close'] > result_df['m1_fib236']) & (result_df['close'].shift(1) <= result_df['m1_fib236'].shift(1))
        crossunder = (result_df['close'] < result_df['m1_fib786']) & (result_df['close'].shift(1) >= result_df['m1_fib786'].shift(1))
        
        result_df.loc[crossover, 'binary_ls'] = 1
        result_df.loc[crossunder, 'binary_ls'] = -1
        
        # Forward fill binary_ls
        result_df['binary_ls'] = result_df['binary_ls'].replace(0, np.nan).ffill().fillna(0).astype(int)
        
        # Detect M1 FIB hits
        tolerance = 0.01  # Default mintick
        result_df['m1_hit_fib0'] = self.detect_fib_hits(result_df['low'], result_df['m1_lowest_low'], tolerance)
        result_df['m1_hit_fib100'] = self.detect_fib_hits(result_df['high'], result_df['m1_highest_high'], tolerance)
        
        # M1 Entry conditions
        ma_crossover = (result_df['close'] > result_df['m1_ma']) & (result_df['close'].shift(1) <= result_df['m1_ma'].shift(1))
        ma_crossunder = (result_df['close'] < result_df['m1_ma']) & (result_df['close'].shift(1) >= result_df['m1_ma'].shift(1))
        
        result_df['m1_long_entry'] = ma_crossover & (result_df['binary_ls'] == 1)
        result_df['m1_short_entry'] = ma_crossunder & (result_df['binary_ls'] == -1)
        
        # Resample for M5 if not provided
        if m5_df is None:
            m5_df = result_df.resample(self.secondary_timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
        else:
            m5_df = m5_df.copy()
            if not isinstance(m5_df.index, pd.DatetimeIndex):
                if 'datetime' in m5_df.columns:
                    m5_df.set_index('datetime', inplace=True)
        
        # Calculate M5 levels
        fib_levels_m5 = self.calculate_fib_levels(m5_df['high'], m5_df['low'])
        m5_df['m5_lowest_low'] = fib_levels_m5['lowest_low']
        m5_df['m5_highest_high'] = fib_levels_m5['highest_high']
        m5_df['m5_fib236'] = fib_levels_m5['fib236']
        m5_df['m5_fib786'] = fib_levels_m5['fib786']
        m5_df['m5_ma'] = self.thrive_ema(m5_df['close'], self.fib_period)
        
        # M5 Entry conditions
        m5_df['m5_long_entry'] = (m5_df['close'] > m5_df['m5_ma']) & (m5_df['close'] > m5_df['m5_fib236'])
        m5_df['m5_short_entry'] = (m5_df['close'] < m5_df['m5_ma']) & (m5_df['close'] < m5_df['m5_fib786'])
        
        # Detect M5 FIB hits
        m5_df['m5_hit_fib0'] = self.detect_fib_hits(m5_df['low'], m5_df['m5_lowest_low'], tolerance)
        m5_df['m5_hit_fib100'] = self.detect_fib_hits(m5_df['high'], m5_df['m5_highest_high'], tolerance)
        m5_df['m5_hit_fib236'] = (
            (m5_df['close'] - m5_df['m5_fib236']).abs() <= tolerance |
            ((m5_df['low'] <= m5_df['m5_fib236']) & (m5_df['high'] >= m5_df['m5_fib236']))
        )
        m5_df['m5_hit_fib786'] = (
            (m5_df['close'] - m5_df['m5_fib786']).abs() <= tolerance |
            ((m5_df['low'] <= m5_df['m5_fib786']) & (m5_df['high'] >= m5_df['m5_fib786']))
        )
        
        # Merge M5 data back to M1 timeframe
        m5_columns = ['m5_lowest_low', 'm5_highest_high', 'm5_fib236', 'm5_fib786', 'm5_ma',
                     'm5_long_entry', 'm5_short_entry', 'm5_hit_fib0', 'm5_hit_fib100',
                     'm5_hit_fib236', 'm5_hit_fib786']
        
        for col in m5_columns:
            result_df[col] = m5_df[col].reindex(result_df.index, method='ffill')
        
        # Generate final signals
        result_df['long_signal'] = result_df['m1_long_entry'] & result_df['m5_long_entry']
        result_df['short_signal'] = result_df['m1_short_entry'] & result_df['m5_short_entry']
        
        # Calculate ZigZag if enabled
        if self.show_zigzag:
            try:
                bar_index = pd.Series(range(len(result_df)), index=result_df.index)
                zz_high_bars, zz_low_bars, zz_high_values, zz_low_values = self.calculate_zigzag(
                    result_df['high'],
                    result_df['low'],
                    bar_index,
                    mintick=tolerance
                )
                result_df['zigzag_high_bars'] = None
                result_df['zigzag_low_bars'] = None
                result_df['zigzag_high_values'] = None
                result_df['zigzag_low_values'] = None
                
                # Store ZigZag data in result (for plotting)
                result_df.attrs['zigzag_high_bars'] = zz_high_bars
                result_df.attrs['zigzag_low_bars'] = zz_low_bars
                result_df.attrs['zigzag_high_values'] = zz_high_values
                result_df.attrs['zigzag_low_values'] = zz_low_values
            except Exception as e:
                print(f"Warning: ZigZag calculation failed: {e}")
        
        return result_df, m5_df
    
    def plot(
        self,
        df: pd.DataFrame,
        m5_df: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (15, 10),
        show_plot: bool = True
    ):
        """
        Plot the MAESTRO indicator
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with calculated indicators
        m5_df : pd.DataFrame, optional
            M5 DataFrame (if None, will use data from df)
        figsize : Tuple[int, int]
            Figure size
        show_plot : bool
            Whether to display the plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            print("matplotlib is required for plotting. Install with: pip install matplotlib")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot price
        ax.plot(df.index, df['close'], label='Close', color='white', alpha=0.3, linewidth=1)
        
        # Plot M1 MA
        ax.plot(df.index, df['m1_ma'], label='M1 THRIVE EMA', color='blue', linewidth=2)
        
        # Fill between MA and close
        ax.fill_between(df.index, df['m1_ma'], df['close'],
                       where=(df['close'] > df['m1_ma']),
                       color='green', alpha=0.15, label='Bullish Zone')
        ax.fill_between(df.index, df['m1_ma'], df['close'],
                       where=(df['close'] <= df['m1_ma']),
                       color='red', alpha=0.15, label='Bearish Zone')
        
        # Plot M5 MA if enabled
        if self.show_m5_levels and 'm5_ma' in df.columns:
            ax.plot(df.index, df['m5_ma'], label='M5 THRIVE EMA', color='blue', alpha=0.7, linewidth=1, linestyle='--')
        
        # Plot Fibonacci levels
        ax.plot(df.index, df['m1_fib236'], label='FIB 23.6%', color='yellow', alpha=0.5, linestyle=':')
        ax.plot(df.index, df['m1_fib786'], label='FIB 78.6%', color='orange', alpha=0.5, linestyle=':')
        ax.plot(df.index, df['m1_lowest_low'], label='FIB 0%', color='cyan', alpha=0.5, linestyle=':')
        ax.plot(df.index, df['m1_highest_high'], label='FIB 100%', color='magenta', alpha=0.5, linestyle=':')
        
        # Plot signals
        if self.show_signals:
            long_signals = df[df['long_signal'] == True]
            short_signals = df[df['short_signal'] == True]
            
            if len(long_signals) > 0:
                ax.scatter(long_signals.index, long_signals['low'] * 0.999,
                          marker='^', color='green', s=100, label='Long Signal', zorder=5)
            if len(short_signals) > 0:
                ax.scatter(short_signals.index, short_signals['high'] * 1.001,
                          marker='v', color='red', s=100, label='Short Signal', zorder=5)
        
        # Plot ZigZag if enabled
        if self.show_zigzag and 'zigzag_high_bars' in df.attrs:
            zz_high_bars = df.attrs['zigzag_high_bars']
            zz_low_bars = df.attrs['zigzag_low_bars']
            zz_high_values = df.attrs['zigzag_high_values']
            zz_low_values = df.attrs['zigzag_low_values']
            
            if zz_high_bars:
                high_indices = [df.index[i] for i in zz_high_bars if i < len(df)]
                high_prices = [zz_high_values[j] for j, i in enumerate(zz_high_bars) if i < len(df)]
                ax.scatter(high_indices, high_prices, marker='o', color='white', s=50, alpha=0.7, label='ZZ High')
            
            if zz_low_bars:
                low_indices = [df.index[i] for i in zz_low_bars if i < len(df)]
                low_prices = [zz_low_values[j] for j, i in enumerate(zz_low_bars) if i < len(df)]
                ax.scatter(low_indices, low_prices, marker='o', color='white', s=50, alpha=0.7, label='ZZ Low')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title('MAESTRO Trading Indicator')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
        return fig, ax


# Example usage
if __name__ == "__main__":
    # Example: Load data and calculate indicators
    print("MAESTRO Trading Indicator - Python Version")
    print("=" * 50)
    print("\nUsage example:")
    print("""
    import pandas as pd
    from maestro import MAESTRO
    
    # Load your OHLCV data
    df = pd.read_csv('your_data.csv', index_col='datetime', parse_dates=True)
    
    # Initialize MAESTRO
    maestro = MAESTRO(
        fib_period=1440,
        secondary_timeframe='5T',
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
    
    print(f"Found {len(long_signals)} long signals")
    print(f"Found {len(short_signals)} short signals")
    """)
