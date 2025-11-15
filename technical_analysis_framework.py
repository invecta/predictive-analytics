"""
Technical Analysis Framework for Stock Market Assessment
Combining Technical Indicators with Macroeconomic Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TechnicalMarketAnalysis:
    """
    Comprehensive technical analysis framework combining:
    - Technical indicators (MAESTRO, trend analysis, momentum)
    - Macroeconomic environment assessment
    - Multi-timeframe analysis
    - Risk assessment
    """
    
    def __init__(self, maestro_indicator=None):
        """
        Initialize technical analysis framework
        
        Parameters:
        -----------
        maestro_indicator : MAESTRO instance, optional
            Pre-initialized MAESTRO indicator
        """
        self.maestro = maestro_indicator
        
    def analyze_trend_structure(
        self,
        df: pd.DataFrame,
        short_period: int = 50,
        medium_period: int = 200,
        long_period: int = 500
    ) -> Dict[str, any]:
        """
        Analyze trend structure using multiple moving averages
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data with 'close' column
        short_period : int
            Short-term MA period
        medium_period : int
            Medium-term MA period
        long_period : int
            Long-term MA period
            
        Returns:
        --------
        Dict containing trend analysis results
        """
        close = df['close']
        
        # Calculate moving averages
        ma_short = close.rolling(window=short_period).mean()
        ma_medium = close.rolling(window=medium_period).mean()
        ma_long = close.rolling(window=long_period).mean()
        
        # Current values
        current_price = close.iloc[-1]
        current_ma_short = ma_short.iloc[-1]
        current_ma_medium = ma_medium.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        
        # Trend determination
        bullish_alignment = (current_ma_short > current_ma_medium > current_ma_long)
        bearish_alignment = (current_ma_short < current_ma_medium < current_ma_long)
        
        # Price position relative to MAs
        price_above_all = (current_price > current_ma_short > current_ma_medium > current_ma_long)
        price_below_all = (current_price < current_ma_short < current_ma_medium < current_ma_long)
        
        # Momentum indicators
        rsi = self.calculate_rsi(close, period=14)
        current_rsi = rsi.iloc[-1]
        
        # Volatility
        atr = self.calculate_atr(df, period=14)
        current_atr = atr.iloc[-1]
        volatility_percent = (current_atr / current_price) * 100
        
        # Trend strength
        ma_slope_short = (ma_short.iloc[-1] - ma_short.iloc[-20]) / ma_short.iloc[-20] * 100
        ma_slope_medium = (ma_medium.iloc[-1] - ma_medium.iloc[-20]) / ma_medium.iloc[-20] * 100
        
        return {
            'trend_direction': 'BULLISH' if bullish_alignment else 'BEARISH' if bearish_alignment else 'NEUTRAL',
            'trend_strength': 'STRONG' if abs(ma_slope_medium) > 2 else 'MODERATE' if abs(ma_slope_medium) > 0.5 else 'WEAK',
            'price_position': 'ABOVE_ALL' if price_above_all else 'BELOW_ALL' if price_below_all else 'MIXED',
            'rsi': current_rsi,
            'rsi_signal': 'OVERSOLD' if current_rsi < 30 else 'OVERBOUGHT' if current_rsi > 70 else 'NEUTRAL',
            'volatility_percent': volatility_percent,
            'volatility_assessment': 'HIGH' if volatility_percent > 3 else 'MODERATE' if volatility_percent > 1.5 else 'LOW',
            'ma_short': current_ma_short,
            'ma_medium': current_ma_medium,
            'ma_long': current_ma_long,
            'ma_slope_short': ma_slope_short,
            'ma_slope_medium': ma_slope_medium
        }
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def analyze_support_resistance(
        self,
        df: pd.DataFrame,
        lookback: int = 100
    ) -> Dict[str, any]:
        """
        Identify key support and resistance levels
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data
        lookback : int
            Lookback period for identifying levels
            
        Returns:
        --------
        Dict with support/resistance levels
        """
        recent_data = df.tail(lookback)
        high = recent_data['high']
        low = recent_data['low']
        close = recent_data['close']
        
        # Identify pivot points
        resistance_levels = []
        support_levels = []
        
        # Simple pivot identification
        for i in range(2, len(recent_data) - 2):
            if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i+1]:
                if high.iloc[i] > high.iloc[i-2] and high.iloc[i] > high.iloc[i+2]:
                    resistance_levels.append(high.iloc[i])
            if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i+1]:
                if low.iloc[i] < low.iloc[i-2] and low.iloc[i] < low.iloc[i+2]:
                    support_levels.append(low.iloc[i])
        
        current_price = close.iloc[-1]
        
        # Find nearest support and resistance
        resistance_above = [r for r in resistance_levels if r > current_price]
        support_below = [s for s in support_levels if s < current_price]
        
        nearest_resistance = min(resistance_above) if resistance_above else None
        nearest_support = max(support_below) if support_below else None
        
        return {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_levels': sorted(set(support_levels), reverse=True)[:5],
            'resistance_levels': sorted(set(resistance_levels))[:5],
            'distance_to_support': ((current_price - nearest_support) / current_price * 100) if nearest_support else None,
            'distance_to_resistance': ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None
        }
    
    def assess_macro_environment(
        self,
        macro_data: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Assess macroeconomic environment factors
        
        Parameters:
        -----------
        macro_data : Dict, optional
            Dictionary containing macroeconomic indicators
            Expected keys: 'inflation_rate', 'interest_rates', 'gdp_growth', 
                          'unemployment', 'market_sentiment', 'geopolitical_risk'
            
        Returns:
        --------
        Dict with macroeconomic assessment
        """
        if macro_data is None:
            # Default assessment framework
            return {
                'inflation_assessment': 'MODERATE',
                'monetary_policy': 'NEUTRAL',
                'economic_growth': 'MODERATE',
                'market_sentiment': 'NEUTRAL',
                'overall_bias': 'NEUTRAL',
                'note': 'Provide macro_data for specific assessment'
            }
        
        # Assess inflation impact
        inflation_rate = macro_data.get('inflation_rate', 2.0)
        if inflation_rate > 4:
            inflation_assessment = 'HIGH'
        elif inflation_rate < 1:
            inflation_assessment = 'LOW'
        else:
            inflation_assessment = 'MODERATE'
        
        # Assess monetary policy
        interest_rates = macro_data.get('interest_rates', 2.0)
        if interest_rates > 5:
            monetary_policy = 'RESTRICTIVE'
        elif interest_rates < 1:
            monetary_policy = 'ACCOMMODATIVE'
        else:
            monetary_policy = 'NEUTRAL'
        
        # Assess economic growth
        gdp_growth = macro_data.get('gdp_growth', 2.0)
        if gdp_growth > 3:
            economic_growth = 'STRONG'
        elif gdp_growth < 0:
            economic_growth = 'RECESSION'
        else:
            economic_growth = 'MODERATE'
        
        # Overall bias
        positive_factors = sum([
            inflation_assessment in ['LOW', 'MODERATE'],
            monetary_policy in ['ACCOMMODATIVE', 'NEUTRAL'],
            economic_growth in ['STRONG', 'MODERATE']
        ])
        
        if positive_factors >= 2:
            overall_bias = 'POSITIVE'
        elif positive_factors <= 1:
            overall_bias = 'NEGATIVE'
        else:
            overall_bias = 'NEUTRAL'
        
        return {
            'inflation_assessment': inflation_assessment,
            'monetary_policy': monetary_policy,
            'economic_growth': economic_growth,
            'market_sentiment': macro_data.get('market_sentiment', 'NEUTRAL'),
            'geopolitical_risk': macro_data.get('geopolitical_risk', 'MODERATE'),
            'overall_bias': overall_bias
        }
    
    def comprehensive_market_assessment(
        self,
        df: pd.DataFrame,
        macro_data: Optional[Dict] = None,
        use_maestro: bool = True
    ) -> Dict[str, any]:
        """
        Comprehensive market assessment combining technical and fundamental analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data with OHLCV columns
        macro_data : Dict, optional
            Macroeconomic data
        use_maestro : bool
            Whether to use MAESTRO indicator if available
            
        Returns:
        --------
        Comprehensive assessment dictionary
        """
        assessment = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': df['close'].iloc[-1],
            'price_change_1d': ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100) if len(df) > 1 else 0,
            'price_change_5d': ((df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100) if len(df) > 5 else 0,
        }
        
        # Technical analysis
        trend_analysis = self.analyze_trend_structure(df)
        assessment.update({f'trend_{k}': v for k, v in trend_analysis.items()})
        
        # Support/Resistance
        sr_analysis = self.analyze_support_resistance(df)
        assessment.update({f'sr_{k}': v for k, v in sr_analysis.items()})
        
        # MAESTRO analysis if available
        if use_maestro and self.maestro:
            try:
                maestro_result, _ = self.maestro.calculate_signals(df)
                last_bar = maestro_result.iloc[-1]
                
                assessment['maestro_binary_ls'] = last_bar.get('binary_ls', 0)
                assessment['maestro_fib236'] = last_bar.get('m1_fib236', None)
                assessment['maestro_fib786'] = last_bar.get('m1_fib786', None)
                assessment['maestro_long_signal'] = last_bar.get('long_signal', False)
                assessment['maestro_short_signal'] = last_bar.get('short_signal', False)
                assessment['maestro_ma'] = last_bar.get('m1_ma', None)
                
                # MAESTRO signal interpretation
                if last_bar.get('long_signal', False):
                    assessment['maestro_signal'] = 'BULLISH'
                elif last_bar.get('short_signal', False):
                    assessment['maestro_signal'] = 'BEARISH'
                else:
                    assessment['maestro_signal'] = 'NEUTRAL'
            except Exception as e:
                assessment['maestro_error'] = str(e)
        
        # Macroeconomic assessment
        macro_assessment = self.assess_macro_environment(macro_data)
        assessment.update({f'macro_{k}': v for k, v in macro_assessment.items()})
        
        # Overall verdict
        assessment['overall_verdict'] = self._generate_verdict(assessment)
        
        return assessment
    
    def _generate_verdict(self, assessment: Dict) -> Dict[str, any]:
        """
        Generate overall market verdict based on all factors
        
        Parameters:
        -----------
        assessment : Dict
            Complete assessment dictionary
            
        Returns:
        --------
        Dict with verdict and reasoning
        """
        # Score factors
        bullish_factors = 0
        bearish_factors = 0
        
        # Trend factors
        if assessment.get('trend_trend_direction') == 'BULLISH':
            bullish_factors += 2
        elif assessment.get('trend_trend_direction') == 'BEARISH':
            bearish_factors += 2
        
        if assessment.get('trend_price_position') == 'ABOVE_ALL':
            bullish_factors += 1
        elif assessment.get('trend_price_position') == 'BELOW_ALL':
            bearish_factors += 1
        
        # RSI factors
        rsi = assessment.get('trend_rsi', 50)
        if rsi < 30:
            bullish_factors += 1  # Oversold = potential bounce
        elif rsi > 70:
            bearish_factors += 1  # Overbought = potential pullback
        
        # MAESTRO factors
        if assessment.get('maestro_signal') == 'BULLISH':
            bullish_factors += 2
        elif assessment.get('maestro_signal') == 'BEARISH':
            bearish_factors += 2
        
        # Macro factors
        if assessment.get('macro_overall_bias') == 'POSITIVE':
            bullish_factors += 1
        elif assessment.get('macro_overall_bias') == 'NEGATIVE':
            bearish_factors += 1
        
        # Generate verdict
        if bullish_factors > bearish_factors + 1:
            verdict = 'BULLISH'
            confidence = 'HIGH' if bullish_factors >= 5 else 'MODERATE'
        elif bearish_factors > bullish_factors + 1:
            verdict = 'BEARISH'
            confidence = 'HIGH' if bearish_factors >= 5 else 'MODERATE'
        else:
            verdict = 'NEUTRAL'
            confidence = 'MODERATE'
        
        # Timeframe assessment
        if assessment.get('trend_trend_strength') == 'STRONG':
            timeframe = 'SHORT_TO_MEDIUM_TERM'
        else:
            timeframe = 'MEDIUM_TO_LONG_TERM'
        
        return {
            'direction': verdict,
            'confidence': confidence,
            'timeframe': timeframe,
            'bullish_score': bullish_factors,
            'bearish_score': bearish_factors,
            'key_factors': self._identify_key_factors(assessment, bullish_factors, bearish_factors)
        }
    
    def _identify_key_factors(self, assessment: Dict, bullish: int, bearish: int) -> List[str]:
        """Identify key factors influencing the verdict"""
        factors = []
        
        if assessment.get('trend_trend_direction') == 'BULLISH':
            factors.append(f"Strong bullish trend structure (Trend Strength: {assessment.get('trend_trend_strength')})")
        elif assessment.get('trend_trend_direction') == 'BEARISH':
            factors.append(f"Bearish trend structure (Trend Strength: {assessment.get('trend_trend_strength')})")
        
        rsi = assessment.get('trend_rsi', 50)
        if rsi < 30:
            factors.append(f"RSI indicates oversold conditions ({rsi:.1f}) - potential reversal")
        elif rsi > 70:
            factors.append(f"RSI indicates overbought conditions ({rsi:.1f}) - potential correction")
        
        if assessment.get('maestro_signal'):
            factors.append(f"MAESTRO indicator: {assessment.get('maestro_signal')} signal")
        
        if assessment.get('macro_overall_bias'):
            factors.append(f"Macroeconomic environment: {assessment.get('macro_overall_bias')} bias")
        
        return factors
    
    def generate_market_outlook_report(
        self,
        df: pd.DataFrame,
        macro_data: Optional[Dict] = None,
        use_maestro: bool = True
    ) -> str:
        """
        Generate comprehensive market outlook report
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data
        macro_data : Dict, optional
            Macroeconomic data
        use_maestro : bool
            Use MAESTRO indicator
            
        Returns:
        --------
        Formatted report string
        """
        assessment = self.comprehensive_market_assessment(df, macro_data, use_maestro)
        verdict = assessment['overall_verdict']
        
        report = f"""
================================================================================
COMPREHENSIVE MARKET ANALYSIS REPORT
Analysis Date: {assessment['analysis_date']}
================================================================================

CURRENT MARKET CONDITIONS
-------------------------
Current Price: ${assessment['current_price']:.2f}
1-Day Change: {assessment['price_change_1d']:+.2f}%
5-Day Change: {assessment['price_change_5d']:+.2f}%

TECHNICAL ANALYSIS
------------------
Trend Direction: {assessment['trend_trend_direction']}
Trend Strength: {assessment['trend_trend_strength']}
Price Position: {assessment['trend_price_position']}
RSI: {assessment['trend_rsi']:.2f} ({assessment['trend_rsi_signal']})
Volatility: {assessment['trend_volatility_assessment']} ({assessment['trend_volatility_percent']:.2f}%)

Support/Resistance Levels:
  Nearest Support: {f"${assessment['sr_nearest_support']:.2f} ({assessment['sr_distance_to_support']:.2f}% below)" if assessment['sr_nearest_support'] else 'N/A'}
  Nearest Resistance: {f"${assessment['sr_nearest_resistance']:.2f} ({assessment['sr_distance_to_resistance']:.2f}% above)" if assessment['sr_nearest_resistance'] else 'N/A'}

MAESTRO INDICATOR ANALYSIS
--------------------------
"""
        if use_maestro and self.maestro:
            report += f"""
Binary LS Flag: {assessment.get('maestro_binary_ls', 'N/A')}
MAESTRO Signal: {assessment.get('maestro_signal', 'N/A')}
Fibonacci 23.6%: ${assessment.get('maestro_fib236', 0):.2f}
Fibonacci 78.6%: ${assessment.get('maestro_fib786', 0):.2f}
Thrive EMA: ${assessment.get('maestro_ma', 0):.2f}
"""
        else:
            report += "MAESTRO indicator not available or not used.\n"
        
        report += f"""
MACROECONOMIC ENVIRONMENT
-------------------------
Inflation Assessment: {assessment['macro_inflation_assessment']}
Monetary Policy: {assessment['macro_monetary_policy']}
Economic Growth: {assessment['macro_economic_growth']}
Market Sentiment: {assessment['macro_market_sentiment']}
Overall Macro Bias: {assessment['macro_overall_bias']}

================================================================================
OVERALL VERDICT
================================================================================

Direction: {verdict['direction']}
Confidence: {verdict['confidence']}
Timeframe: {verdict['timeframe']}

Key Factors:
"""
        for factor in verdict['key_factors']:
            report += f"  • {factor}\n"
        
        report += f"""
Score Summary:
  Bullish Factors: {verdict['bullish_score']}
  Bearish Factors: {verdict['bearish_score']}

================================================================================
RISK DISCLAIMER
================================================================================
This analysis is for informational purposes only and should not be considered
as financial advice. Market conditions can change rapidly. Always conduct your
own research and consider consulting with qualified financial advisors before
making investment decisions.

================================================================================
"""
        return report


# Example usage and market outlook function
def generate_market_outlook(
    price_data: pd.DataFrame,
    macro_indicators: Optional[Dict] = None,
    maestro_indicator=None
) -> Tuple[Dict, str]:
    """
    Generate market outlook based on current conditions
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Historical price data (OHLCV format)
    macro_indicators : Dict, optional
        Macroeconomic indicators dictionary
    maestro_indicator : MAESTRO instance, optional
        MAESTRO indicator instance
        
    Returns:
    --------
    Tuple of (assessment_dict, report_string)
    """
    analyzer = TechnicalMarketAnalysis(maestro_indicator=maestro_indicator)
    assessment = analyzer.comprehensive_market_assessment(price_data, macro_indicators)
    report = analyzer.generate_market_outlook_report(price_data, macro_indicators)
    
    return assessment, report


if __name__ == "__main__":
    print("Technical Market Analysis Framework")
    print("=" * 80)
    print("\nThis framework provides:")
    print("  • Comprehensive technical analysis")
    print("  • Macroeconomic environment assessment")
    print("  • Multi-indicator signal generation")
    print("  • Risk-aware market outlook")
    print("\nUse generate_market_outlook() function for complete analysis.")

