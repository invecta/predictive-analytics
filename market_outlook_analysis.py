"""
Market Outlook Analysis - Technical Assessment Framework
Provides structured approach to analyzing future stock market conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime
from technical_analysis_framework import TechnicalMarketAnalysis, generate_market_outlook
from maestro import MAESTRO


def analyze_future_market_conditions(
    historical_data: pd.DataFrame,
    current_macro_environment: dict = None
) -> dict:
    """
    Comprehensive analysis of future stock market conditions based on current data
    
    This function combines:
    1. Technical analysis using multiple indicators
    2. Trend structure analysis
    3. Support/resistance identification
    4. Macroeconomic environment assessment
    5. Risk factor evaluation
    
    Parameters:
    -----------
    historical_data : pd.DataFrame
        Historical OHLCV price data with datetime index
    current_macro_environment : dict, optional
        Current macroeconomic indicators:
        {
            'inflation_rate': float,
            'interest_rates': float,
            'gdp_growth': float,
            'unemployment': float,
            'market_sentiment': str,  # 'BULLISH', 'BEARISH', 'NEUTRAL'
            'geopolitical_risk': str  # 'LOW', 'MODERATE', 'HIGH'
        }
    
    Returns:
    --------
    dict: Comprehensive market outlook with predictions and reasoning
    """
    
    # Initialize MAESTRO indicator for advanced technical analysis
    maestro = MAESTRO(
        fib_period=1440,
        secondary_timeframe='5T',
        zigzag_period=777,
        show_signals=True
    )
    
    # Initialize technical analysis framework
    analyzer = TechnicalMarketAnalysis(maestro_indicator=maestro)
    
    # Generate comprehensive assessment
    assessment, report = generate_market_outlook(
        historical_data,
        current_macro_environment,
        maestro
    )
    
    # Extract key insights
    outlook = {
        'analysis_timestamp': datetime.now().isoformat(),
        'current_price': float(assessment['current_price']),
        'short_term_outlook': _determine_short_term_outlook(assessment),
        'medium_term_outlook': _determine_medium_term_outlook(assessment),
        'long_term_outlook': _determine_long_term_outlook(assessment),
        'key_levels': {
            'support': float(assessment.get('sr_nearest_support', 0)) if assessment.get('sr_nearest_support') else None,
            'resistance': float(assessment.get('sr_nearest_resistance', 0)) if assessment.get('sr_nearest_resistance') else None,
            'fibonacci_236': float(assessment.get('maestro_fib236', 0)) if assessment.get('maestro_fib236') else None,
            'fibonacci_786': float(assessment.get('maestro_fib786', 0)) if assessment.get('maestro_fib786') else None
        },
        'risk_assessment': _assess_risks(assessment),
        'recommended_strategy': _recommend_strategy(assessment),
        'confidence_level': assessment['overall_verdict']['confidence'],
        'detailed_report': report
    }
    
    return outlook


def _determine_short_term_outlook(assessment: dict) -> dict:
    """Determine short-term (1-5 days) market outlook"""
    trend = assessment['trend_trend_direction']
    rsi = assessment['trend_rsi']
    maestro_signal = assessment.get('maestro_signal', 'NEUTRAL')
    
    # Short-term factors
    if maestro_signal == 'BULLISH' and trend == 'BULLISH':
        direction = 'BULLISH'
        reasoning = "Strong technical alignment: MAESTRO bullish signal + bullish trend structure"
    elif maestro_signal == 'BEARISH' and trend == 'BEARISH':
        direction = 'BEARISH'
        reasoning = "Strong technical alignment: MAESTRO bearish signal + bearish trend structure"
    elif rsi < 30:
        direction = 'BULLISH'
        reasoning = f"Oversold conditions (RSI: {rsi:.1f}) suggest potential short-term bounce"
    elif rsi > 70:
        direction = 'BEARISH'
        reasoning = f"Overbought conditions (RSI: {rsi:.1f}) suggest potential short-term correction"
    else:
        direction = 'NEUTRAL'
        reasoning = "Mixed signals - consolidation likely"
    
    return {
        'direction': direction,
        'timeframe': '1-5 days',
        'reasoning': reasoning,
        'key_levels': {
            'support': assessment.get('sr_nearest_support'),
            'resistance': assessment.get('sr_nearest_resistance')
        }
    }


def _determine_medium_term_outlook(assessment: dict) -> dict:
    """Determine medium-term (1-3 months) market outlook"""
    trend_strength = assessment['trend_trend_strength']
    trend_direction = assessment['trend_trend_direction']
    macro_bias = assessment['macro_overall_bias']
    
    # Medium-term factors weigh trend structure and macro environment
    if trend_strength == 'STRONG' and trend_direction == 'BULLISH':
        if macro_bias == 'POSITIVE':
            direction = 'BULLISH'
            confidence = 'HIGH'
            reasoning = "Strong bullish trend structure supported by positive macroeconomic environment"
        else:
            direction = 'BULLISH'
            confidence = 'MODERATE'
            reasoning = "Strong bullish trend structure, but macro environment is neutral/negative"
    elif trend_strength == 'STRONG' and trend_direction == 'BEARISH':
        if macro_bias == 'NEGATIVE':
            direction = 'BEARISH'
            confidence = 'HIGH'
            reasoning = "Strong bearish trend structure exacerbated by negative macroeconomic environment"
        else:
            direction = 'BEARISH'
            confidence = 'MODERATE'
            reasoning = "Strong bearish trend structure, but macro environment may provide support"
    else:
        direction = 'NEUTRAL'
        confidence = 'MODERATE'
        reasoning = "Weak trend structure suggests range-bound or choppy conditions"
    
    return {
        'direction': direction,
        'timeframe': '1-3 months',
        'confidence': confidence,
        'reasoning': reasoning,
        'trend_strength': trend_strength
    }


def _determine_long_term_outlook(assessment: dict) -> dict:
    """Determine long-term (3-12 months) market outlook"""
    macro_bias = assessment['macro_overall_bias']
    economic_growth = assessment['macro_economic_growth']
    monetary_policy = assessment['macro_monetary_policy']
    trend_direction = assessment['trend_trend_direction']
    
    # Long-term factors emphasize macroeconomic fundamentals
    if macro_bias == 'POSITIVE' and economic_growth == 'STRONG':
        direction = 'BULLISH'
        confidence = 'HIGH'
        reasoning = "Strong economic fundamentals support long-term bullish outlook"
    elif macro_bias == 'NEGATIVE' or economic_growth == 'RECESSION':
        direction = 'BEARISH'
        confidence = 'HIGH'
        reasoning = "Weak economic fundamentals suggest challenging long-term conditions"
    elif monetary_policy == 'RESTRICTIVE':
        direction = 'NEUTRAL_TO_BEARISH'
        confidence = 'MODERATE'
        reasoning = "Restrictive monetary policy may limit long-term growth potential"
    else:
        direction = 'NEUTRAL'
        confidence = 'MODERATE'
        reasoning = "Mixed macroeconomic signals suggest moderate long-term outlook"
    
    return {
        'direction': direction,
        'timeframe': '3-12 months',
        'confidence': confidence,
        'reasoning': reasoning,
        'macro_factors': {
            'economic_growth': economic_growth,
            'monetary_policy': monetary_policy,
            'overall_bias': macro_bias
        }
    }


def _assess_risks(assessment: dict) -> dict:
    """Assess key risks to market outlook"""
    risks = []
    risk_level = 'MODERATE'
    
    volatility = assessment['trend_volatility_percent']
    if volatility > 3:
        risks.append(f"High volatility ({volatility:.2f}%) increases short-term risk")
        risk_level = 'HIGH'
    
    rsi = assessment['trend_rsi']
    if rsi > 70:
        risks.append(f"Overbought conditions (RSI: {rsi:.1f}) suggest potential correction risk")
    elif rsi < 30:
        risks.append(f"Oversold conditions (RSI: {rsi:.1f}) may indicate underlying weakness")
    
    macro_risk = assessment.get('macro_geopolitical_risk', 'MODERATE')
    if macro_risk == 'HIGH':
        risks.append("Elevated geopolitical risk may cause market volatility")
        risk_level = 'HIGH'
    
    if assessment['macro_monetary_policy'] == 'RESTRICTIVE':
        risks.append("Restrictive monetary policy may limit market upside")
    
    if not risks:
        risks.append("No significant risk factors identified at current levels")
        risk_level = 'LOW'
    
    return {
        'overall_risk_level': risk_level,
        'identified_risks': risks,
        'volatility_assessment': assessment['trend_volatility_assessment']
    }


def _recommend_strategy(assessment: dict) -> dict:
    """Recommend trading/investment strategy based on analysis"""
    verdict = assessment['overall_verdict']
    direction = verdict['direction']
    confidence = verdict['confidence']
    timeframe = verdict['timeframe']
    
    if direction == 'BULLISH' and confidence == 'HIGH':
        strategy = 'LONG_BIAS'
        approach = "Consider long positions with trend-following approach. Use pullbacks to key support levels as entry opportunities."
    elif direction == 'BEARISH' and confidence == 'HIGH':
        strategy = 'SHORT_BIAS'
        approach = "Consider defensive positioning or short positions. Use rallies to resistance levels as exit/short entry opportunities."
    elif direction == 'BULLISH' and confidence == 'MODERATE':
        strategy = 'CAUTIOUS_LONG'
        approach = "Moderate bullish bias with strict risk management. Consider smaller position sizes and tight stop-losses."
    elif direction == 'BEARISH' and confidence == 'MODERATE':
        strategy = 'DEFENSIVE'
        approach = "Defensive positioning recommended. Consider reducing exposure or hedging strategies."
    else:
        strategy = 'NEUTRAL_RANGE_TRADING'
        approach = "Range-bound conditions expected. Consider range trading strategies between support and resistance levels."
    
    return {
        'strategy': strategy,
        'approach': approach,
        'timeframe': timeframe,
        'risk_management': 'Always use appropriate position sizing and stop-losses regardless of outlook'
    }


# Main function to answer: "What does the future stock market look like?"
def future_stock_market_outlook(
    price_data: pd.DataFrame,
    macro_environment: dict = None
) -> str:
    """
    Generate comprehensive answer to: "What does the future stock market look like?"
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Historical OHLCV data
    macro_environment : dict, optional
        Current macroeconomic indicators
        
    Returns:
    --------
    Formatted outlook report
    """
    outlook = analyze_future_market_conditions(price_data, macro_environment)
    
    report = f"""
================================================================================
FUTURE STOCK MARKET OUTLOOK - COMPREHENSIVE ANALYSIS
================================================================================
Analysis Date: {outlook['analysis_timestamp']}
Current Price: ${outlook['current_price']:.2f}

EXECUTIVE SUMMARY
-----------------
Based on comprehensive technical analysis and macroeconomic assessment:

SHORT-TERM OUTLOOK (1-5 days):
  Direction: {outlook['short_term_outlook']['direction']}
  {outlook['short_term_outlook']['reasoning']}
  
  Key Levels:
    Support: {f"${outlook['short_term_outlook']['key_levels']['support']:.2f}" if outlook['short_term_outlook']['key_levels']['support'] else 'N/A'}
    Resistance: {f"${outlook['short_term_outlook']['key_levels']['resistance']:.2f}" if outlook['short_term_outlook']['key_levels']['resistance'] else 'N/A'}

MEDIUM-TERM OUTLOOK (1-3 months):
  Direction: {outlook['medium_term_outlook']['direction']}
  Confidence: {outlook['medium_term_outlook']['confidence']}
  {outlook['medium_term_outlook']['reasoning']}
  
  Trend Strength: {outlook['medium_term_outlook']['trend_strength']}

LONG-TERM OUTLOOK (3-12 months):
  Direction: {outlook['long_term_outlook']['direction']}
  Confidence: {outlook['long_term_outlook']['confidence']}
  {outlook['long_term_outlook']['reasoning']}
  
  Macroeconomic Factors:
    Economic Growth: {outlook['long_term_outlook']['macro_factors']['economic_growth']}
    Monetary Policy: {outlook['long_term_outlook']['macro_factors']['monetary_policy']}
    Overall Bias: {outlook['long_term_outlook']['macro_factors']['overall_bias']}

RISK ASSESSMENT
---------------
Overall Risk Level: {outlook['risk_assessment']['overall_risk_level']}
Volatility: {outlook['risk_assessment']['volatility_assessment']}

Identified Risks:
"""
    for risk in outlook['risk_assessment']['identified_risks']:
        report += f"  • {risk}\n"
    
    report += f"""
RECOMMENDED STRATEGY
--------------------
Strategy: {outlook['recommended_strategy']['strategy']}
Timeframe: {outlook['recommended_strategy']['timeframe']}

Approach: {outlook['recommended_strategy']['approach']}

Risk Management: {outlook['recommended_strategy']['risk_management']}

KEY TECHNICAL LEVELS
--------------------
"""
    for level_name, level_value in outlook['key_levels'].items():
        if level_value:
            report += f"  {level_name.replace('_', ' ').title()}: ${level_value:.2f}\n"
    
    report += f"""
CONFIDENCE LEVEL
---------------
Overall Confidence: {outlook['confidence_level']}

================================================================================
DETAILED TECHNICAL ANALYSIS
================================================================================
{outlook['detailed_report']}

================================================================================
DISCLAIMER
================================================================================
This analysis is based on technical indicators and available macroeconomic data.
Market conditions are dynamic and can change rapidly. Past performance does not
guarantee future results. This analysis should not be considered as financial
advice. Always conduct thorough research and consult with qualified financial
professionals before making investment decisions.

================================================================================
"""
    
    return report


if __name__ == "__main__":
    print("=" * 80)
    print("FUTURE STOCK MARKET OUTLOOK ANALYSIS")
    print("=" * 80)
    print("\nThis module provides comprehensive analysis of future market conditions")
    print("combining technical analysis with macroeconomic assessment.")
    print("\nUsage:")
    print("  from market_outlook_analysis import future_stock_market_outlook")
    print("  outlook = future_stock_market_outlook(your_price_data, macro_data)")
    print("  print(outlook)")

