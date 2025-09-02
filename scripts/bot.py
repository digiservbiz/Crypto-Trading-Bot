import streamlit as st
import time
import torch
import pandas as pd
import requests
from scripts.inference.ai_engine import AIEngine
from scripts.exchange import Exchange
from scripts.notifier import Notifier
from prometheus_client import Gauge
from scripts.logger import get_logger
from flair.models import TextClassifier
from flair.data import Sentence

logger = get_logger(__name__)

# Load pre-trained sentiment analysis model
classifier = TextClassifier.load('en-sentiment')

# Metrics
BALANCE = Gauge('bot_balance', 'Current balance in USDT')
TOTAL_PROFIT_LOSS = Gauge('bot_total_profit_loss', 'Total profit or loss in USDT')
OPEN_POSITIONS = Gauge('bot_open_positions', 'Number of open positions')
TRADES = Gauge('bot_trades', 'Trades', ['symbol', 'type', 'price', 'quantity'])

def get_news(symbol):
    """Fetches news articles for a given cryptocurrency symbol."""
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories={symbol}"
    response = requests.get(url)
    news_data = response.json()['Data']
    return [article['body'] for article in news_data]

def get_sentiment(text):
    """Analyzes the sentiment of a given text."""
    sentence = Sentence(text)
    classifier.predict(sentence)
    return sentence.labels[0].score * (1 if sentence.labels[0].value == 'POSITIVE' else -1)

def run_bot(config):
    ai_engine = AIEngine(config)
    exchange = Exchange(config)
    notifier = Notifier(config)
    
    # Use dictionaries to manage state for multiple symbols
    symbols = config['data']['symbols']
    positions = {symbol: None for symbol in symbols}
    entry_prices = {symbol: 0 for symbol in symbols}
    highest_prices = {symbol: 0 for symbol in symbols}
    
    initial_balance = exchange.get_balance('USDT')['free']

    while st.session_state.bot_running:
        try:
            # Update global metrics
            balance = exchange.get_balance('USDT')
            BALANCE.set(balance['free'])
            TOTAL_PROFIT_LOSS.set(balance['free'] - initial_balance)
            OPEN_POSITIONS.set(sum(1 for p in positions.values() if p is not None))

            # Loop through each symbol to trade
            for symbol in symbols:
                try:
                    # Fetch data for the current symbol
                    ohlcv = exchange.exchange.fetch_ohlcv(symbol, config['data']['timeframe'], limit=config['data']['lookback'])
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['returns'] = df['close'].pct_change()
                    df['volatility'] = df['returns'].rolling(20).std()
                    df.ta.macd(append=True)
                    df.ta.rsi(append=True)
                    df.ta.bbands(append=True)
                    df.ta.atr(append=True)
                    df.ta.obv(append=True)
                    df.dropna(inplace=True)

                    # Get sentiment
                    news = get_news(symbol.split('/')[0])
                    sentiment = sum(get_sentiment(article) for article in news) / len(news) if news else 0

                    # Make prediction for the current symbol
                    features = ['close', 'volume', 'volatility', 'MACD_12_26_9', 'RSI_14', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'ATRr_14', 'OBV']
                    model_input_tensor = torch.FloatTensor(df[features].values).unsqueeze(0)
                    model_input_dict = {'volume': df[['volume']].values}
                    # Pass the symbol to the predict function
                    prediction = ai_engine.predict(model_input_tensor, model_input_dict, symbol)

                    # Manage position for the current symbol
                    if positions[symbol]:
                        pnl = (df['close'].iloc[-1] - entry_prices[symbol]) / entry_prices[symbol]
                        
                        take_profit_percentage = config['trading']['take_profit_percentage']
                        if config['trading']['dynamic_take_profit']['enabled']:
                            volatility = df['volatility'].iloc[-1]
                            take_profit_percentage = volatility * config['trading']['dynamic_take_profit']['volatility_multiplier']
                        
                        if positions[symbol] == 'buy':
                            highest_prices[symbol] = max(highest_prices[symbol], df['close'].iloc[-1])
                            trailing_stop_price = highest_prices[symbol] * (1 - config['trading']['trailing_stop']['percentage'])
                            if df['close'].iloc[-1] < trailing_stop_price or pnl > take_profit_percentage:
                                logger.info(f"[{symbol}] Closing position with PnL: {pnl:.2f}")
                                notifier.send_message(f"[{symbol}] Closing position with PnL: {pnl:.2f}")
                                # exchange.create_order(symbol, 'market', 'sell', trade_amount)
                                TRADES.labels(symbol, 'sell', df['close'].iloc[-1], trade_amount).set(1)
                                positions[symbol] = None
                        elif positions[symbol] == 'sell':
                            highest_prices[symbol] = min(highest_prices[symbol], df['close'].iloc[-1])
                            trailing_stop_price = highest_prices[symbol] * (1 + config['trading']['trailing_stop']['percentage'])
                            if df['close'].iloc[-1] > trailing_stop_price or pnl < -take_profit_percentage:
                                logger.info(f"[{symbol}] Closing position with PnL: {pnl:.2f}")
                                notifier.send_message(f"[{symbol}] Closing position with PnL: {pnl:.2f}")
                                # exchange.create_order(symbol, 'market', 'buy', trade_amount)
                                TRADES.labels(symbol, 'buy', df['close'].iloc[-1], trade_amount).set(1)
                                positions[symbol] = None

                    # Calculate position size
                    trade_amount = (balance['free'] * config['trading']['risk_percentage']) / df['close'].iloc[-1]

                    # Execute trade for the current symbol
                    if not positions[symbol]:
                        if prediction['direction'] and not prediction['is_anomaly'][-1] and sentiment > 0.5:
                            logger.info(f"[{symbol}] Buying {trade_amount} of {symbol}...")
                            notifier.send_message(f"[{symbol}] Buying {trade_amount} of {symbol}...")
                            # exchange.create_order(symbol, 'market', 'buy', trade_amount)
                            TRADES.labels(symbol, 'buy', df['close'].iloc[-1], trade_amount).set(1)
                            positions[symbol] = 'buy'
                            entry_prices[symbol] = df['close'].iloc[-1]
                            highest_prices[symbol] = df['close'].iloc[-1]
                        elif not prediction['direction'] and not prediction['is_anomaly'][-1] and sentiment < -0.5:
                            logger.info(f"[{symbol}] Selling {trade_amount} of {symbol}...")
                            notifier.send_message(f"[{symbol}] Selling {trade_amount} of {symbol}...")
                            # exchange.create_order(symbol, 'market', 'sell', trade_amount)
                            TRADES.labels(symbol, 'sell', df['close'].iloc[-1], trade_amount).set(1)
                            positions[symbol] = 'sell'
                            entry_prices[symbol] = df['close'].iloc[-1]
                            highest_prices[symbol] = df['close'].iloc[-1]

                except Exception as e:
                    logger.error(f"An error occurred in the trading loop for {symbol}: {e}")
                    notifier.send_message(f"An error occurred in the trading loop for {symbol}: {e}")
            
            # Wait before the next cycle
            time.sleep(60)

        except Exception as e:
            logger.error(f"An error occurred in the main bot loop: {e}")
            notifier.send_message(f"An error occurred in the main bot loop: {e}")
            time.sleep(60)
