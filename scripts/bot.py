import streamlit as st
import time
import torch
import pandas as pd
from scripts.inference.ai_engine import AIEngine
from scripts.exchange import Exchange
from scripts.notifier import Notifier
from prometheus_client import Gauge
from scripts.logger import get_logger

logger = get_logger(__name__)


# Metrics
BALANCE = Gauge('bot_balance', 'Current balance in USDT')
TOTAL_PROFIT_LOSS = Gauge('bot_total_profit_loss', 'Total profit or loss in USDT')
OPEN_POSITIONS = Gauge('bot_open_positions', 'Number of open positions')

def run_bot(config):
    ai_engine = AIEngine(config)
    exchange = Exchange(config)
    notifier = Notifier(config)
    position = None
    initial_balance = exchange.get_balance('USDT')['free']
    entry_price = 0

    while st.session_state.bot_running:
        try:
            # Update metrics
            balance = exchange.get_balance('USDT')
            BALANCE.set(balance['free'])
            TOTAL_PROFIT_LOSS.set(balance['free'] - initial_balance)
            OPEN_POSITIONS.set(1 if position else 0)

            # Fetch data
            ohlcv = exchange.exchange.fetch_ohlcv(config['data']['symbol'], config['data']['timeframe'], limit=config['data']['lookback'])
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            df.ta.macd(append=True)
            df.ta.rsi(append=True)
            df.ta.bbands(append=True)
            df.dropna(inplace=True)

            # Make prediction
            features = ['close', 'volume', 'volatility', 'MACD_12_26_9', 'RSI_14', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0']
            model_input_tensor = torch.FloatTensor(df[features].values).unsqueeze(0)
            model_input_dict = {'volume': df[['volume']].values}
            prediction = ai_engine.predict(model_input_tensor, model_input_dict)

            # Manage position
            if position:
                pnl = (df['close'].iloc[-1] - entry_price) / entry_price
                if position == 'buy':
                    if pnl < -config['trading']['stop_loss_percentage'] or pnl > config['trading']['take_profit_percentage']:
                        logger.info(f"Closing position with PnL: {pnl:.2f}")
                        notifier.send_message(f"Closing position with PnL: {pnl:.2f}")
                        # exchange.create_order(config['data']['symbol'], 'market', 'sell', trade_amount)
                        position = None
                elif position == 'sell':
                    if pnl > config['trading']['stop_loss_percentage'] or pnl < -config['trading']['take_profit_percentage']:
                        logger.info(f"Closing position with PnL: {pnl:.2f}")
                        notifier.send_message(f"Closing position with PnL: {pnl:.2f}")
                        # exchange.create_order(config['data']['symbol'], 'market', 'buy', trade_amount)
                        position = None

            # Calculate position size
            trade_amount = (balance['free'] * config['trading']['risk_percentage']) / df['close'].iloc[-1]

            # Execute trade
            if not position:
                if prediction['direction'] and not prediction['is_anomaly'][-1]:
                    logger.info(f"Buying {trade_amount} of {config['data']['symbol']}...")
                    notifier.send_message(f"Buying {trade_amount} of {config['data']['symbol']}...")
                    # exchange.create_order(config['data']['symbol'], 'market', 'buy', trade_amount)
                    position = 'buy'
                    entry_price = df['close'].iloc[-1]
                else:
                    logger.info(f"Selling {trade_amount} of {config['data']['symbol']}...")
                    notifier.send_message(f"Selling {trade_amount} of {config['data']['symbol']}...")
                    # exchange.create_order(config['data']['symbol'], 'market', 'sell', trade_amount)
                    position = 'sell'
                    entry_price = df['close'].iloc[-1]
        except Exception as e:
            logger.error(f"An error occurred in the trading loop: {e}")
            notifier.send_message(f"An error occurred in the trading loop: {e}")

        time.sleep(60)