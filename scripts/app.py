import streamlit as st
import yaml
import pandas as pd
import time
import torch
from scripts.training.train_sequential import train as train_sequential
from scripts.backtest import backtest
from scripts.inference.ai_engine import AIEngine
from scripts.exchange import Exchange
from prometheus_client import start_http_server, Gauge

# Metrics
BALANCE = Gauge('bot_balance', 'Current balance in USDT')
TOTAL_PROFIT_LOSS = Gauge('bot_total_profit_loss', 'Total profit or loss in USDT')
OPEN_POSITIONS = Gauge('bot_open_positions', 'Number of open positions')

def main():
    st.title('Crypto Trading Bot')

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Sidebar
    st.sidebar.title('Configuration')
    model_type = st.sidebar.selectbox('Model Type', ['lstm', 'transformer'])
    config['models']['model_type'] = model_type

    st.sidebar.title('Training')
    if st.sidebar.button('Train Models'):
        with st.spinner('Training models...'):
            train_sequential(config)
        st.sidebar.success('Models trained successfully!')

    st.sidebar.title('Backtesting')
    if st.sidebar.button('Run Backtest'):
        with st.spinner('Running backtest...'):
            backtest_results = backtest(config)
            st.session_state.backtest_results = backtest_results
        st.sidebar.success('Backtest complete!')

    # Main content
    if 'backtest_results' in st.session_state:
        st.header('Backtest Results')
        st.write(f"Total Return: {st.session_state.backtest_results['total_return']:.4f}")
        st.write(f"Sharpe Ratio: {st.session_state.backtest_results['sharpe_ratio']:.2f}")
        st.write(f"Max Drawdown: {st.session_state.backtest_results['max_drawdown']:.4f}")
        st.write(f"Calmar Ratio: {st.session_state.backtest_results['calmar_ratio']:.2f}")
        st.line_chart(st.session_state.backtest_results['cumulative_returns'])

    st.sidebar.title('Live Trading')
    if st.sidebar.button('Start Bot'):
        st.session_state.bot_running = True
        st.sidebar.success('Bot started!')

    if st.sidebar.button('Stop Bot'):
        st.session_state.bot_running = False
        st.sidebar.success('Bot stopped!')

    if 'bot_running' not in st.session_state:
        st.session_state.bot_running = False

    if st.sidebar.button('Start Bot'):
        st.session_state.bot_running = True
        st.sidebar.success('Bot started!')

        # Start the metrics server in a separate thread
        from threading import Thread
        metrics_thread = Thread(target=start_http_server, args=(8000,))
        metrics_thread.start()

        # Run the bot in a separate thread
        bot_thread = Thread(target=run_bot, args=(config,))
        bot_thread.start()

    if st.sidebar.button('Stop Bot'):
        st.session_state.bot_running = False
        st.sidebar.success('Bot stopped!')

    if st.session_state.bot_running:
        st.header('Live Trading')
        exchange = Exchange(config)
        balance = exchange.get_balance('USDT')
        st.write(f"USDT Balance: {balance['free']:.2f}")

from scripts.logger import get_logger

logger = get_logger(__name__)

def run_bot(config):
    ai_engine = AIEngine(config)
    exchange = Exchange(config)
    position = None
    initial_balance = exchange.get_balance('USDT')['free']

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

            # Execute trade
            if prediction['direction'] and not prediction['is_anomaly'][-1]:
                if position != 'buy':
                    logger.info("Buying...")
                    # exchange.create_order(config['data']['symbol'], 'market', 'buy', 1)
                    position = 'buy'
            else:
                if position != 'sell':
                    logger.info("Selling...")
                    # exchange.create_order(config['data']['symbol'], 'market', 'sell', 1)
                    position = 'sell'
        except Exception as e:
            logger.error(f"An error occurred in the trading loop: {e}")

        time.sleep(60)


if __name__ == '__main__':
    main()
