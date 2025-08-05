import streamlit as st
import yaml
import pandas as pd
import time
import torch
from scripts.training.train_sequential import train as train_sequential
from scripts.backtest import backtest
from scripts.inference.ai_engine import AIEngine
from scripts.exchange import Exchange

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

        # Run the bot in a separate thread
        from threading import Thread
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

def run_bot(config):
    ai_engine = AIEngine(config)
    exchange = Exchange(config)
    position = None

    while st.session_state.bot_running:
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
                print("Buying...")
                # exchange.create_order(config['data']['symbol'], 'market', 'buy', 1)
                position = 'buy'
        else:
            if position != 'sell':
                print("Selling...")
                # exchange.create_order(config['data']['symbol'], 'market', 'sell', 1)
                position = 'sell'

        time.sleep(60)


if __name__ == '__main__':
    main()
