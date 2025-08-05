import streamlit as st
import yaml
import pandas as pd
import time
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

    if 'bot_running' in st.session_state and st.session_state.bot_running:
        st.header('Live Trading')
        exchange = Exchange(config)
        balance = exchange.get_balance('USDT')
        st.write(f"USDT Balance: {balance['free']:.2f}")

        # This is a simplified loop. In a real application, this would run in a separate thread.
        with st.spinner('Bot is running...'):
            while st.session_state.bot_running:
                # Fetch data
                # Make prediction
                # Execute trade
                time.sleep(60)


if __name__ == '__main__':
    main()
