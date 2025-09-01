import streamlit as st
import yaml
from scripts.training.train_sequential import train as train_sequential
from scripts.backtest import backtest
from scripts.bot import run_bot
from prometheus_client import start_http_server
from threading import Thread

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
    
    if 'bot_running' not in st.session_state:
        st.session_state.bot_running = False

    if st.sidebar.button('Start Bot'):
        st.session_state.bot_running = True
        st.sidebar.success('Bot started!')

        # Start the metrics server in a separate thread
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

if __name__ == '__main__':
    main()