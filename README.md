# Crypto Trading Bot

This project is a cryptocurrency trading bot that uses a machine learning model to predict price movements and execute trades.

## Features

*   **Automated Trading:** The bot fully automates the trading process, from market analysis to trade execution.
*   **Machine Learning-Powered:** It uses a sophisticated AI model to make informed trading decisions.
*   **Customizable Configuration:** You can easily configure the bot to suit your trading preferences and risk tolerance.
*   **Real-Time Monitoring:** The bot provides real-time monitoring of your portfolio and trading performance.
*   **Extensible Architecture:** The project is designed to be easily extensible, allowing you to add new features and trading strategies.
*   **Dynamic Position Sizing:** The bot can adjust its trade sizes based on market volatility, using larger positions in less volatile conditions and smaller positions when the market is choppy.
*   **Adaptive Model Selection:** The bot can switch between different machine learning models that are optimized for specific market conditions, such as high or low volatility.
*   **Advanced Sentiment Analysis:** The bot's sentiment analysis capabilities have been upgraded to incorporate a time-decay factor and source-based weighting, allowing it to prioritize more recent news from more reliable sources.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You will need to have Docker installed on your machine to run this project.

### Installation

1.  Build the Docker image:

    ```bash
    docker build -t crypto-trading-bot .
    ```

2.  Run the Docker container:

    ```bash
    docker run -p 8501:8501 -p 8000:8000 crypto-trading-bot
    ```

## Usage

Once the container is running, you can access the Streamlit interface by navigating to `http://localhost:8501` in your web browser.

The application also exposes a Prometheus metrics endpoint at `http://localhost:8000`.
