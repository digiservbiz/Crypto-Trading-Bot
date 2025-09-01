# Crypto Trading Bot Documentation

## 1. Project Overview

This document provides a comprehensive guide to the Crypto Trading Bot, a project designed to automate cryptocurrency trading using machine learning. The bot leverages a sequential model to predict market trends and execute trades accordingly.

### 1.1. Key Features

*   **Automated Trading:** The bot fully automates the trading process, from market analysis to trade execution.
*   **Machine Learning-Powered:** It uses a sophisticated AI model to make informed trading decisions.
*   **Customizable Configuration:** You can easily configure the bot to suit your trading preferences and risk tolerance.
*   **Real-Time Monitoring:** The bot provides real-time monitoring of your portfolio and trading performance.
*   **Extensible Architecture:** The project is designed to be easily extensible, allowing you to add new features and trading strategies.

### 1.2. Technology Stack

*   **Python:** The core programming language used for the bot's development.
*   **Streamlit:** A Python library for creating interactive web applications, used for the bot's user interface.
*   **PyTorch:** A deep learning framework used for building and training the AI model.
*   **Docker:** A containerization platform used for packaging and deploying the bot in a portable and consistent environment.
*   **Prometheus:** A monitoring and alerting toolkit used for collecting and storing metrics about the bot's performance.
*   **Telegram:** A messaging app used for sending real-time notifications about the bot's activity.

## 2. Getting Started

This section will guide you through the process of setting up and running the Crypto Trading Bot on your local machine.

### 2.1. Prerequisites

Before you begin, ensure you have the following software installed on your system:

*   **Git:** For cloning the project repository.
*   **Docker:** For building and running the bot in a containerized environment.
*   **Python 3.8 or higher:** For running the bot locally without Docker.

### 2.2. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/crypto-trading-bot.git
    cd crypto-trading-bot
    ```

2.  **Set up the environment:**

    *   **(Recommended) Using Docker:**

        ```bash
        docker build -t crypto-trading-bot .
        docker run -p 8501:8501 -p 8000:8000 crypto-trading-bot
        ```

    *   **(Alternative) Local Installation:**

        ```bash
        pip install -r requirements.txt
        ```

3.  **Configure the bot:**

    *   Rename the `config.example.yaml` file to `config.yaml`.
    *   Open the `config.yaml` file and enter your exchange API keys, Telegram bot token, and other settings.

## 3. Usage

Once the bot is running, you can access the Streamlit interface by navigating to `http://localhost:8501` in your web browser.

### 3.1. Training the Model

1.  In the Streamlit sidebar, click the "Train Models" button.
2.  The bot will train the AI model using the data specified in the `config.yaml` file.
3.  Once the training is complete, you will see a success message in the sidebar.

### 3.2. Running a Backtest

1.  In the Streamlit sidebar, click the "Run Backtest" button.
2.  The bot will run a backtest using the trained model and the historical data.
3.  The backtest results, including the total return, Sharpe ratio, and other metrics, will be displayed in the main content area.

### 3.3. Live Trading

1.  In the Streamlit sidebar, click the "Start Bot" button.
2.  The bot will start live trading on the exchange specified in the `config.yaml` file.
3.  You can monitor the bot's performance in the "Live Trading" section of the main content area.

## 4. Project Structure

The project is organized into the following directories and files:

```
.
├── Dockerfile
├── DOCUMENTATION.md
├── README.md
├── config.yaml
├── requirements.txt
├── scripts
│   ├── __init__.py
│   ├── app.py
│   ├── backtest.py
│   ├── bot.py
│   ├── exchange.py
│   ├── inference
│   │   ├── __init__.py
│   │   ├── ai_engine.py
│   │   └── models
│   │       ├── __init__.py
│   │       ├── anomaly_detection.py
│   │       └── sequential.py
│   ├── logger.py
│   ├── notifier.py
│   └── training
│       ├── __init__.py
│       └── train_sequential.py
└── tests
    ├── __init__.py
    └── test_ai_engine.py
```

*   **`Dockerfile`:** Defines the Docker image for the project.
*   **`DOCUMENTATION.md`:** This file.
*   **`README.md`:** Provides a brief overview of the project and instructions on how to get started.
*   **`config.yaml`:** The main configuration file for the bot.
*   **`requirements.txt`:** Lists the Python dependencies for the project.
*   **`scripts/`:** Contains the main Python source code for the bot.
*   **`scripts/app.py`:** The main entry point for the Streamlit application.
*   **`scripts/backtest.py`:** Implements the backtesting functionality.
*   **`scripts/bot.py`:** Contains the core logic for the trading bot.
*   **`scripts/exchange.py`:** Implements the interface for interacting with cryptocurrency exchanges.
*   **`scripts/inference/`:** Contains the code for making predictions with the AI model.
*   **`scripts/inference/ai_engine.py`:** The main entry point for the AI engine.
*   **`scripts/inference/models/`:** Contains the AI model definitions.
*   **`scripts/logger.py`:** Implements the logging functionality.
*   **`scripts/notifier.py`:** Implements the notification functionality.
*   **`scripts/training/`:** Contains the code for training the AI model.
*   **`tests/`:** Contains the unit tests for the project.

## 5. AI Model

The AI model is a crucial component of the trading bot. It is responsible for analyzing market data and predicting future price movements.

### 5.1. Model Architecture

The bot uses a sequential model, which can be either a Long Short-Term Memory (LSTM) or a Transformer network. These models are well-suited for time-series data, as they can learn to recognize patterns and dependencies over time.

### 5.2. Training

The model is trained on historical price data, which is fetched from the exchange specified in the `config.yaml` file. The training process involves feeding the model a sequence of historical data and asking it to predict the next data point in the sequence. The model's predictions are then compared to the actual data, and the model's parameters are adjusted to minimize the error.

### 5.3. Inference

Once the model is trained, it can be used to make predictions on new data. In live trading, the bot continuously feeds the model the latest market data and uses the model's predictions to make trading decisions. If the model predicts that the price of a cryptocurrency is likely to go up, the bot will buy it. If the model predicts that the price is likely to go down, the bot will sell it.

## 6. Configuration

The `config.yaml` file is the main configuration file for the bot. It is divided into several sections, which are explained below:

### 6.1. `app_config`

*   **`trade_interval_minutes`:** The interval, in minutes, at which the bot will execute trades.
*   **`allow_mock_orders`:** If set to `true`, the bot will simulate trades instead of executing them on a real exchange. This is useful for testing the bot without risking real money.
*   **`backtesting_enabled`:** If set to `true`, the bot will run a backtest on startup.
*   **`train_model_enabled`:** If set to `true`, the bot will train the AI model on startup.
*   **`start_bot_enabled`:** If set to `true`, the bot will start live trading on startup.

### 6.2. `exchange_config`

*   **`exchange`:** The name of the exchange to use. The bot currently supports `binance` and `coinbase`.
*   **`api_key`:** Your API key for the exchange.
*   **`api_secret`:** Your API secret for the exchange.
*   **`api_passphrase`:** Your API passphrase for the exchange (only required for some exchanges).
*   **`symbol`:** The trading symbol to use, in the format `BASE/QUOTE` (e.g., `BTC/USDT`).

### 6.3. `ai_config`

*   **`model_name`:** The name of the AI model to use. The bot currently supports `lstm` and `transformer`.
*   **`input_features`:** A list of the features to use as input to the model. The available features are `open`, `high`, `low`, `close`, and `volume`.
*   **`output_feature`:** The feature that the model should predict. This should be one of the features in the `input_features` list.
*   **`sequence_length`:** The number of historical data points to use as input to the model.
*   **`batch_size`:** The number of samples to use in each training batch.
*   **`num_epochs`:** The number of times to iterate over the entire training dataset.

### 6.4. `prometheus_config`

*   **`enabled`:** If set to `true`, the bot will expose a Prometheus metrics endpoint.
*   **`port`:** The port on which to expose the Prometheus metrics endpoint.

### 6.5. `telegram_config`

*   **`enabled`:** If set to `true`, the bot will send notifications to a Telegram bot.
*   **`token`:** Your Telegram bot token.
*   **`chat_id`:** The ID of the chat to which the bot should send notifications.

## 7. Development

This section provides instructions on how to contribute to the project.

### 7.1. Running the Tests

To run the unit tests, use the following command:

```bash
pytest
```

### 7.2. Adding a New Exchange

To add a new exchange, you will need to create a new file in the `scripts/exchange/` directory that implements the `Exchange` class. You will also need to add the new exchange to the `get_exchange` function in the `scripts/exchange/__init__.py` file.

### 7.3. Adding a New AI Model

To add a new AI model, you will need to create a new file in the `scripts/inference/models/` directory that implements the `SequentialModel` class. You will also need to add the new model to the `get_model` function in the `scripts/inference/ai_engine.py` file.
