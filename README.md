# Crypto Trading Bot

This project is a cryptocurrency trading bot that uses a machine learning model to predict price movements and execute trades.

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
