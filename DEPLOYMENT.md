# Deployment Guide

This guide provides instructions on how to deploy the crypto trading bot using Docker and Docker Compose.

## Prerequisites

- Docker
- Docker Compose

## 1. Clone the repository

```bash
git clone https://github.com/your-username/crypto-trading-bot.git
cd crypto-trading-bot
```

## 2. Configure the bot

Copy the `config.yaml` file to `config.live.yaml` and update it with your live trading parameters.

```bash
cp config.yaml config.live.yaml
```

Update the `config.live.yaml` file with your exchange API keys and other trading parameters.

## 3. Build and run the bot

Use the `docker-compose.yml` file to build and run the bot, Prometheus, and Grafana services.

```bash
docker-compose up --build
```

The bot will now be running in the background. You can access the Streamlit web interface at `http://localhost:8501`, Prometheus at `http://localhost:9090`, and Grafana at `http://localhost:3000`.

## 4. Monitor the bot

Use the Grafana dashboard to monitor the bot's performance in real-time. The dashboard is pre-configured to show the bot's balance, total profit/loss, and the number of open positions.

## 5. Stop the bot

To stop the bot, press `Ctrl+C` in the terminal where you ran the `docker-compose up` command. You can also run the following command to stop the services:

```bash
docker-compose down
```
