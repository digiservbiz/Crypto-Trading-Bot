#!/bin/bash

# Build the Docker image
docker build -t crypto-trading-bot .

# Run the Docker container
docker run -p 8501:8501 crypto-trading-bot
