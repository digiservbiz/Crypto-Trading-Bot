import pandas as pd
import yaml
from inference.ai_engine import AIEngine
import torch

def backtest(config):
    # Load historical data
    df = pd.read_csv(config['data']['sample_data_path'])
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df.dropna(inplace=True)

    # Load the AI engine
    ai_engine = AIEngine(config)

    # Simulate the trading strategy
    positions = []
    for i in range(config['data']['lookback'], len(df)):
        # Prepare the input data for the model
        model_input_tensor = torch.FloatTensor(
            df.iloc[i-config['data']['lookback']:i][['close', 'volume', 'volatility']].values
        ).unsqueeze(0)

        model_input_dict = {
            'volume': df.iloc[i-config['data']['lookback']:i][['volume']].values
        }

        # Get the prediction from the AI engine
        prediction = ai_engine.predict(model_input_tensor, model_input_dict)

        # Simple trading strategy
        if prediction['direction'] and not prediction['is_anomaly'][-1]:
            positions.append(1)  # Buy
        else:
            positions.append(-1)  # Sell

    # Calculate strategy returns
    df['position'] = pd.Series(positions, index=df.index[config['data']['lookback']:])
    df['strategy_returns'] = df['position'].shift(1) * df['returns']

    # Print performance summary
    print("Backtest Results:")
    print(f"Total Return: {df['strategy_returns'].sum():.4f}")
    print(f"Sharpe Ratio: {df['strategy_returns'].mean() / df['strategy_returns'].std() * (252**0.5):.2f}")


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    backtest(config)
