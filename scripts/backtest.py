import pandas as pd
import yaml
from inference.ai_engine import AIEngine
import torch
import pandas_ta as ta

def backtest(config):
    # Load historical data
    df = pd.read_csv(config['data']['sample_data_path'])
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df.ta.macd(append=True)
    df.ta.rsi(append=True)
    df.ta.bbands(append=True)
    df.dropna(inplace=True)

    # Load the AI engine
    ai_engine = AIEngine(config)

    # Simulate the trading strategy
    positions = []
    features = ['close', 'volume', 'volatility', 'MACD_12_26_9', 'RSI_14', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0']
    for i in range(config['data']['lookback'], len(df)):
        # Prepare the input data for the model
        model_input_tensor = torch.FloatTensor(
            df.iloc[i-config['data']['lookback']:i][features].values
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

    # Simulate transaction costs
    transaction_costs = 0.001
    df['strategy_returns'] = df['strategy_returns'] - (df['position'].diff().abs() * transaction_costs)

    # Calculate performance metrics
    cumulative_returns = (1 + df['strategy_returns']).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std() * (252**0.5)

    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    calmar_ratio = total_return / abs(max_drawdown)

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'cumulative_returns': cumulative_returns
    }

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    results = backtest(config)
    print("Backtest Results:")
    print(f"Total Return: {results['total_return']:.4f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
