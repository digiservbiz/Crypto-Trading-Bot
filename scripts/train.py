import argparse
import yaml
from .training.train_anomaly import train_anomaly
from .training.train_garch import train_garch
from .training.train_lstm import train as train_lstm

def main():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('model', type=str, help='The model to train (anomaly, garch, or lstm)')
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if args.model == 'anomaly':
        train_anomaly(config)
    elif args.model == 'garch':
        train_garch(config)
    elif args.model == 'lstm':
        train_lstm(config)
    else:
        print(f"Unknown model: {args.model}")

if __name__ == '__main__':
    main()
