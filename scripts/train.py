import argparse
import yaml
from scripts.training.train_anomaly import train_anomaly
from scripts.training.train_garch import train_garch
from scripts.training.train_sequential import train as train_sequential

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
    elif args.model == 'sequential':
        train_sequential(config)
    else:
        print(f"Unknown model: {args.model}")

if __name__ == '__main__':
    main()
