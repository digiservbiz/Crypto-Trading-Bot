import telegram
import yaml

class Notifier:
    def __init__(self, config):
        with open('config.yaml', 'r') as f:
            keys = yaml.safe_load(f)['telegram']

        self.bot = telegram.Bot(token=keys['token'])
        self.chat_id = keys['chat_id']

    def send_message(self, message):
        self.bot.send_message(chat_id=self.chat_id, text=message)

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    notifier = Notifier(config)
    notifier.send_message('Hello from the trading bot!')
