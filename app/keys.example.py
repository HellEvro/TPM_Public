# Устарело: пример ключей перенесён в configs/keys.example.py
# Скопируйте configs/keys.example.py -> configs/keys.py и заполните ключи

EXCHANGES = {
    'BYBIT': {
        'enabled': True,
        'api_key': "YOUR_BYBIT_API_KEY_HERE",
        'api_secret': "YOUR_BYBIT_SECRET_KEY_HERE",
        'test_server': False,
        'position_mode': 'Hedge',
        'limit_order_offset': 0.01
    },
    'BINANCE': {
        'enabled': True,
        'api_key': "YOUR_BINANCE_API_KEY_HERE",
        'api_secret': "YOUR_BINANCE_SECRET_KEY_HERE",
        'position_mode': 'Hedge',
        'limit_order_offset': 0.02
    },
    'OKX': {
        'enabled': True,
        'api_key': "YOUR_OKX_API_KEY_HERE",
        'api_secret': "YOUR_OKX_SECRET_KEY_HERE",
        'passphrase': "YOUR_OKX_PASSPHRASE_HERE",
        'position_mode': 'Hedge',
        'limit_order_offset': 0.015
    }
}

# Telegram settings
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID_HERE"

