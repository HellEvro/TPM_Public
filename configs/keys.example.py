# ШАБЛОН (только плейсхолдеры). Копируйте в configs/keys.py при первом запуске и подставьте свои ключи.
# Рабочий файл с ключами — configs/keys.py; не перезаписывайте его из этого файла.
# ==================== БЛОК 1: БИРЖИ ====================
EXCHANGES = {
    'BYBIT': {
        'enabled': True,
        'api_key': "YOUR_BYBIT_API_KEY_HERE",
        'api_secret': "YOUR_BYBIT_SECRET_KEY_HERE",
        'test_server': False,
        'position_mode': 'Hedge',
        'margin_mode': 'auto',  # auto | cross | isolated — режим маржи: auto=следовать бирже, cross/isolated=переключать при входе
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
    },
    'KUCOIN': {
        'enabled': True,
        'api_key': "YOUR_KUCOIN_FUTURES_API_KEY_HERE",
        'api_secret': "YOUR_KUCOIN_FUTURES_SECRET_KEY_HERE",
        'position_mode': 'Hedge',
        'limit_order_offset': 0.01
    }
}
# ==================== БЛОК 2: TELEGRAM ====================
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID_HERE"

