# Торговый движок ботов
# Автопатч: создание/дополнение bot_config.py из bot_config.example.py при импорте
try:
    from bot_engine._patch_config import run_patch
    run_patch()
except Exception:
    pass
