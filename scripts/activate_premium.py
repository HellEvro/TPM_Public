"""
Активация InfoBot AI Premium лицензии

Этот скрипт помогает пользователям активировать их премиум лицензию.
"""

import sys
sys.path.append('.')

import os
from pathlib import Path


def activate_premium_license():
    """Активирует премиум лицензию"""
    
    print("=" * 60)
    print("InfoBot AI Premium - License Activation")
    print("=" * 60)
    print()
    
    # Получаем hardware ID
    try:
        # В режиме разработки показываем hardware ID
        if os.getenv('AI_DEV_MODE') == '1':
            print("[DEV MODE] Development mode active - no license needed")
            print()
            print("To use AI modules in dev mode:")
            print("  1. set AI_DEV_MODE=1")
            print("  2. Edit bot_config.py: AI_ENABLED = True")
            print("  3. python bots.py")
            print()
            return
        
        # Пытаемся импортировать систему лицензий
        try:
            sys.path.insert(0, 'InfoBot_AI_Premium')
            from license.hardware_id import get_hardware_id, get_short_hardware_id
            from license.license_manager import LicenseManager
            
            hw_id = get_short_hardware_id()
            print(f"Hardware ID: {hw_id}")
            print()
            print("Please provide this Hardware ID when purchasing your license.")
            print()
            
        except ImportError:
            print("[ERROR] Premium license system not found")
            print()
            print("InfoBot AI Premium is not installed.")
            print()
            print("Options:")
            print("  1. Use development mode (free, for testing):")
            print("     set AI_DEV_MODE=1")
            print()
            print("  2. Purchase a license:")
            print("     Visit: https://infobot.ai/premium")
            print()
            return
        
        # Запрашиваем ключ активации
        print("Enter your activation key (format: XXXX-XXXX-XXXX-XXXX)")
        print("or press Enter to cancel:")
        print()
        
        activation_key = input("Activation key: ").strip()
        
        if not activation_key:
            print()
            print("Activation cancelled")
            return
        
        # Проверяем формат ключа
        if len(activation_key.replace('-', '')) != 16:
            print()
            print("[ERROR] Invalid activation key format")
            print("Expected: XXXX-XXXX-XXXX-XXXX")
            return
        
        print()
        print("Activating license...")
        print()
        
        # TODO: В production здесь будет запрос к серверу активации
        # Сейчас для разработки создаем локальную лицензию
        
        print("[INFO] Online activation not available yet")
        print("[INFO] Creating local developer license...")
        print()
        
        # Создаем тестовую лицензию для разработки
        manager = LicenseManager()
        
        full_hw_id = get_hardware_id()
        
        license = manager.generate_license(
            user_email='developer@local',
            license_type='developer',
            hardware_id=full_hw_id
        )
        
        # Сохраняем файл лицензии
        license_path = Path('license.lic')
        
        with open(license_path, 'wb') as f:
            f.write(license['encrypted_license'])
        
        print("[OK] License activated successfully!")
        print()
        print(f"License type: {license['license_data']['type']}")
        print(f"Expires: {license['license_data']['expires_at']}")
        print(f"License file: {license_path}")
        print()
        print("Next steps:")
        print("  1. Edit bot_config.py:")
        print("     AI_ENABLED = True")
        print("     AI_ANOMALY_DETECTION_ENABLED = True")
        print()
        print("  2. Restart the bot:")
        print("     python bots.py")
        print()
        print("  3. Check logs for:")
        print("     [AI] License: developer")
        print("     [AI] Anomaly Detector loaded")
        print()
    
    except Exception as e:
        print(f"[ERROR] Activation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    activate_premium_license()

