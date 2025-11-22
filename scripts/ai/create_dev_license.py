"""
Создание developer лицензии для тестирования
"""

import sys
sys.path.append('.')
sys.path.insert(0, 'InfoBot_AI_Premium')

from license.hardware_id import get_hardware_id
from license.license_manager import LicenseManager
from pathlib import Path

def create_dev_license():
    """Создает developer лицензию"""
    
    print("=" * 60)
    print("Creating Developer License")
    print("=" * 60)
    print()
    
    # Получаем hardware ID
    hw_id = get_hardware_id()
    print(f"Hardware ID: {hw_id[:16]}...")
    print()
    
    # Создаем менеджер лицензий
    manager = LicenseManager()
    
    # Генерируем developer лицензию
    license_data = manager.generate_license(
        user_email='developer@local',
        license_type='developer',
        hardware_id=hw_id
    )
    
    print("[OK] License generated")
    print(f"Type: {license_data['license_data']['type']}")
    print(f"Expires: {license_data['license_data']['expires_at']}")
    print()
    
    # Сохраняем
    license_path = Path('license.lic')
    
    with open(license_path, 'wb') as f:
        f.write(license_data['encrypted_license'])
    
    print(f"[OK] License saved: {license_path}")
    print()
    print("License is ready!")
    print()
    print("Now restart the bot:")
    print("  python bots.py")
    print()

if __name__ == '__main__':
    create_dev_license()

