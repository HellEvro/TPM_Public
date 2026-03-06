"""
Обновляет ai_manager.py с защищенной проверкой лицензий
Используется перед копированием в публичную версию
"""

from pathlib import Path
import re

def update_ai_manager():
    """Обновляет _check_license в ai_manager.py"""
    
    print("=" * 60)
    print("UPDATING AI_MANAGER WITH LICENSE CHECKER")
    print("=" * 60)
    print()
    
    source_file = Path('source/license_checker.py')
    target_file = Path('../bot_engine/ai/ai_manager.py')
    
    if not source_file.exists():
        print("[ERROR] Source file not found!")
        return False
    
    if not target_file.exists():
        print("[ERROR] Target file not found!")
        return False
    
    # Читаем исходник
    source_code = source_file.read_text(encoding='utf-8')
    
    # Извлекаем логику проверки
    checker_content = source_code.split('class LicenseChecker')[1]
    checker_content = checker_content.split('"""')[2] if '"""' in checker_content else checker_content
    
    # Читаем ai_manager.py
    ai_manager_code = target_file.read_text(encoding='utf-8')
    
    # Заменяем метод _check_license
    pattern = r'def _check_license\(self\):.*?(?=\n    def \w+|\nclass \w+|$)'
    replacement = '''def _check_license(self):
        """Встроенная проверка лицензии (скрытая реализация)"""
        if not AIConfig.AI_ENABLED:
            return
        
        # Проверяем .lic файл в корне
        root = Path(__file__).parent.parent.parent
        lic_files = [f for f in os.listdir(root) if f.endswith('.lic')]
        
        if not lic_files:
            self._license_valid = False
            return
        
        # Расшифровка лицензии (ключи замаскированы)
        try:
            lic_file = root / lic_files[0]
            with open(lic_file, 'rb') as f:
                d = f.read()
            
            # Ключи (обфусцированы)
            from cryptography.fernet import Fernet
            from base64 import urlsafe_b64encode
            import hmac
            import hashlib
            
            k1 = 'InfoBot' + 'AI2024'
            k2 = 'Premium' + 'License'
            k3 = 'Key_SECRET'
            sk = (k1 + k2 + k3 + '_DO_NOT_SHARE').encode()[:32]
            x = urlsafe_b64encode(sk)
            cf = Fernet(x)
            
            # Расшифровка
            dec = cf.decrypt(d)
            ld = json.loads(dec.decode())
            
            # Проверка подписи
            sk2 = 'SECRET' + '_SIGNATURE_' + 'KEY_2024_PREMIUM'
            dtv = json.dumps({k:v for k,v in ld.items() if k != 'signature'}, sort_keys=True)
            es = hmac.new(sk2.encode(), dtv.encode(), hashlib.sha256).hexdigest()
            
            if not hmac.compare_digest(ld.get('signature', ''), es):
                self._license_valid = False
                return
            
            # Проверка срока
            ea = datetime.fromisoformat(ld['expires_at'])
            if datetime.now() > ea:
                self._license_valid = False
                return
            
            # Лицензия валидна
            self._license_valid = True
            self._license_info = {
                'type': ld.get('type', 'premium'),
                'expires_at': ld['expires_at'],
                'features': ld.get('features', {
                    'anomaly_detection': True,
                    'lstm_predictor': True,
                    'pattern_recognition': True,
                    'risk_management': True,
                })
            }
            
        except Exception as e:
            self._license_valid = False
            logger.warning(f"[AI] License check failed: {e}")'''
    
    # Заменяем
    updated_code = re.sub(pattern, replacement, ai_manager_code, flags=re.DOTALL)
    
    # Сохраняем
    target_file.write_text(updated_code, encoding='utf-8')
    
    print(f"[OK] Updated: {target_file}")
    print()
    print("=" * 60)
    print("AI_MANAGER UPDATED")
    print("=" * 60)
    print()
    print("Now copy to public version:")
    print("  python ../scripts/copy_to_public.py")
    print()
    
    return True

if __name__ == '__main__':
    update_ai_manager()

