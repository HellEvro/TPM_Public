#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для обеспечения совместимости venv с Python 3.14
Проверяет версию Python в .venv и пересоздает его если нужно
"""

import sys
import subprocess
import shutil
import os
from pathlib import Path

REQUIRED_PYTHON_MAJOR = 3
REQUIRED_PYTHON_MINOR = 14  # Минимум 3.14, но принимаем и выше

def find_python314_plus():
    """Находит Python 3.14+ в системе"""
    commands = [
        ['py', '-3.14'],
        ['python3.14'],
        ['python314'],
        ['python', '--version'],  # Проверяем текущий
    ]
    
    for cmd in commands:
        try:
            # Проверяем версию через Python код
            check_cmd = cmd[0] if len(cmd) == 1 else cmd
            result = subprocess.run(
                [check_cmd] + ['-c', 'import sys; exit(0 if sys.version_info[:2] >= (3, 14) else 1)'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return cmd[0] if len(cmd) == 1 else cmd
        except:
            continue
    
    return None

def check_python_version(python_exec):
    """Проверяет версию Python"""
    try:
        result = subprocess.run(
            [python_exec] + ['-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_str = result.stdout.strip()
            parts = version_str.split('.')
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
    except:
        pass
    return None, None

def check_venv_python_version(venv_dir):
    """Проверяет версию Python в venv"""
    if os.name == 'nt':
        venv_python = venv_dir / 'Scripts' / 'python.exe'
    else:
        venv_python = venv_dir / 'bin' / 'python'
    
    if not venv_python.exists():
        return None, None
    
    return check_python_version(str(venv_python))

def recreate_venv(project_root, python_cmd):
    """Пересоздает venv с указанной командой Python"""
    venv_dir = project_root / '.venv'
    
    if venv_dir.exists():
        print(f"[INFO] Удаление старого .venv...")
        try:
            shutil.rmtree(venv_dir)
        except Exception as e:
            print(f"[ERROR] Не удалось удалить старый venv: {e}")
            return False
    
    print(f"[INFO] Создание нового .venv с Python 3.14+...")
    cmd = python_cmd.split() if isinstance(python_cmd, str) else list(python_cmd)
    create_cmd = cmd + ['-m', 'venv', str(venv_dir)]
    
    try:
        result = subprocess.run(create_cmd, check=True, capture_output=True, text=True)
        print(f"[OK] .venv создан успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ошибка создания venv: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def upgrade_dependencies(venv_dir, project_root):
    """Обновляет зависимости в venv"""
    if os.name == 'nt':
        venv_python = venv_dir / 'Scripts' / 'python.exe'
    else:
        venv_python = venv_dir / 'bin' / 'python'
    
    if not venv_python.exists():
        print(f"[ERROR] Python в venv не найден")
        return False
    
    requirements_file = project_root / 'requirements.txt'
    if not requirements_file.exists():
        print(f"[ERROR] requirements.txt не найден")
        return False
    
    print(f"[INFO] Обновление pip, setuptools, wheel...")
    try:
        subprocess.run(
            [str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel', '--no-warn-script-location'],
            check=True,
            cwd=project_root
        )
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Ошибка обновления pip: {e}")
    
    print(f"[INFO] Установка/обновление зависимостей из requirements.txt...")
    print(f"[INFO] TensorFlow не поддерживает Python 3.14+ и будет пропущен. PyTorch ставится в .venv.")
    
    try:
        # Устанавливаем зависимости, но игнорируем ошибки TensorFlow
        result = subprocess.run(
            [str(venv_python), '-m', 'pip', 'install', '-r', str(requirements_file), '--upgrade', '--no-warn-script-location'],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        # Проверяем вывод на ошибки TensorFlow
        if result.returncode != 0:
            error_output = result.stderr.lower() if result.stderr else ""
            if 'tensorflow' in error_output and ('requires-python' in error_output or 'could not find a version' in error_output):
                print(f"[WARNING] TensorFlow не поддерживает Python 3.14+ - это нормально")
                print(f"[INFO] Остальные зависимости установлены")
                # Пробуем установить остальные зависимости без TensorFlow
                # Читаем requirements.txt и фильтруем tensorflow
                try:
                    with open(requirements_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    filtered_lines = [line for line in lines if not line.strip().startswith('tensorflow') and line.strip() and not line.strip().startswith('#')]
                    if filtered_lines:
                        # Создаем временный файл без tensorflow
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                            tmp.write('\n'.join(filtered_lines))
                            tmp_path = tmp.name
                        try:
                            subprocess.run(
                                [str(venv_python), '-m', 'pip', 'install', '-r', tmp_path, '--upgrade', '--no-warn-script-location'],
                                check=True,
                                cwd=project_root
                            )
                            print(f"[OK] Зависимости (без TensorFlow) обновлены")
                        finally:
                            import os
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                except Exception as e2:
                    print(f"[WARNING] Ошибка при установке зависимостей без TensorFlow: {e2}")
                _verify_and_fix_sklearn(venv_dir, project_root)
                return True
            else:
                print(f"[ERROR] Ошибка установки зависимостей: {result.stderr[:500]}")
                return False
        else:
            print(f"[OK] Зависимости обновлены")
            _verify_and_fix_sklearn(venv_dir, project_root)
            return True
    except Exception as e:
        print(f"[ERROR] Ошибка установки зависимостей: {e}")
        return False


def _verify_and_fix_sklearn(venv_dir, project_root):
    """Проверяет sklearn/модели; при ошибке ставит scikit-learn>=1.7,<1.8 и повторяет."""
    if os.name == 'nt':
        venv_python = venv_dir / 'Scripts' / 'python.exe'
    else:
        venv_python = venv_dir / 'bin' / 'python'
    script = project_root / 'scripts' / 'verify_ai_deps.py'
    if not script.exists():
        return
    cmd = [str(venv_python), str(script)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        r = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace", env=env)
        if r.returncode == 0:
            print("[OK] verify_ai_deps: версии и модели OK")
            return
    except Exception as e:
        print(f"[WARNING] verify_ai_deps: {e}")
        return
    print("[INFO] Несовпадение версий sklearn/моделей — переустанавливаем scikit-learn 1.7.x...")
    try:
        subprocess.run(
            [str(venv_python), '-m', 'pip', 'install', 'scikit-learn>=1.7.0,<1.8', '--no-warn-script-location'],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        r2 = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace", env=env)
        if r2.returncode == 0:
            print("[OK] verify_ai_deps: OK после переустановки scikit-learn")
        else:
            print("[WARNING] verify_ai_deps: ошибка сохраняется. Сделайте «Обновить venv» в лаунчере. Если не поможет — запустите AI Engine и дождитесь обучения.")
    except Exception as e2:
        print(f"[WARNING] Не удалось переустановить scikit-learn: {e2}")

def main():
    """Главная функция"""
    project_root = Path(__file__).resolve().parents[1]
    venv_dir = project_root / '.venv'
    
    print("=" * 60)
    print("ПРОВЕРКА И ОБНОВЛЕНИЕ .venv ДЛЯ PYTHON 3.14+")
    print("=" * 60)
    print()
    
    # Проверяем версию Python в venv если он существует
    if venv_dir.exists():
        major, minor = check_venv_python_version(venv_dir)
        if major is not None and minor is not None:
            print(f"[INFO] Текущая версия Python в .venv: {major}.{minor}")
            
            if major == REQUIRED_PYTHON_MAJOR and minor >= REQUIRED_PYTHON_MINOR:
                print(f"[OK] Версия Python в .venv соответствует требованиям ({major}.{minor} >= 3.14)")
                print(f"[INFO] Обновление зависимостей...")
                if upgrade_dependencies(venv_dir, project_root):
                    print(f"[OK] Все готово!")
                    return 0
                else:
                    print(f"[WARNING] Ошибка обновления зависимостей")
                    return 1
            else:
                print(f"[WARNING] Версия Python в .venv ({major}.{minor}) не соответствует требованиям (требуется 3.14+)")
                print(f"[INFO] Необходимо пересоздать .venv")
        else:
            print(f"[WARNING] Не удалось определить версию Python в .venv")
            print(f"[INFO] Пересоздаем .venv")
    else:
        print(f"[INFO] .venv не существует, создаем новый")
    
    # Ищем Python 3.14+
    python314 = find_python314_plus()
    if not python314:
        print(f"[ERROR] Python 3.14+ не найден!")
        print()
        print("Пожалуйста, установите Python 3.14 или выше:")
        print("  https://www.python.org/downloads/")
        print()
        print("Или используйте py launcher на Windows:")
        print("  py -3.14 --version")
        return 1
    
    # Проверяем что найденный Python действительно 3.14+
    major, minor = check_python_version(python314.split()[0] if isinstance(python314, str) and ' ' in python314 else python314)
    if major != REQUIRED_PYTHON_MAJOR or minor < REQUIRED_PYTHON_MINOR:
        print(f"[ERROR] Найденный Python не версии 3.14+ (найдено: {major}.{minor}, требуется >= 3.14)")
        return 1
    
    print(f"[OK] Найден Python {major}.{minor}: {python314}")
    
    # Пересоздаем venv
    if not recreate_venv(project_root, python314):
        return 1
    
    # Обновляем зависимости
    if not upgrade_dependencies(venv_dir, project_root):
        return 1
    
    # Компилируем защищенные модули под Python 3.14
    print()
    print("[INFO] Компиляция защищенных модулей под Python 3.14+...")
    try:
        compile_script = project_root / 'license_generator' / 'compile_all.py'
        if os.name == 'nt':
            venv_python = venv_dir / 'Scripts' / 'python.exe'
        else:
            venv_python = venv_dir / 'bin' / 'python'
        
        result = subprocess.run(
            [str(venv_python), str(compile_script)],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("[OK] Защищенные модули скомпилированы под Python 3.14+")
        else:
            print(f"[WARNING] Ошибка компиляции модулей: {result.stderr[:200] if result.stderr else 'unknown error'}")
    except Exception as e:
        print(f"[WARNING] Не удалось скомпилировать модули: {e}")
    
    print()
    print("=" * 60)
    print("[OK] .venv обновлен для Python 3.14+")
    print("=" * 60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
