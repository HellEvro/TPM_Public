#!/usr/bin/env python3
# -*- кодировка: utf-8 -*-
"""
Оболочка для защищённого AI лаунчера.
Вся рабочая логика находится в bot_engine/ai/_ai_launcher.pyc
"""

# ⚠️ КРИТИЧНО: Устанавливаем переменную окружения для идентификации процесса ai.py
# Это гарантирует, что функции из filters.py будут сохранять свечи в ai_data.db, а не в bots_data.db
import os
import sys
import warnings
# Подавление FutureWarning LeafSpec (PyTorch/зависимости) — до любых импортов, которые могут его вызвать
warnings.filterwarnings("ignore", category=FutureWarning, message=".*LeafSpec.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*TreeSpec.*is_leaf.*")
os.environ['INFOBOT_AI_PROCESS'] = 'true'
# Подавление UserWarning sklearn для процесса и дочерних (joblib воркеры не наследуют filters).
_pw = os.environ.get("PYTHONWARNINGS", "").strip()
_add = "ignore::UserWarning:sklearn.utils.parallel,ignore::FutureWarning"
os.environ["PYTHONWARNINGS"] = f"{_pw},{_add}" if _pw else _add
# Корень проекта в path до импорта utils — иначе sklearn_parallel_config не найдётся при запуске из другой директории
_root = os.path.dirname(os.path.abspath(__file__))
if _root and _root not in sys.path:
    sys.path.insert(0, _root)
import utils.sklearn_parallel_config  # noqa: F401 — вариант 1 до импорта sklearn


def _get_total_ram_mb():
    """Возвращает объём общей ОЗУ системы в MB или None при ошибке."""
    try:
        import psutil
        total_bytes = psutil.virtual_memory().total
        return int(total_bytes / (1024 * 1024))
    except Exception:
        pass
    if sys.platform == 'linux':
        try:
            with open('/proc/meminfo', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        return kb // 1024
        except Exception:
            pass
    return None


def _compute_memory_limit_mb():
    """
    Вычисляет лимит ОЗУ в MB. Источники (приоритет):
    1) bot_config.SystemConfig: AI_MEMORY_PCT, AI_MEMORY_LIMIT_MB
    2) Переменные окружения: AI_MEMORY_PCT, AI_MEMORY_LIMIT_MB
    Если заданы и процент, и MB — приоритет у процента.
    """
    pct_val = None
    limit_mb_val = None
    try:
        from bot_engine.config_loader import SystemConfig
        pct_val = getattr(SystemConfig, 'AI_MEMORY_PCT', 0) or 0
        limit_mb_val = getattr(SystemConfig, 'AI_MEMORY_LIMIT_MB', 0) or 0
    except Exception:
        pass
    if not pct_val and not limit_mb_val:
        pct_str = os.environ.get('AI_MEMORY_PCT', '').strip()
        if pct_str:
            try:
                pct_val = float(pct_str.replace(',', '.'))
                pct_val = max(1.0, min(100.0, pct_val))
            except ValueError:
                pct_val = None
        limit_mb_str = os.environ.get('AI_MEMORY_LIMIT_MB', '').strip()
        if limit_mb_str:
            try:
                limit_mb_val = int(limit_mb_str)
            except ValueError:
                limit_mb_val = 0
    if pct_val and pct_val > 0:
        total_mb = _get_total_ram_mb()
        if total_mb and total_mb > 0:
            limit_mb = int(total_mb * pct_val / 100.0)
            if limit_mb > 0:
                # Если задана верхняя граница в MB (например 4 ГБ) — не превышаем её
                if limit_mb_val and limit_mb_val > 0:
                    limit_mb = min(limit_mb, limit_mb_val)
                os.environ['AI_MEMORY_LIMIT_MB'] = str(limit_mb)
                return limit_mb, 'pct', total_mb, pct_val
    if limit_mb_val and limit_mb_val > 0:
        os.environ['AI_MEMORY_LIMIT_MB'] = str(limit_mb_val)
        return limit_mb_val, 'mb', None, None
    return None, None, None, None


def _apply_memory_limit_if_configured():
    """
    Ограничение потребления ОЗУ процессом ai.py (AI_MEMORY_LIMIT_MB или AI_MEMORY_PCT).
    На Linux/macOS: resource.setrlimit(RLIMIT_AS).
    На Windows: лимит задаётся через Job Object в _apply_windows_job_limits() (один Job на CPU+ОЗУ).
    Сообщения выводятся только в главном процессе.
    """
    computed, kind, total_mb, pct = _compute_memory_limit_mb()
    if computed is None:
        try:
            import multiprocessing
            if multiprocessing.current_process().name == 'MainProcess':
                sys.stderr.write(
                    "[AI] Лимит ОЗУ не задан: задайте AI_MEMORY_PCT или AI_MEMORY_LIMIT_MB в bot_config.SystemConfig или в env.\n"
                )
        except Exception:
            pass
        return
    limit_mb = computed
    try:
        import multiprocessing
        _is_main = multiprocessing.current_process().name == 'MainProcess'
    except Exception:
        _is_main = True
    if _is_main:
        if kind == 'pct' and total_mb is not None and pct is not None:
            sys.stderr.write(f"[AI] Лимит ОЗУ: {limit_mb} MB ({pct:.0f}% от {total_mb} MB)\n")
        elif kind == 'mb':
            sys.stderr.write(f"[AI] Лимит ОЗУ: {limit_mb} MB (AI_MEMORY_LIMIT_MB)\n")
    if sys.platform == 'win32':
        # На Windows лимит применится в _apply_windows_job_limits() вместе с CPU
        return
    try:
        import resource
        limit_bytes = limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, OSError) as e:
        if _is_main:
            sys.stderr.write(f"[AI] Не удалось установить лимит ОЗУ {limit_mb} MB: {e}\n")
    except Exception:
        pass


_apply_memory_limit_if_configured()

# Храним handle Job Object на Windows, чтобы лимиты CPU/ОЗУ не снимались (не закрывать handle)
_win_job_handle = []


def _apply_windows_job_limits():
    """
    Windows: один Job Object для лимита ОЗУ (ProcessMemoryLimit) и/или CPU (%).
    Читает AI_MEMORY_LIMIT_MB из env (уже установлен _apply_memory_limit_if_configured),
    AI_CPU_PCT из bot_config или env. Ручку Job не закрываем — иначе лимиты перестают действовать.
    """
    if sys.platform != 'win32':
        return
    try:
        import multiprocessing
        if multiprocessing.current_process().name != 'MainProcess':
            return
    except Exception:
        pass
    memory_mb = 0
    try:
        limit_str = os.environ.get('AI_MEMORY_LIMIT_MB', '').strip()
        if limit_str:
            memory_mb = int(limit_str)
    except ValueError:
        pass
    cpu_pct = 0
    try:
        from bot_engine.config_loader import SystemConfig
        cpu_pct = getattr(SystemConfig, 'AI_CPU_PCT', 0) or 0
    except Exception:
        pass
    if not cpu_pct:
        pct_str = os.environ.get('AI_CPU_PCT', '').strip()
        if pct_str:
            try:
                cpu_pct = int(float(pct_str.replace(',', '.')))
                cpu_pct = max(1, min(100, cpu_pct))
            except ValueError:
                pass
    if not memory_mb and not cpu_pct:
        return
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        CreateJobObjectW = kernel32.CreateJobObjectW
        CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        CreateJobObjectW.restype = wintypes.HANDLE
        AssignProcessToJobObject = kernel32.AssignProcessToJobObject
        AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        AssignProcessToJobObject.restype = wintypes.BOOL
        SetInformationJobObject = kernel32.SetInformationJobObject
        SetInformationJobObject.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p]
        SetInformationJobObject.restype = wintypes.BOOL
        job = CreateJobObjectW(None, None)
        if not job:
            return
        if not AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            kernel32.CloseHandle(job)
            return
        applied = []
        if memory_mb and memory_mb > 0:
            JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100
            JobObjectExtendedLimitInformation = 9
            limit_bytes = memory_mb * 1024 * 1024

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ('ReadOperationCount', ctypes.c_ulonglong),
                    ('WriteOperationCount', ctypes.c_ulonglong),
                    ('OtherOperationCount', ctypes.c_ulonglong),
                    ('ReadTransferCount', ctypes.c_ulonglong),
                    ('WriteTransferCount', ctypes.c_ulonglong),
                    ('OtherTransferCount', ctypes.c_ulonglong),
                ]

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ('PerProcessUserTimeLimit', ctypes.c_ulonglong),
                    ('PerJobUserTimeLimit', ctypes.c_ulonglong),
                    ('LimitFlags', wintypes.DWORD),
                    ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t),
                    ('ActiveProcessLimit', wintypes.DWORD),
                    ('Affinity', ctypes.c_size_t),
                    ('PriorityClass', wintypes.DWORD),
                    ('SchedulingClass', wintypes.DWORD),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ('IoInfo', IO_COUNTERS),
                    ('ProcessMemoryLimit', ctypes.c_size_t),
                    ('JobMemoryLimit', ctypes.c_size_t),
                    ('PeakProcessMemoryUsed', ctypes.c_size_t),
                    ('PeakJobMemoryUsed', ctypes.c_size_t),
                ]

            basic = JOBOBJECT_BASIC_LIMIT_INFORMATION()
            basic.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
            ext = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            ext.BasicLimitInformation = basic
            ext.ProcessMemoryLimit = limit_bytes
            if SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(ext)):
                applied.append(f"ОЗУ {memory_mb} MB")
        if cpu_pct and cpu_pct > 0:
            JOB_OBJECT_CPU_RATE_CONTROL_ENABLE = 0x1
            JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x4
            JobObjectCpuRateControlInformation = 15
            cpu_rate = cpu_pct * 100

            class JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ('ControlFlags', wintypes.DWORD),
                    ('CpuRate', wintypes.DWORD),
                ]

            info = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(
                ControlFlags=JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
                CpuRate=cpu_rate,
            )
            if SetInformationJobObject(job, JobObjectCpuRateControlInformation, ctypes.byref(info)):
                applied.append(f"CPU {cpu_pct}%")
        if applied:
            sys.stderr.write(f"[AI] Windows Job Object: {', '.join(applied)}\n")
            _win_job_handle.append(job)
        else:
            kernel32.CloseHandle(job)
    except Exception:
        pass


def _apply_cpu_limit_if_configured():
    """
    Ограничение загрузки CPU в % (только Windows 8+, Job Object).
    На Windows лимиты CPU и ОЗУ применяются вместе в _apply_windows_job_limits().
    """
    if sys.platform == 'win32':
        _apply_windows_job_limits()
        return
    # На не-Windows CPU не ограничиваем (только ОЗУ через setrlimit)
    pass


def _set_gpu_memory_fraction_env():
    """Выставляет AI_GPU_MEMORY_FRACTION в env из конфига (применяется при первом использовании CUDA в lstm_predictor/pytorch_setup)."""
    frac = 0
    try:
        from bot_engine.config_loader import SystemConfig
        frac = getattr(SystemConfig, 'AI_GPU_MEMORY_FRACTION', 0) or 0
    except Exception:
        pass
    if not frac:
        frac_str = os.environ.get('AI_GPU_MEMORY_FRACTION', '').strip()
        if frac_str:
            try:
                frac = float(frac_str.replace(',', '.'))
            except ValueError:
                pass
    if frac and 0 < frac <= 1:
        os.environ['AI_GPU_MEMORY_FRACTION'] = str(max(0.01, min(1.0, frac)))


_set_gpu_memory_fraction_env()
_apply_cpu_limit_if_configured()


# Проверка и автоматическая установка PyTorch ПЕРЕД импортом защищенного модуля
def _check_and_install_pytorch():
    """Проверяет наличие PyTorch и устанавливает его при необходимости"""
    try:
        import torch
        # PyTorch уже установлен
        return True
    except ImportError:
        # PyTorch не установлен, нужно установить
        import sys
        import subprocess
        import platform
        
        print("=" * 80)
        print("🔍 ПРОВЕРКА PYTORCH")
        print("=" * 80)
        print("⚠️ PyTorch не найден. Начинаю автоматическую установку...")
        print()
        
        # Определяем путь к скрипту установки
        script_dir = os.path.dirname(os.path.abspath(__file__))
        setup_script = os.path.join(script_dir, 'scripts', 'setup_python_gpu.py')
        
        if not os.path.exists(setup_script):
            print("❌ Ошибка: не найден скрипт установки PyTorch")
            print(f"   Ожидаемый путь: {setup_script}")
            print()
            print("💡 Установите PyTorch вручную:")
            print("   python scripts/setup_python_gpu.py")
            print("   или")
            print("   pip install torch torchvision torchaudio")
            return False
        
        # Запускаем скрипт установки
        print(f"🚀 Запускаю установку PyTorch через {setup_script}...")
        print()
        
        try:
            result = subprocess.run(
                [sys.executable, setup_script],
                cwd=script_dir,
                timeout=600,  # 10 минут максимум
                capture_output=False  # Показываем вывод в реальном времени
            )
            
            if result.returncode == 0:
                print()
                print("=" * 80)
                print("✅ PyTorch успешно установлен!")
                print("=" * 80)
                print()
                
                # Проверяем, что PyTorch теперь доступен
                try:
                    import torch
                    print(f"✅ PyTorch версия: {torch.__version__}")
                    if torch.cuda.is_available():
                        print(f"✅ CUDA доступна: {torch.version.cuda}")
                        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
                    else:
                        print("ℹ️ CUDA недоступна, будет использоваться CPU")
                    print()
                    return True
                except ImportError:
                    print("⚠️ PyTorch установлен, но не импортируется. Может потребоваться перезапуск.")
                    return False
            else:
                print()
                print("=" * 80)
                print("❌ Ошибка установки PyTorch")
                print("=" * 80)
                print("💡 Попробуйте установить вручную:")
                print("   python scripts/setup_python_gpu.py")
                print("   или")
                print("   pip install torch torchvision torchaudio")
                print("=" * 80)
                return False
                
        except subprocess.TimeoutExpired:
            print()
            print("❌ Установка PyTorch заняла слишком много времени (>10 минут)")
            print("💡 Попробуйте установить вручную: python scripts/setup_python_gpu.py")
            return False
        except Exception as e:
            print()
            print(f"❌ Ошибка при запуске установки PyTorch: {e}")
            print("💡 Попробуйте установить вручную: python scripts/setup_python_gpu.py")
            return False

# Выполняем проверку PyTorch перед импортом защищенного модуля
_check_and_install_pytorch()


def _run_rebuild_bot_history_from_exchange():
    """При старте ai.py подтягивает историю биржи в bot_trades_history (bots_data.db), чтобы ИИ видел сделки для обучения."""
    if os.environ.get("INFOBOT_SKIP_REBUILD_BOT_HISTORY", "").strip().lower() in ("1", "true", "yes"):
        return
    try:
        import subprocess
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _rebuild = os.path.join(_script_dir, "scripts", "rebuild_bot_history_from_exchange.py")
        if os.path.isfile(_rebuild):
            subprocess.run(
                [sys.executable, _rebuild],
                cwd=_script_dir,
                timeout=120,
                capture_output=False,
            )
    except Exception:
        pass


_run_rebuild_bot_history_from_exchange()

try:
    from utils.memory_utils import force_collect_full
    force_collect_full()
except Exception:
    pass


def _init_timeframe_from_config():
    """При старте ai.py подгружаем текущий таймфрейм из configs/bot_config.py (единый конфиг)."""
    try:
        from bot_engine.config_loader import set_current_timeframe, get_current_timeframe
        tf = get_current_timeframe()
        if tf:
            set_current_timeframe(tf)
    except Exception:
        pass


_init_timeframe_from_config()

try:
    from utils.memory_utils import force_collect_full
    force_collect_full()
except Exception:
    pass

# Настройка логирования ПЕРЕД импортом защищенного модуля (свой конфиг: ai_launcher_config)
import logging
try:
    from bot_engine.ai.ai_launcher_config import AILauncherConfig
    from utils.color_logger import setup_color_logging
    console_levels = getattr(AILauncherConfig, 'CONSOLE_LOG_LEVELS', [])
    setup_color_logging(console_log_levels=console_levels if console_levels else None)
except Exception as e:
    try:
        from utils.color_logger import setup_color_logging
        setup_color_logging()
    except Exception as setup_error:
        import sys
        sys.stderr.write(f"❌ Ошибка настройки логирования: {setup_error}\n")

from typing import TYPE_CHECKING, Any
from bot_engine.ai import _infobot_ai_protected as _protected_module


if TYPE_CHECKING:
    def main(*args: Any, **kwargs: Any) -> Any: ...


# Патч для перенаправления data_service.json в БД
def _patch_ai_system_update_data_status():
    """
    Патчит метод _update_data_status в классе AISystem для сохранения в БД вместо файла
    """
    try:
        # Импортируем helper для работы с БД
        from bot_engine.ai.data_service_status_helper import update_data_service_status_in_db

        # Получаем класс AISystem из защищенного модуля
        if hasattr(_protected_module, 'AISystem'):
            AISystem = _protected_module.AISystem

            # Сохраняем оригинальный метод (на случай если понадобится)
            original_update_data_status = AISystem._update_data_status

            # Заменяем метод на версию, которая сохраняет в БД
            def patched_update_data_status(self, **kwargs):
                """Патченная версия _update_data_status - сохраняет в БД вместо файла"""
                try:
                    update_data_service_status_in_db(**kwargs)
                except Exception as e:
                    # В случае ошибки пробуем оригинальный метод (fallback)
                    try:
                        original_update_data_status(self, **kwargs)
                    except:
                        pass

            # Применяем патч
            AISystem._update_data_status = patched_update_data_status

    except Exception as e:
        # Если патч не удался, продолжаем работу без него
        pass

# Применяем патч ПЕРЕД импортом глобальных переменных
_patch_ai_system_update_data_status()


_globals = globals()
_skip = {'__name__', '__doc__', '__package__', '__loader__', '__spec__', '__file__'}

for _key, _value in _protected_module.__dict__.items():
    if _key in _skip:
        continue
    _globals[_key] = _value

del _globals, _skip, _key, _value

try:
    from utils.memory_utils import force_collect_full
    force_collect_full()
except Exception:
    pass


if __name__ == '__main__':
    _protected_module.main()
