"""
Вариант 1: везде sklearn.utils.parallel (Parallel + delayed).

Проблема: sklearn предупреждает, что delayed должен использоваться вместе
с Parallel из sklearn, а не из joblib. Иначе не прокидываются set_config(),
threadpoolctl и т.д.

Мы не вызываем Parallel/delayed явно — их используют cross_val_score,
RandomForest, GradientBoosting и т.д. внутри sklearn. Они импортируют из joblib.

Решение: до любого импорта sklearn подменяем в модуле joblib:
  joblib.Parallel   → sklearn.utils.parallel.Parallel
  joblib.delayed    → sklearn.utils.parallel.delayed

Воркеры joblib (loky) — отдельные процессы, не выполняют наш патч.
Подавляем только этот UserWarning (PYTHONWARNINGS для воркеров, filterwarnings — для главного).

На другом ПК (например Win11) при тех же версиях возможны отличия в сборках/поведении.
Если локально всё ок, а на втором ПК — спам UserWarning / lock / FakeTensor:
  - Запусти на обоих: python scripts/diagnose_env_diff.py > env_*.txt и сравни.
  - На проблемном ПК задай INFOBOT_SKLEARN_SINGLE_THREAD=1 (принудительно без воркеров loky).

Импортировать при старте (bots.py, app.py, ai.py) до импорта sklearn.
"""
from __future__ import annotations

import os
import sys
import warnings

# Принудительно: воркеры loky наследуют env и не выполняют патч.
_cur = os.environ.get("PYTHONWARNINGS", "")
_sk = "ignore::UserWarning:sklearn.utils.parallel"
if _sk not in _cur:
    os.environ["PYTHONWARNINGS"] = _sk if not _cur.strip() else f"{_cur},{_sk}"

# Исключаем воркеры loky — не создаём дочерние процессы:
# - INFOBOT_SKLEARN_SINGLE_THREAD=1: принудительно;
# - cpu_count <= 4 (например Win11 мини-ПК): авто, т.к. на малых ядрах чаще спам/lock.
_n = os.cpu_count()
if os.environ.get("INFOBOT_SKLEARN_SINGLE_THREAD", "").strip().lower() in ("1", "true", "yes", "on"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
elif (_n is not None and _n <= 4) and not os.environ.get("LOKY_MAX_CPU_COUNT", "").strip():
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# Первый запуск sklearn→scipy может занимать 5–15+ с (компиляция/загрузка). Чтобы не казалось, что «завис»:
if os.environ.get("INFOBOT_QUIET_STARTUP", "").lower() not in ("1", "true", "yes"):
    try:
        sys.stderr.write("[InfoBot] Loading sklearn/scipy...\n")
        sys.stderr.flush()
    except Exception:
        pass

import joblib
from sklearn.utils.parallel import Parallel, delayed

setattr(joblib, "Parallel", Parallel)
setattr(joblib, "delayed", delayed)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*delayed.*Parallel.*joblib.*",
)
