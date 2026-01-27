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

Тогда весь код (в т.ч. внутри sklearn) при использовании joblib получит
вариант 1 — полностью через sklearn.

Воркеры joblib (loky) — отдельные процессы, не выполняют наш патч.
PYTHONWARNINGS наследуется дочерними процессами; filterwarnings — для главного.
Подавляем только этот UserWarning, остальные логи не трогаем.

Импортировать при старте (bots.py, app.py, ai.py) до импорта sklearn.
"""
from __future__ import annotations

import os
import warnings

# До импорта sklearn. Воркеры loky наследуют env и не выполняют патч.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:sklearn.utils.parallel")

import joblib
from sklearn.utils.parallel import Parallel, delayed

setattr(joblib, "Parallel", Parallel)
setattr(joblib, "delayed", delayed)

# В главном процессе (воркеры — через PYTHONWARNINGS).
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
