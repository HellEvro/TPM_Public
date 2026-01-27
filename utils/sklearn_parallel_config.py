"""
Вариант A (sklearn): Parallel и delayed — оба из sklearn.utils.parallel.

Правило: не смешивать — либо оба из sklearn, либо оба из joblib.
У нас пайплайны/модели sklearn, поэтому везде используем вариант sklearn.

Проблема: если где-то delayed из sklearn, а Parallel (или вызовы параллелизма)
идёт через joblib/внутренние вызовы — sklearn даёт UserWarning и не прокидывает
set_config() в воркеры.

Решение:
  1) До любого импорта sklearn подменяем в модуле joblib:
       joblib.Parallel  → sklearn.utils.parallel.Parallel
       joblib.delayed   → sklearn.utils.parallel.delayed
     Тогда любой код (в т.ч. внутри sklearn), делающий "from joblib import Parallel, delayed",
     после загрузки этого конфига получает оба из sklearn.

  2) В своём коде не вызывать "from joblib import Parallel" или "from joblib import delayed".
     joblib используем только для joblib.dump / joblib.load.
     Если понадобятся Parallel/delayed — брать:
       from sklearn.utils.parallel import Parallel, delayed

  3) Конфиг импортировать первым (до sklearn) в bots.py, app.py, ai.py и в модулях
     ai_trainer, parameter_quality_predictor, anomaly_detector, pattern_detector, ml_risk_predictor.

Воркеры loky — отдельные процессы, наш патч там не выполняется.
LOKY_MAX_CPU_COUNT=1 по умолчанию, PYTHONWARNINGS + filterwarnings — для подавления остаточных предупреждений.
"""
from __future__ import annotations

import os
import sys
import warnings

# Принудительно: воркеры loky наследуют env и не выполняют патч.
_cur = os.environ.get("PYTHONWARNINGS", "")
_sk = "ignore::UserWarning:sklearn.utils.parallel"
_msg = "ignore:.*delayed.*should be used with.*Parallel.*:UserWarning::"
_to_add = [x for x in (_sk, _msg) if x and x not in _cur]
if _to_add:
    _new = ",".join(_to_add) if not _cur.strip() else f"{_cur},{','.join(_to_add)}"
    os.environ["PYTHONWARNINGS"] = _new

# Исключаем воркеры loky — не создаём дочерние процессы (предупреждение идёт из воркеров).
# По умолчанию ВСЕГДА 1 воркер, чтобы полностью убрать спам UserWarning про delayed/Parallel.
# Чтобы снова включить параллелизм: INFOBOT_SKLEARN_PARALLEL=1 и LOKY_MAX_CPU_COUNT=N.
if os.environ.get("INFOBOT_SKLEARN_PARALLEL", "").strip().lower() not in ("1", "true", "yes", "on"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
else:
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

# Единый источник: для своего кода использовать "from sklearn.utils.parallel import Parallel, delayed"
__all__ = ["Parallel", "delayed"]

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*delayed.*Parallel.*joblib.*",
)
# Точные фразы из sklearn.utils.parallel (скопированы из исходников sklearn):
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*sklearn\.utils\.parallel\.delayed.*should be used with.*sklearn\.utils\.parallel\.Parallel.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*sklearn\.utils\.parallel\.Parallel.*needs to be used.*joblib\.delayed.*",
)

# Реэкспорт: для своего кода в проекте, если понадобятся Parallel/delayed —
# from utils.sklearn_parallel_config import Parallel, delayed  # оба из sklearn
__all__ = ["Parallel", "delayed"]
