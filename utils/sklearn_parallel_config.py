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
LOKY_MAX_CPU_COUNT=1 по умолчанию, чтобы не плодить воркеры.

Когда INFOBOT_SKLEARN_PARALLEL не включён: подменяем Parallel на подкласс с
backend='sequential'. Все задачи выполняются в одном процессе, конфиг всегда
прокидывается — UserWarning про delayed/Parallel не возникает.
"""
from __future__ import annotations

import os
import re
import sys
import warnings

# Исключаем воркеры loky — не создаём дочерние процессы (предупреждение идёт из воркеров).
# По умолчанию ВСЕГДА 1 воркер, чтобы полностью убрать спам UserWarning про delayed/Parallel.
# Чтобы снова включить параллелизм: INFOBOT_SKLEARN_PARALLEL=1 и LOKY_MAX_CPU_COUNT=N.
_use_parallel = os.environ.get("INFOBOT_SKLEARN_PARALLEL", "").strip().lower() in ("1", "true", "yes", "on")
if not _use_parallel:
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
# Импорт sklearn.utils.parallel выполняется до патча — внутренний код sklearn
# при последующих «from joblib import Parallel, delayed» должен получать уже наши классы.
# Поэтому конфиг обязан загружаться первым (app.py, ai.py, bots.py + sys.path).
from sklearn.utils.parallel import Parallel as _SklearnParallel
from sklearn.utils.parallel import delayed

# При отключённом параллелизме принудительно backend='sequential': все задачи в одном процессе,
# конфиг всегда прокидывается, UserWarning про delayed/Parallel не возникает (нет воркеров).
if not _use_parallel:

    class Parallel(_SklearnParallel):
        def __init__(self, *args, **kwargs):
            if "backend" not in kwargs:
                kwargs["backend"] = "sequential"
            super().__init__(*args, **kwargs)
else:
    Parallel = _SklearnParallel

setattr(joblib, "Parallel", Parallel)
setattr(joblib, "delayed", delayed)

# Надёжное подавление UserWarning: фильтр + обёртка warn (на случай вызова до фильтра или из воркеров).
_sklearn_delayed_msg = "`sklearn.utils.parallel.delayed` should be used with "
warnings.filterwarnings(
    "ignore",
    message=re.escape(_sklearn_delayed_msg) + r".*",
    category=UserWarning,
)


def _wrap_warn():
    """Подавляет предупреждение sklearn delayed/Parallel даже при вызове до filterwarnings."""
    _orig_warn = warnings.warn

    def _warn(*args, **kwargs):
        if args:
            msg = args[0] if isinstance(args[0], str) else getattr(args[0], "args", (None,))[0]
            if msg and _sklearn_delayed_msg in str(msg):
                return
        _orig_warn(*args, **kwargs)

    warnings.warn = _warn


_wrap_warn()

__all__ = ["Parallel", "delayed"]
