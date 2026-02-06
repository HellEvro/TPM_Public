"""
Конфигурация sklearn/joblib: подмена joblib.Parallel и joblib.delayed на
sklearn.utils.parallel.* до любых импортов sklearn.

Устраняет UserWarning:
  `sklearn.utils.parallel.delayed` should be used with
  `sklearn.utils.parallel.Parallel` to make it possible to propagate
  the scikit-learn configuration of the current thread to the joblib workers.

Импортировать при старте (bots.py, app.py, ai.py) до импорта sklearn.
"""
from __future__ import annotations

import joblib
from sklearn.utils import parallel as _sk_parallel

# Подмена: все последующие использования joblib.Parallel / joblib.delayed
# (в т.ч. cross_val_score, GridSearchCV, оценки с n_jobs) будут использовать
# sklearn-версии и корректно прокидывать конфиг в воркеры.
setattr(joblib, "Parallel", _sk_parallel.Parallel)
setattr(joblib, "delayed", _sk_parallel.delayed)
