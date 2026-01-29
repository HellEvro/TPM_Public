# -*- coding: utf-8 -*-
"""
Утилита управления сборкой мусора (GC) — аналог явного освобождения памяти, как в Java.

Использует встроенный модуль gc Python (reference counting + cyclic GC).
Рекомендуется вызывать force_collect() после тяжёлых операций (обучение моделей,
обработка больших датасетов, циклы по множеству символов и т.д.).
Для AI-пайплайна используйте force_collect_full() — GC + очистка кэша PyTorch/CUDA.
"""

import gc as _gc
from typing import Optional

# Поколения GC (как в Python): 0 — молодые, 1 — средние, 2 — все (полная сборка)
GENERATION_YOUNG = 0
GENERATION_MIDDLE = 1
GENERATION_FULL = 2


def force_collect(generation: int = GENERATION_FULL) -> int:
    """
    Принудительная сборка мусора. По умолчанию — полная (все поколения).

    :param generation: 0 — только молодые объекты, 1 — 0+1, 2 — полная сборка (рекомендуется)
    :return: количество собранных объектов (для логирования при необходимости)
    """
    return _gc.collect(generation)


def _empty_torch_cuda_cache() -> None:
    """Очищает кэш CUDA PyTorch, если torch доступен и CUDA используется."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def force_collect_full(generation: int = GENERATION_FULL) -> int:
    """
    Полная очистка памяти: GC + освобождение кэша PyTorch/CUDA (если есть).
    Рекомендуется вызывать после тяжёлых AI-операций (обучение, инференс по батчам).

    :param generation: поколение GC (по умолчанию 2 — полная сборка)
    :return: количество собранных объектов gc
    """
    n = _gc.collect(generation)
    _empty_torch_cuda_cache()
    return n


def collect_if_enabled(generation: int = GENERATION_FULL) -> Optional[int]:
    """
    Вызвать сборку мусора только если GC не отключён (gc.isenabled()).

    :param generation: поколение (по умолчанию 2 — полная)
    :return: количество собранных объектов или None, если GC отключён
    """
    if _gc.isenabled():
        return _gc.collect(generation)
    return None


def get_threshold() -> tuple:
    """Текущие пороги GC (gen0, gen1, gen2). Для диагностики."""
    return _gc.get_threshold()


def set_threshold(gen0: int, gen1: Optional[int] = None, gen2: Optional[int] = None) -> None:
    """
    Настроить пороги срабатывания GC (по умолчанию 700, 10, 10).

    Более низкие значения — чаще сборка, меньше пиков памяти, чуть больше CPU.
    Вызывать только при обоснованной необходимости.
    """
    if gen1 is None:
        gen1 = 10
    if gen2 is None:
        gen2 = 10
    _gc.set_threshold(gen0, gen1, gen2)


def disable() -> None:
    """Отключить автоматический циклический GC (не трогает reference counting)."""
    _gc.disable()


def enable() -> None:
    """Включить автоматический циклический GC."""
    _gc.enable()
