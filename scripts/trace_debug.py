"""
Расширенный модуль трейсинга исполнения кода.

Используется при ENABLE_CODE_TRACING = True в SystemConfig для
пошагового вывода выполняемых строк и записи в отдельный лог.
"""
from __future__ import annotations

import linecache
import logging
import os
import sys
import time
from datetime import datetime
from types import FrameType
from typing import Iterable, Optional

class _DefaultTraceConfig:
    """Запасная конфигурация на случай недоступности основных настроек."""

    ENABLE_CODE_TRACING = False
    TRACE_INCLUDE_KEYWORDS: Iterable[str] = []
    TRACE_SKIP_KEYWORDS: Iterable[str] = []
    TRACE_WRITE_TO_FILE = False
    TRACE_LOG_FILE = 'logs/trace.log'
    TRACE_MAX_LINE_LENGTH = 200

try:
    from bot_engine.config_loader import SystemConfig as _InitialTraceConfig
except Exception:  # pragma: no cover
    # Фолбэк, если конфиг нельзя импортировать (например, при раннем старте)
    _InitialTraceConfig = _DefaultTraceConfig  # type: ignore

TRACE_ENABLED = False
TRACE_INCLUDE_KEYWORDS: Iterable[str] = []
TRACE_SKIP_KEYWORDS: Iterable[str] = []
TRACE_WRITE_TO_FILE = False
TRACE_LOG_FILE = 'logs/trace.log'
TRACE_MAX_LINE_LENGTH = 200

_trace_logger: Optional[logging.Logger] = None
_last_logged: dict[str, float] = {}
_global_last_time: Optional[float] = None

def _apply_trace_config(config_cls) -> None:
    """Применяет значения из переданного конфига к модулю трейсинга."""
    global TRACE_ENABLED, TRACE_INCLUDE_KEYWORDS, TRACE_SKIP_KEYWORDS
    global TRACE_WRITE_TO_FILE, TRACE_LOG_FILE, TRACE_MAX_LINE_LENGTH
    global _trace_logger, _last_logged, _global_last_time

    TRACE_ENABLED = bool(getattr(config_cls, 'ENABLE_CODE_TRACING', False))
    TRACE_INCLUDE_KEYWORDS = list(getattr(config_cls, 'TRACE_INCLUDE_KEYWORDS', []))
    TRACE_SKIP_KEYWORDS = list(getattr(config_cls, 'TRACE_SKIP_KEYWORDS', []))
    TRACE_WRITE_TO_FILE = bool(getattr(config_cls, 'TRACE_WRITE_TO_FILE', False))
    TRACE_LOG_FILE = getattr(config_cls, 'TRACE_LOG_FILE', 'logs/trace.log')
    TRACE_MAX_LINE_LENGTH = int(getattr(config_cls, 'TRACE_MAX_LINE_LENGTH', 200) or 200)

    # Сбрасываем состояние логгера/кеша строк при смене конфигурации
    _trace_logger = None
    _last_logged = {}
    _global_last_time = None

def set_trace_config(config_cls) -> None:
    """
    Позволяет переопределить конфигурацию трейсинга.

    Используется, например, ai.py, чтобы задать собственные настройки,
    отличные от SystemConfig.
    """
    _apply_trace_config(config_cls)

# Применяем конфиг по умолчанию сразу после инициализации модуля
_apply_trace_config(_InitialTraceConfig)

def _should_trace_file(filename: Optional[str]) -> bool:
    """Возвращает True, если указанный файл нужно трейсить."""
    if not filename:
        return False

    normalized = filename.replace('\\', '/')

    for skip in TRACE_SKIP_KEYWORDS:
        if skip and skip in normalized:
            return False

    if TRACE_INCLUDE_KEYWORDS:
        return any(keyword in normalized for keyword in TRACE_INCLUDE_KEYWORDS)

    # Если список include пустой — по умолчанию трейсим весь проект
    return True

def _setup_trace_logger() -> logging.Logger:
    """Создает (или возвращает существующий) логгер для трейсинга."""
    global _trace_logger

    if _trace_logger:
        return _trace_logger

    logger = logging.getLogger('TRACE')
    logger.setLevel(logging.DEBUG)

    # Удаляем предыдущие хендлеры, чтобы не плодить дубликаты
    logger.handlers.clear()

    formatter = logging.Formatter('[TRACE] %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if TRACE_WRITE_TO_FILE:
        log_path = os.path.abspath(TRACE_LOG_FILE)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _trace_logger = logger
    return logger

def _format_line(frame: FrameType, delta: float) -> str:
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    func_name = frame.f_code.co_name
    short_file = os.path.basename(filename)

    try:
        line_code = linecache.getline(filename, lineno).strip()
    except Exception:
        line_code = "???"

    if len(line_code) > TRACE_MAX_LINE_LENGTH:
        line_code = line_code[:TRACE_MAX_LINE_LENGTH] + "..."

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    delta_str = f"+{delta:.3f}s" if delta > 0.01 else ""

    return f"[{timestamp}] {delta_str:>10} | {short_file}:{lineno:4} | {func_name:30} | {line_code}"

def trace_calls(frame: FrameType, event: str, arg):
    """Callback для sys.settrace: логирует каждую исполняемую строку."""
    global _global_last_time

    if event != 'line':
        return trace_calls

    filename = frame.f_code.co_filename
    if not _should_trace_file(filename):
        return trace_calls

    current_time = time.time()
    delta = (current_time - _global_last_time) if _global_last_time is not None else 0
    _global_last_time = current_time

    key = f"{filename}:{frame.f_lineno}"
    last_time_for_line = _last_logged.get(key, 0)

    # Чтобы не спамить, повторно логируем строку только если:
    #  - прошло больше 1 секунды с предыдущей строки (любой), или
    #  - прошло больше 0.3 сек с предыдущего вывода этой же строки
    if delta > 1.0 or (current_time - last_time_for_line) > 0.3:
        log_message = _format_line(frame, delta)
        logger = _setup_trace_logger()

        _last_logged[key] = current_time

    return trace_calls

def enable_trace():
    """Включает трейсинг, если он разрешен в конфигурации."""
    if not TRACE_ENABLED:
        print("[TRACE] Code tracing disabled in SystemConfig (ENABLE_CODE_TRACING=False)")
        return

    _setup_trace_logger()
    print("=" * 80)
    print("TRACE: FULL CODE TRACING ENABLED")
    print("=" * 80)
    sys.settrace(trace_calls)

def disable_trace():
    """Выключает трейсинг."""
    sys.settrace(None)
    print("TRACE: DISABLED")
