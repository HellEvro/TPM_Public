# -*- coding: utf-8 -*-
"""
Хранение списков фильтров монет (белый и чёрный) в data/coin_filters.json.

Система работает только с этим файлом, БД для фильтров не используется.
"""
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger('Bots.CoinFiltersConfig')

_DEFAULT = {'whitelist': [], 'blacklist': [], 'scope': 'all'}
_lock = threading.Lock()


def _get_data_dir() -> Path:
    """Путь к папке data (рядом с bot_engine)."""
    current = Path(__file__).resolve()
    for parent in [current.parent.parent] + list(current.parents):
        if parent and (parent / 'bots.py').exists() and (parent / 'bot_engine').exists():
            return parent / 'data'
    try:
        return current.parents[1] / 'data'
    except IndexError:
        return current.parent.parent / 'data'


def _path_json() -> Path:
    return _get_data_dir() / 'coin_filters.json'


def _path_sentinel() -> Path:
    """Файл-маркер: миграция из БД в JSON уже выполнена (один раз при следующем запуске bots.py)."""
    return _get_data_dir() / '.coin_filters_migrated_from_db'


def run_coin_filters_migration_once() -> bool:
    """
    Однократная миграция списков фильтров из БД в data/coin_filters.json.

    Вызывается при следующем запуске bots.py. Если маркер .coin_filters_migrated_from_db
    уже есть в data/ — миграция не выполняется. Иначе загружаем whitelist/blacklist/scope
    из БД, сохраняем в JSON и создаём маркер.

    Returns:
        True, если миграция была выполнена в этом запуске; False, если пропущена (уже была или ошибка).
    """
    if _path_sentinel().exists():
        return False
    try:
        from bot_engine.bots_database import get_bots_database
        db = get_bots_database()
        db_filters = db.load_coin_filters()
        w = db_filters.get('whitelist') or []
        b = db_filters.get('blacklist') or []
        s = db_filters.get('scope', 'all')
        if not w and not b and s == 'all':
            # В БД пусто — создаём пустой coin_filters.json и маркер, чтобы файл всегда был после первого запуска
            save_coin_filters(whitelist=[], blacklist=[], scope='all')
            _path_sentinel().touch()
            return False
        ok = save_coin_filters(whitelist=w, blacklist=b, scope=s)
        if not ok:
            return False
        _path_sentinel().touch()
        logger.info(
            "✅ Миграция фильтров монет: списки перенесены из БД в data/coin_filters.json (whitelist=%s, blacklist=%s, scope=%s)",
            len(w), len(b), s,
        )
        return True
    except Exception as e:
        logger.warning("Миграция фильтров из БД в JSON пропущена: %s", e)
        return False


def load_coin_filters() -> Dict[str, Any]:
    """
    Загружает фильтры из data/coin_filters.json.

    Returns:
        {'whitelist': [...], 'blacklist': [...], 'scope': 'all'|'whitelist'|'blacklist'}
    """
    with _lock:
        path = _path_json()
        if not path.exists():
            return _DEFAULT.copy()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            out = {
                'whitelist': data.get('whitelist', []),
                'blacklist': data.get('blacklist', []),
                'scope': data.get('scope', 'all'),
            }
            if not isinstance(out['whitelist'], list):
                out['whitelist'] = []
            if not isinstance(out['blacklist'], list):
                out['blacklist'] = []
            if out['scope'] not in ('all', 'whitelist', 'blacklist'):
                out['scope'] = 'all'
            return out
        except Exception as e:
            logger.error("Ошибка загрузки фильтров из %s: %s", path, e)
            return _DEFAULT.copy()


def save_coin_filters(
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = None,
    scope: Optional[str] = None,
) -> bool:
    """
    Сохраняет списки фильтров (белый, чёрный) и scope в data/coin_filters.json.

    None для аргумента означает «не менять» (обновляются только переданные поля).
    """
    data_dir = _path_json().parent
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError as e:
        logger.error("Не удалось создать папку data: %s", e)
        return False

    with _lock:
        current = load_coin_filters()
        if whitelist is not None:
            current['whitelist'] = [str(s).strip().upper() for s in whitelist if s and str(s).strip()]
        if blacklist is not None:
            current['blacklist'] = [str(s).strip().upper() for s in blacklist if s and str(s).strip()]
        if scope is not None and scope in ('all', 'whitelist', 'blacklist'):
            current['scope'] = scope

        path_json = _path_json()
        try:
            with open(path_json, 'w', encoding='utf-8') as f:
                json.dump(current, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Ошибка записи %s: %s", path_json, e)
            return False
        return True
