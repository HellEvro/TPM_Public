# -*- coding: utf-8 -*-
"""
Хранение списков фильтров монет (белый и чёрный) в data/coin_filters.json.

Система работает только с этим файлом, БД для фильтров не используется.

Миграция при запуске: списки читаются только из data/bots_data.db (проект), пишутся в data/coin_filters.json.
Опционально: python -m bot_engine.coin_filters_config --from-db "путь\\к\\другой\\bots_data.db" — разовый перенос из другого файла БД.
"""
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger('Bots.CoinFiltersConfig')

_DEFAULT = {'whitelist': [], 'blacklist': [], 'scope': 'all'}
_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')


def _normalize_item(item: Union[str, Dict]) -> Dict[str, Any]:
    """Приводит элемент к виду { symbol, added_at, updated_at }."""
    if isinstance(item, str):
        return {'symbol': item.strip().upper(), 'added_at': None, 'updated_at': None}
    if isinstance(item, dict) and item.get('symbol'):
        return {
            'symbol': str(item['symbol']).strip().upper(),
            'added_at': item.get('added_at'),
            'updated_at': item.get('updated_at'),
        }
    return None


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


def load_coin_filters_from_db_file(db_path: str) -> Dict[str, Any]:
    """
    Загружает whitelist/blacklist/scope из указанного файла bots_data.db (symbol, added_at, updated_at).
    """
    out = {'whitelist': [], 'blacklist': [], 'scope': 'all'}
    path = Path(db_path).resolve()
    if not path.exists():
        return out
    try:
        conn = sqlite3.connect(str(path), timeout=10.0)
        try:
            cur = conn.cursor()
            cur.execute("SELECT symbol, added_at, updated_at FROM coin_filters_whitelist ORDER BY symbol")
            out['whitelist'] = [{'symbol': row[0], 'added_at': row[1], 'updated_at': row[2]} for row in cur.fetchall()]
            cur.execute("SELECT symbol, added_at, updated_at FROM coin_filters_blacklist ORDER BY symbol")
            out['blacklist'] = [{'symbol': row[0], 'added_at': row[1], 'updated_at': row[2]} for row in cur.fetchall()]
            cur.execute("SELECT value FROM auto_bot_config WHERE key = 'scope'")
            row = cur.fetchone()
            if row and row[0] in ('all', 'whitelist', 'blacklist'):
                out['scope'] = row[0]
        finally:
            conn.close()
    except Exception as e:
        logger.warning("Загрузка фильтров из файла БД %s: %s", path, e)
    return out


def run_coin_filters_migration_once() -> bool:
    """
    Миграция списков фильтров из БД в data/coin_filters.json.

    Выполняется, когда файла coin_filters.json ещё нет (маркер не учитываем —
    если файла нет, переносим из БД и создаём файл). После успешной записи ставим маркер.

    Returns:
        True, если миграция была выполнена в этом запуске; False, если файл уже есть или ошибка.
    """
    path_json = _path_json()
    if path_json.exists():
        return False
    
    # Файла нет — создаём: сначала пробуем напрямую записать пустой файл (без импорта БД),
    # затем пытаемся загрузить данные из БД если она доступна
    try:
        w, b, s = [], [], 'all'
        
        # Сначала создаём файл с пустыми списками (быстро, без зависаний)
        data_dir = path_json.parent
        try:
            os.makedirs(data_dir, exist_ok=True)
        except OSError:
            pass
        
        # Пробуем загрузить из БД (но не блокируем если БД недоступна)
        db_path = data_dir / 'bots_data.db'
        if db_path.exists():
            try:
                data = load_coin_filters_from_db_file(str(db_path))
                w, b, s = data.get('whitelist', []), data.get('blacklist', []), data.get('scope', 'all')
            except Exception as e:
                logger.debug("Миграция: не удалось прочитать data/bots_data.db: %s", e)
        
        # Записываем файл (даже если списки пустые)
        ok = save_coin_filters(whitelist=w, blacklist=b, scope=s)
        if not ok:
            logger.warning("Миграция: не удалось записать data/coin_filters.json — повторится при следующем запуске")
            return False
        
        _path_sentinel().touch()
        if w or b or s != 'all':
            logger.info(
                "✅ Миграция фильтров монет: списки перенесены из БД в data/coin_filters.json (whitelist=%s, blacklist=%s, scope=%s)",
                len(w), len(b), s,
            )
        else:
            logger.info("✅ Миграция фильтров монет: создан data/coin_filters.json (пустые списки)")
        logger.info("✅ Путь к файлу фильтров: %s", path_json.resolve())
        return True
    except Exception as e:
        logger.warning("Миграция фильтров из БД в JSON пропущена: %s", e)
        return False


def _normalize_list(items: list) -> List[Dict[str, Any]]:
    """Приводит список (строк или объектов) к списку { symbol, added_at, updated_at }."""
    result = []
    for item in items if isinstance(items, list) else []:
        n = _normalize_item(item)
        if n and n['symbol']:
            result.append(n)
    return result


def load_coin_filters() -> Dict[str, Any]:
    """
    Загружает фильтры из data/coin_filters.json.
    whitelist/blacklist — список объектов { symbol, added_at, updated_at }.
    """
    with _lock:
        path = _path_json()
        if not path.exists():
            return _DEFAULT.copy()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            out = {
                'whitelist': _normalize_list(data.get('whitelist', [])),
                'blacklist': _normalize_list(data.get('blacklist', [])),
                'scope': data.get('scope', 'all'),
            }
            if out['scope'] not in ('all', 'whitelist', 'blacklist'):
                out['scope'] = 'all'
            return out
        except Exception as e:
            logger.error("Ошибка загрузки фильтров из %s: %s", path, e)
            return _DEFAULT.copy()


def _merge_list_with_dates(current_list: List[Dict], new_items: List) -> List[Dict]:
    """Объединяет новый список с текущим: сохраняет added_at/updated_at для существующих символов, для новых — now."""
    new_normalized = _normalize_list(new_items)
    by_symbol = {item['symbol']: item for item in current_list}
    now = _now_iso()
    for item in new_normalized:
        sym = item['symbol']
        if sym in by_symbol:
            by_symbol[sym] = {
                'symbol': sym,
                'added_at': by_symbol[sym].get('added_at') or item.get('added_at') or now,
                'updated_at': item.get('updated_at') or now,
            }
        else:
            by_symbol[sym] = {
                'symbol': sym,
                'added_at': item.get('added_at') or now,
                'updated_at': item.get('updated_at') or now,
            }
    return list(by_symbol.values())


def save_coin_filters(
    whitelist: Optional[List] = None,
    blacklist: Optional[List] = None,
    scope: Optional[str] = None,
) -> bool:
    """
    Сохраняет списки фильтров (белый, чёрный) и scope в data/coin_filters.json.
    Элементы — строки или объекты { symbol, added_at?, updated_at? }; при сохранении приводятся к объектам с датами.
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
            current['whitelist'] = _merge_list_with_dates(current['whitelist'], whitelist)
        if blacklist is not None:
            current['blacklist'] = _merge_list_with_dates(current['blacklist'], blacklist)
        if scope is not None and scope in ('all', 'whitelist', 'blacklist'):
            current['scope'] = scope

        path_json = _path_json()
        try:
            with open(path_json, 'w', encoding='utf-8') as f:
                json.dump(current, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.error("Ошибка записи %s: %s", path_json, e)
            return False
        return True


def main():
    """Миграция из указанного файла БД в data/coin_filters.json. Пример: python -m bot_engine.coin_filters_config --from-db \"e:\\Downloads\\bots_data.db\" """
    import argparse
    parser = argparse.ArgumentParser(description="Перенос списков фильтров из bots_data.db в data/coin_filters.json")
    parser.add_argument("--from-db", required=True, metavar="PATH", help="Путь к файлу bots_data.db (например e:\\Downloads\\bots_data.db)")
    args = parser.parse_args()
    path_db = Path(args.from_db).resolve()
    if not path_db.exists():
        print(f"Ошибка: файл не найден: {path_db}")
        return 1
    data = load_coin_filters_from_db_file(str(path_db))
    if not data.get("whitelist") and not data.get("blacklist"):
        print("В указанной БД нет записей в whitelist/blacklist.")
    ok = save_coin_filters(whitelist=data["whitelist"], blacklist=data["blacklist"], scope=data["scope"])
    if not ok:
        print("Ошибка записи data/coin_filters.json")
        return 1
    path_json = _path_json().resolve()
    print(f"Готово: whitelist={len(data['whitelist'])}, blacklist={len(data['blacklist'])}, scope={data['scope']}")
    print(f"Файл: {path_json}")
    return 0


if __name__ == "__main__":
    exit(main())
