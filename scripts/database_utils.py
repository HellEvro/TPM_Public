#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилиты для программной работы с базами данных InfoBot

Возможности:
- Подключение к базам данных
- Выполнение SQL-скриптов из файлов
- Выполнение миграций
- Работа с транзакциями
"""

import os
import sys
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager

# Определяем корневую директорию проекта
ROOT = Path(__file__).parent.parent

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


class DatabaseConnection:
    """Класс для работы с подключением к базе данных"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
    
    def connect(self) -> bool:
        """Подключается к базе данных"""
        try:
            self.conn = sqlite3.connect(self.db_path, timeout=10.0)
            self.conn.row_factory = sqlite3.Row  # Возвращаем результаты как словари
            return True
        except Exception as e:
            raise ConnectionError(f"Не удалось подключиться к БД {self.db_path}: {e}")
    
    def disconnect(self):
        """Отключается от базы данных"""
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
            self.conn = None
    
    @contextmanager
    def transaction(self):
        """Контекстный менеджер для транзакций"""
        if not self.conn:
            raise RuntimeError("Нет подключения к БД")
        
        try:
            yield self.conn
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e
    
    def execute_query(self, query: str, params: Tuple = None) -> Tuple[Optional[List], Optional[str]]:
        """
        Выполняет SQL запрос
        
        Args:
            query: SQL запрос
            params: Параметры для запроса (опционально)
        
        Returns:
            (results, error_message) - результаты запроса или сообщение об ошибке
        """
        if not self.conn:
            return None, "Нет подключения к БД"
        
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Если запрос изменяет данные, коммитим
            if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER')):
                self.conn.commit()
                return [], None  # Изменяющие запросы не возвращают данные
            
            # Получаем результаты
            results = cursor.fetchall()
            # Конвертируем Row объекты в списки словарей
            rows = []
            for row in results:
                rows.append(dict(row))
            return rows, None
        except sqlite3.Error as e:
            return None, str(e)
    
    def execute_script(self, script: str, stop_on_error: bool = True) -> Tuple[bool, Optional[str], int]:
        """
        Выполняет SQL-скрипт (может содержать несколько запросов, разделенных ';')
        
        Args:
            script: SQL-скрипт (может содержать несколько запросов)
            stop_on_error: Останавливать выполнение при ошибке
        
        Returns:
            (success, error_message, executed_count) - успех, сообщение об ошибке, количество выполненных запросов
        """
        if not self.conn:
            return False, "Нет подключения к БД", 0
        
        # Разделяем скрипт на отдельные запросы
        queries = split_sql_script(script)
        
        executed_count = 0
        
        try:
            cursor = self.conn.cursor()
            
            for query in queries:
                query = query.strip()
                if not query:
                    continue
                
                try:
                    cursor.execute(query)
                    executed_count += 1
                except sqlite3.Error as e:
                    if stop_on_error:
                        self.conn.rollback()
                        return False, f"Ошибка выполнения запроса #{executed_count + 1}: {e}", executed_count
                    # Продолжаем выполнение даже при ошибке
                    print(f"Предупреждение: ошибка в запросе #{executed_count + 1}: {e}", file=sys.stderr)
            
            # Коммитим все изменения
            self.conn.commit()
            return True, None, executed_count
            
        except Exception as e:
            self.conn.rollback()
            return False, f"Ошибка выполнения скрипта: {e}", executed_count


def split_sql_script(script: str) -> List[str]:
    """
    Разделяет SQL-скрипт на отдельные запросы
    
    Args:
        script: SQL-скрипт
    
    Returns:
        Список отдельных SQL-запросов
    """
    queries = []
    current_query = []
    in_string = False
    string_char = None
    i = 0
    
    while i < len(script):
        char = script[i]
        
        # Обрабатываем строки (чтобы не разбивать по ';' внутри строк)
        if char in ("'", '"') and (i == 0 or script[i-1] != '\\'):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None
        
        # Обрабатываем комментарии
        if not in_string and char == '-' and i + 1 < len(script) and script[i+1] == '-':
            # Однострочный комментарий - пропускаем до конца строки
            while i < len(script) and script[i] != '\n':
                current_query.append(script[i])
                i += 1
            if i < len(script):
                current_query.append(script[i])
                i += 1
            continue
        
        if not in_string and char == '/' and i + 1 < len(script) and script[i+1] == '*':
            # Многострочный комментарий
            current_query.append(script[i])
            i += 1
            current_query.append(script[i])
            i += 1
            while i + 1 < len(script):
                if script[i] == '*' and script[i+1] == '/':
                    current_query.append(script[i])
                    current_query.append(script[i+1])
                    i += 2
                    break
                current_query.append(script[i])
                i += 1
            continue
        
        # Разделитель запросов (только вне строк)
        if not in_string and char == ';':
            query = ''.join(current_query).strip()
            if query:
                queries.append(query)
            current_query = []
            i += 1
            continue
        
        current_query.append(char)
        i += 1
    
    # Добавляем последний запрос, если он есть
    query = ''.join(current_query).strip()
    if query:
        queries.append(query)
    
    return queries


def load_sql_file(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Загружает SQL-файл с поддержкой разных кодировок
    
    Args:
        file_path: Путь к SQL-файлу
        encoding: Кодировка файла (по умолчанию UTF-8)
    
    Returns:
        Содержимое SQL-файла
    
    Raises:
        FileNotFoundError: Файл не найден
        UnicodeDecodeError: Не удалось прочитать файл
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SQL-файл не найден: {file_path}")
    
    # Пробуем разные кодировки
    encodings = [encoding, 'utf-8-sig', 'cp1251', 'latin-1']
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    raise UnicodeDecodeError(f"Не удалось прочитать файл {file_path} с кодировками: {', '.join(encodings)}")


def execute_sql_script(db_path: str, sql_file: str, stop_on_error: bool = True) -> Tuple[bool, Optional[str], int]:
    """
    Выполняет SQL-скрипт из файла
    
    Args:
        db_path: Путь к базе данных
        sql_file: Путь к SQL-файлу
        stop_on_error: Останавливать выполнение при ошибке
    
    Returns:
        (success, error_message, executed_count) - успех, сообщение об ошибке, количество выполненных запросов
    
    Example:
        >>> success, error, count = execute_sql_script('data/bots_data.db', 'migrations/001_init.sql')
        >>> if success:
        ...     print(f"Выполнено запросов: {count}")
        ... else:
        ...     print(f"Ошибка: {error}")
    """
    try:
        # Загружаем SQL-скрипт
        script_content = load_sql_file(sql_file)
        
        # Подключаемся к БД
        db = DatabaseConnection(db_path)
        db.connect()
        
        try:
            # Выполняем скрипт
            success, error, count = db.execute_script(script_content, stop_on_error=stop_on_error)
            return success, error, count
        finally:
            db.disconnect()
            
    except Exception as e:
        return False, str(e), 0


def execute_sql_string(db_path: str, sql_script: str, stop_on_error: bool = True) -> Tuple[bool, Optional[str], int]:
    """
    Выполняет SQL-скрипт из строки
    
    Args:
        db_path: Путь к базе данных
        sql_script: SQL-скрипт (строка)
        stop_on_error: Останавливать выполнение при ошибке
    
    Returns:
        (success, error_message, executed_count) - успех, сообщение об ошибке, количество выполненных запросов
    
    Example:
        >>> sql = "CREATE TABLE test (id INTEGER); INSERT INTO test VALUES (1);"
        >>> success, error, count = execute_sql_string('data/bots_data.db', sql)
    """
    try:
        db = DatabaseConnection(db_path)
        db.connect()
        
        try:
            success, error, count = db.execute_script(sql_script, stop_on_error=stop_on_error)
            return success, error, count
        finally:
            db.disconnect()
            
    except Exception as e:
        return False, str(e), 0


def execute_migrations(db_path: str, migrations_dir: str, verbose: bool = True) -> Dict[str, Tuple[bool, Optional[str], int]]:
    """
    Выполняет миграции из директории (файлы должны быть отсортированы по имени)
    
    Args:
        db_path: Путь к базе данных
        migrations_dir: Директория с файлами миграций (*.sql)
        verbose: Выводить информацию о выполнении миграций
    
    Returns:
        Словарь с результатами: {filename: (success, error, count), ...}
    
    Example:
        >>> results = execute_migrations('data/bots_data.db', 'migrations/')
        >>> for filename, (success, error, count) in results.items():
        ...     if success:
        ...         print(f"{filename}: выполнено {count} запросов")
        ...     else:
        ...         print(f"{filename}: ошибка - {error}")
    """
    migrations_path = Path(migrations_dir)
    
    if not migrations_path.exists():
        raise FileNotFoundError(f"Директория миграций не найдена: {migrations_dir}")
    
    # Получаем все SQL-файлы и сортируем по имени
    sql_files = sorted(migrations_path.glob("*.sql"))
    
    if verbose:
        print(f"Найдено миграций: {len(sql_files)}")
    
    results = {}
    
    for sql_file in sql_files:
        filename = sql_file.name
        
        if verbose:
            print(f"Выполнение миграции: {filename}...")
        
        success, error, count = execute_sql_script(db_path, str(sql_file), stop_on_error=True)
        
        results[filename] = (success, error, count)
        
        if verbose:
            if success:
                print(f"  ✓ {filename}: выполнено {count} запросов")
            else:
                print(f"  ✗ {filename}: ошибка - {error}")
                break  # Останавливаемся при ошибке
    
    return results


if __name__ == "__main__":
    # Пример использования
    import argparse
    
    parser = argparse.ArgumentParser(description='Утилита для выполнения SQL-скриптов')
    parser.add_argument('db_path', help='Путь к базе данных')
    parser.add_argument('sql_file', help='Путь к SQL-файлу')
    parser.add_argument('--continue-on-error', action='store_true', 
                       help='Продолжать выполнение при ошибке')
    
    args = parser.parse_args()
    
    success, error, count = execute_sql_script(
        args.db_path, 
        args.sql_file, 
        stop_on_error=not args.continue_on_error
    )
    
    if success:
        print(f"✓ Успешно выполнено запросов: {count}")
        sys.exit(0)
    else:
        print(f"✗ Ошибка: {error}")
        print(f"  Выполнено запросов до ошибки: {count}")
        sys.exit(1)

