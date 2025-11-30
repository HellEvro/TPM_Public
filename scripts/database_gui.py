#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI приложение для работы с базами данных InfoBot

Возможности:
- Автоматический поиск всех БД в проекте
- Открытие внешних БД
- SQL редактор для выполнения запросов
- GUI интерфейс для CRUD операций
- Изменяемые размеры окон
"""

import os
import sys
import sqlite3
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
except ImportError:
    print("Ошибка: требуется tkinter. Установите python3-tk или используйте Python с поддержкой tkinter.")
    sys.exit(1)

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Определяем корневую директорию проекта
ROOT = Path(__file__).parent.parent


class DraggablePanel(ttk.LabelFrame):
    """Панель с возможностью перетаскивания, прилипания и изменения размера"""
    
    SNAP_DISTANCE = 15  # Расстояние для прилипания в пикселях
    RESIZE_BORDER = 8  # Ширина области для изменения размера в пикселях
    MIN_WIDTH = 200  # Минимальная ширина панели
    MIN_HEIGHT = 150  # Минимальная высота панели
    
    def __init__(self, parent_canvas, canvas_id, text="", **kwargs):
        # Применяем стиль для панели
        kwargs.setdefault('style', 'TPanel.TLabelframe')
        super().__init__(parent_canvas, text=text, **kwargs)
        self.parent_canvas = parent_canvas
        self.canvas_id = canvas_id  # ID окна на Canvas
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_start_canvas_x = 0
        self.drag_start_canvas_y = 0
        self.drag_start_width = 0
        self.drag_start_height = 0
        self.is_dragging = False
        self.is_resizing = False
        self.resize_edge = None  # 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw'
        self.all_panels = []  # Список всех панелей (обновляется извне)
        
        # Делаем заголовок перетаскиваемым и панель изменяемой по размеру
        self._make_draggable_and_resizable()
    
    def set_panels_list(self, panels):
        """Устанавливает список всех панелей для snap"""
        self.all_panels = [p for p in panels if p != self]
    
    def _make_draggable_and_resizable(self):
        """Делает панель перетаскиваемой через заголовок и изменяемой по размеру через края"""
        # Привязываем события мыши
        self.bind("<Button-1>", self._on_drag_start)
        self.bind("<B1-Motion>", self._on_drag_motion)
        self.bind("<ButtonRelease-1>", self._on_drag_stop)
        
        # Изменяем курсор при наведении на края (для изменения размера) и заголовок
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)
    
    def _get_resize_edge(self, event):
        """Определяет край, рядом с которым находится курсор"""
        width = self.winfo_width()
        height = self.winfo_height()
        x = event.x
        y = event.y
        
        # Проверяем углы
        if x < self.RESIZE_BORDER and y < self.RESIZE_BORDER:
            return 'nw'
        elif x >= width - self.RESIZE_BORDER and y < self.RESIZE_BORDER:
            return 'ne'
        elif x < self.RESIZE_BORDER and y >= height - self.RESIZE_BORDER:
            return 'sw'
        elif x >= width - self.RESIZE_BORDER and y >= height - self.RESIZE_BORDER:
            return 'se'
        # Проверяем края
        elif x < self.RESIZE_BORDER:
            return 'w'
        elif x >= width - self.RESIZE_BORDER:
            return 'e'
        elif y < self.RESIZE_BORDER:
            return 'n'
        elif y >= height - self.RESIZE_BORDER:
            return 's'
        return None
    
    def _get_cursor_for_edge(self, edge):
        """Возвращает курсор для указанного края"""
        cursors = {
            'n': 'sb_v_double_arrow',  # Север
            's': 'sb_v_double_arrow',  # Юг
            'e': 'sb_h_double_arrow',  # Восток
            'w': 'sb_h_double_arrow',  # Запад
            'ne': 'top_right_corner',  # Северо-восток
            'nw': 'top_left_corner',   # Северо-запад
            'se': 'bottom_right_corner',  # Юго-восток
            'sw': 'bottom_left_corner',   # Юго-запад
        }
        return cursors.get(edge, '')
    
    def _is_header_area(self, event):
        """Проверяет, находится ли курсор в области заголовка"""
        # Заголовок занимает примерно верхние 30 пикселей
        return event.y < 30 and event.y > self.RESIZE_BORDER
    
    def _on_motion(self, event):
        """Обработчик движения мыши"""
        # Проверяем, не меняем ли мы размер
        if self.is_resizing:
            return
        
        # Проверяем край для изменения размера
        edge = self._get_resize_edge(event)
        if edge:
            cursor = self._get_cursor_for_edge(edge)
            if cursor:
                self.config(cursor=cursor)
        elif self._is_header_area(event):
            self.config(cursor="hand2")
        else:
            self.config(cursor="")
    
    def _on_leave(self, event):
        """Обработчик выхода курсора из панели"""
        if not self.is_dragging and not self.is_resizing:
            self.config(cursor="")
    
    def _on_drag_start(self, event):
        """Начало перетаскивания или изменения размера"""
        # Проверяем, не кликнули ли мы по краю для изменения размера
        edge = self._get_resize_edge(event)
        if edge:
            # Начинаем изменение размера
            self.is_resizing = True
            self.resize_edge = edge
            self.drag_start_x = event.x_root
            self.drag_start_y = event.y_root
            
            # Получаем текущие размеры и позицию
            coords = self.parent_canvas.coords(self.canvas_id)
            if coords:
                self.drag_start_canvas_x = coords[0]
                self.drag_start_canvas_y = coords[1]
            
            # Получаем текущие размеры из Canvas
            width = self.parent_canvas.itemcget(self.canvas_id, "width")
            height = self.parent_canvas.itemcget(self.canvas_id, "height")
            self.drag_start_width = int(width) if width else self.winfo_width()
            self.drag_start_height = int(height) if height else self.winfo_height()
            
            # Поднимаем панель наверх (z-order)
            self.parent_canvas.tag_raise(self.canvas_id)
            return
        
        # Если кликнули по заголовку - начинаем перетаскивание
        if not self._is_header_area(event):
            return
        
        self.is_dragging = True
        self.drag_start_x = event.x_root
        self.drag_start_y = event.y_root
        
        # Получаем текущую позицию на Canvas
        coords = self.parent_canvas.coords(self.canvas_id)
        if coords:
            self.drag_start_canvas_x = coords[0]
            self.drag_start_canvas_y = coords[1]
        
        # Поднимаем панель наверх (z-order)
        self.parent_canvas.tag_raise(self.canvas_id)
    
    def _on_drag_motion(self, event):
        """Перетаскивание панели или изменение размера"""
        if self.is_resizing:
            # Изменяем размер
            delta_x = event.x_root - self.drag_start_x
            delta_y = event.y_root - self.drag_start_y
            
            new_width = self.drag_start_width
            new_height = self.drag_start_height
            new_x = self.drag_start_canvas_x
            new_y = self.drag_start_canvas_y
            
            # Вычисляем новые размеры и позицию в зависимости от края
            edge = self.resize_edge
            
            # Запад (лево)
            if 'w' in edge:
                new_width = max(self.MIN_WIDTH, self.drag_start_width - delta_x)
                new_x = self.drag_start_canvas_x + (self.drag_start_width - new_width)
            
            # Восток (право)
            if 'e' in edge:
                new_width = max(self.MIN_WIDTH, self.drag_start_width + delta_x)
            
            # Север (верх)
            if 'n' in edge:
                new_height = max(self.MIN_HEIGHT, self.drag_start_height - delta_y)
                new_y = self.drag_start_canvas_y + (self.drag_start_height - new_height)
            
            # Юг (низ)
            if 's' in edge:
                new_height = max(self.MIN_HEIGHT, self.drag_start_height + delta_y)
            
            # Ограничиваем размерами Canvas
            canvas_width = self.parent_canvas.winfo_width()
            canvas_height = self.parent_canvas.winfo_height()
            
            if new_x + new_width > canvas_width:
                new_width = canvas_width - new_x
            if new_y + new_height > canvas_height:
                new_height = canvas_height - new_y
            if new_x < 0:
                new_width += new_x
                new_x = 0
            if new_y < 0:
                new_height += new_y
                new_y = 0
            
            # Обновляем размер и позицию на Canvas
            self.parent_canvas.coords(self.canvas_id, new_x, new_y)
            self.parent_canvas.itemconfig(self.canvas_id, width=int(new_width), height=int(new_height))
            self.parent_canvas.update()
            
        elif self.is_dragging:
            # Перетаскиваем панель
            delta_x = event.x_root - self.drag_start_x
            delta_y = event.y_root - self.drag_start_y
            
            new_x = self.drag_start_canvas_x + delta_x
            new_y = self.drag_start_canvas_y + delta_y
            
            # Проверяем прилипание к другим панелям
            snap_x, snap_y = self._check_snap(new_x, new_y)
            
            # Ограничиваем перемещение границами Canvas
            canvas_width = self.parent_canvas.winfo_width()
            canvas_height = self.parent_canvas.winfo_height()
            widget_width = self.winfo_width()
            widget_height = self.winfo_height()
            
            snap_x = max(0, min(snap_x, max(0, canvas_width - widget_width)))
            snap_y = max(0, min(snap_y, max(0, canvas_height - widget_height)))
            
            # Перемещаем панель на Canvas
            self.parent_canvas.coords(self.canvas_id, snap_x, snap_y)
            self.parent_canvas.update()
    
    def _on_drag_stop(self, event):
        """Окончание перетаскивания или изменения размера"""
        if self.is_resizing:
            self.is_resizing = False
            self.resize_edge = None
            self.config(cursor="")
            # Обновляем scrollregion
            self.parent_canvas.config(scrollregion=self.parent_canvas.bbox("all"))
        elif self.is_dragging:
            self.is_dragging = False
            self.config(cursor="")
            # Обновляем scrollregion
            self.parent_canvas.config(scrollregion=self.parent_canvas.bbox("all"))
    
    def _check_snap(self, x, y):
        """Проверяет прилипание к другим панелям"""
        snap_x, snap_y = x, y
        widget_width = self.winfo_width()
        widget_height = self.winfo_height()
        
        # Проверяем прилипание к каждой панели
        for panel in self.all_panels:
            # Получаем позицию панели на Canvas
            coords = self.parent_canvas.coords(panel.canvas_id)
            if not coords:
                continue
            
            px, py = coords
            pw = panel.winfo_width()
            ph = panel.winfo_height()
            
            # Проверяем прилипание слева (к правой стороне другой панели)
            if abs(x - (px + pw)) < self.SNAP_DISTANCE:
                snap_x = px + pw
            
            # Проверяем прилипание справа (к левой стороне другой панели)
            if abs((x + widget_width) - px) < self.SNAP_DISTANCE:
                snap_x = px - widget_width
            
            # Проверяем прилипание сверху (к нижней стороне другой панели)
            if abs(y - (py + ph)) < self.SNAP_DISTANCE:
                snap_y = py + ph
            
            # Проверяем прилипание снизу (к верхней стороне другой панели)
            if abs((y + widget_height) - py) < self.SNAP_DISTANCE:
                snap_y = py - widget_height
            
            # Проверяем прилипание по горизонтали (выравнивание левого края)
            if abs(x - px) < self.SNAP_DISTANCE:
                snap_x = px
            
            # Проверяем прилипание по вертикали (выравнивание верхнего края)
            if abs(y - py) < self.SNAP_DISTANCE:
                snap_y = py
        
        return snap_x, snap_y


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
            # Ошибка подключения будет обработана в вызывающем коде
            raise
    
    def disconnect(self):
        """Отключается от базы данных"""
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
            self.conn = None
    
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
        
        # Импортируем функцию разделения скрипта
        try:
            from scripts.database_utils import split_sql_script
        except ImportError:
            # Если модуль недоступен, используем простую реализацию
            def split_sql_script(script: str):
                # Простое разделение по ';' (не обрабатывает строки)
                return [q.strip() for q in script.split(';') if q.strip()]
        
        queries = split_sql_script(script)
        executed_count = 0
        
        try:
            cursor = self.conn.cursor()
            
            for query in queries:
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
    
    def get_tables(self) -> List[str]:
        """Получает список всех таблиц в БД"""
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            return [row[0] for row in cursor.fetchall()]
        except:
            return []
    
    def get_table_schema(self, table_name: str) -> List[Dict]:
        """Получает схему таблицы"""
        if not self.conn:
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            return [
                {
                    'cid': col[0],
                    'name': col[1],
                    'type': col[2],
                    'notnull': bool(col[3]),
                    'dflt_value': col[4],
                    'pk': bool(col[5])
                }
                for col in columns
            ]
        except:
            return []
    
    def get_table_data(self, table_name: str, limit: Optional[int] = 1000, offset: int = 0) -> Tuple[List[Dict], int]:
        """
        Получает данные из таблицы с пагинацией
        
        Args:
            table_name: Имя таблицы
            limit: Количество записей (None или 0 = все записи)
            offset: Смещение для пагинации
        
        Returns:
            (rows, total_count) - данные и общее количество записей
        """
        if not self.conn:
            return [], 0
        
        try:
            cursor = self.conn.cursor()
            
            # Получаем общее количество записей
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_count = cursor.fetchone()[0]
            
            # Получаем данные с лимитом или без него
            if limit is None or limit == 0:
                # Загружаем все записи
                cursor.execute(f"SELECT * FROM {table_name}")
            else:
                # Загружаем с пагинацией
                cursor.execute(f"SELECT * FROM {table_name} LIMIT ? OFFSET ?", (limit, offset))
            
            rows = cursor.fetchall()
            
            # Конвертируем в список словарей
            result = []
            for row in rows:
                result.append(dict(row))
            
            return result, total_count
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить данные:\n{e}")
            return [], 0
    
    def get_table_size(self, table_name: str) -> int:
        """
        Получает размер таблицы в байтах
        
        Args:
            table_name: Имя таблицы
            
        Returns:
            Размер таблицы в байтах (0 если ошибка)
        """
        if not self.conn:
            return 0
        
        try:
            cursor = self.conn.cursor()
            
            # Получаем количество записей в таблице
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            if row_count == 0:
                return 0
            
            # Для оптимизации используем меньшую выборку для больших таблиц
            sample_size = min(50, row_count)
            
            # Получаем размер одной записи (примерно) через выборку записей
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {sample_size}")
            sample_rows = cursor.fetchall()
            
            if sample_rows:
                # Вычисляем средний размер записи
                total_sample_size = sum(len(str(row)) for row in sample_rows)
                avg_row_size = total_sample_size // len(sample_rows) if sample_rows else 100
                
                # Общий размер таблицы (приблизительно)
                estimated_size = row_count * avg_row_size
                
                # Добавляем накладные расходы на индексы и структуру (примерно 20%)
                estimated_size = int(estimated_size * 1.2)
                
                return estimated_size
            
            return 0
        except Exception as e:
            # В случае ошибки возвращаем 0
            return 0


class DatabaseGUI(tk.Tk):
    """Главное окно GUI для работы с базами данных"""
    
    def __init__(self):
        super().__init__()
        
        self.title("InfoBot Database Manager")
        self.geometry("1400x900")
        self.minsize(1000, 700)
        
        # Настраиваем современную цветовую схему
        self._setup_styles()
        
        # Текущее подключение к БД
        self.db_conn: Optional[DatabaseConnection] = None
        self.current_table: Optional[str] = None
        
        # Переменные
        self.db_path_var = tk.StringVar()
        self.table_var = tk.StringVar()
        self.search_var = tk.StringVar()
        
        # Хранилище всех загруженных данных для фильтрации
        self.all_table_data: List[Dict] = []
        self.all_table_columns: List[str] = []
        
        # Хранилище для связи item_id -> db_path в дереве БД
        self.db_tree_items: Dict[str, Dict] = {}
        
        # Пагинация
        self.current_page = 1
        self.records_per_page = 100  # По умолчанию
        self.total_records = 0
        self.total_pages = 1
        
        # Настройки (будут загружены из файла)
        self.settings_file = ROOT / "data" / "database_gui_settings.json"
        self.settings = self._load_settings()
        
        # Применяем настройки
        if 'records_per_page' in self.settings:
            self.records_per_page = self.settings.get('records_per_page', 100)
        if 'geometry' in self.settings:
            self.geometry(self.settings.get('geometry', '1400x900'))
        
        # Создаем интерфейс
        self._build_ui()
        
        # Автоматически находим и загружаем БД из проекта
        self._auto_discover_databases()
    
    def _setup_styles(self):
        """Настраивает современные стили для интерфейса"""
        style = ttk.Style()
        
        # Пробуем использовать современную тему
        try:
            style.theme_use('clam')
        except:
            pass
        
        # Цветовая схема
        bg_color = '#f5f5f5'  # Светло-серый фон
        panel_bg = '#ffffff'  # Белый для панелей
        border_color = '#d0d0d0'  # Серая граница
        accent_color = '#0078d4'  # Синий акцент
        hover_color = '#e8f4f8'  # Светло-голубой при наведении
        text_color = '#1a1a1a'  # Темно-серый текст
        
        # Настраиваем стили для Frame
        style.configure('TPanel.TFrame', background=panel_bg, relief='flat', borderwidth=1)
        
        # Стили для LabelFrame
        style.configure('TPanel.TLabelframe', 
                       background=panel_bg, 
                       relief='flat', 
                       borderwidth=1,
                       bordercolor=border_color)
        style.configure('TPanel.TLabelframe.Label',
                       background=panel_bg,
                       foreground=text_color,
                       font=('Segoe UI', 10, 'bold'))
        
        # Стили для кнопок
        style.configure('TPrimary.TButton',
                       background=accent_color,
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(12, 6),
                       font=('Segoe UI', 9))
        style.map('TPrimary.TButton',
                 background=[('active', '#005a9e'), ('pressed', '#004578')])
        
        style.configure('TDefault.TButton',
                       background='#f0f0f0',
                       foreground=text_color,
                       borderwidth=1,
                       bordercolor=border_color,
                       focuscolor='none',
                       padding=(10, 5),
                       font=('Segoe UI', 9))
        style.map('TDefault.TButton',
                 background=[('active', hover_color)])
        
        # Стили для Entry
        style.configure('TSearch.TEntry',
                       fieldbackground='white',
                       foreground=text_color,
                       borderwidth=1,
                       bordercolor=border_color,
                       padding=(8, 6),
                       font=('Segoe UI', 9))
        style.map('TSearch.TEntry',
                 bordercolor=[('focus', accent_color)])
        
        # Стили для Treeview
        style.configure('TModern.Treeview',
                       background='white',
                       foreground=text_color,
                       fieldbackground='white',
                       borderwidth=0,
                       font=('Segoe UI', 9))
        style.configure('TModern.Treeview.Heading',
                       background='#f8f8f8',
                       foreground=text_color,
                       relief='flat',
                       borderwidth=1,
                       bordercolor=border_color,
                       font=('Segoe UI', 9, 'bold'))
        style.map('TModern.Treeview',
                 background=[('selected', accent_color)],
                 foreground=[('selected', 'white')])
        
        # Настраиваем фон главного окна
        self.configure(bg=bg_color)
    
    def _build_ui(self):
        """Создает интерфейс приложения"""
        # Главный контейнер - Canvas для перетаскиваемых панелей
        main_container = tk.Frame(self, bg='#f5f5f5')
        main_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Создаем Canvas для размещения панелей
        self.main_canvas = tk.Canvas(main_container, bg='#f5f5f5', highlightthickness=0)
        self.main_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Хранилище панелей
        self.panels = {}
        
        # === ЛЕВАЯ ПАНЕЛЬ: Управление БД ===
        left_frame_container = ttk.Frame(self.main_canvas)
        left_frame = DraggablePanel(left_frame_container, None, text="Базы данных")
        left_frame.pack(fill=tk.BOTH, expand=True)
        
        # Размещаем панель на Canvas - левая панель слева, занимает всю высоту
        left_id = self.main_canvas.create_window(10, 10, window=left_frame_container, anchor="nw", width=300, height=840)
        left_frame.canvas_id = left_id
        left_frame.parent_canvas = self.main_canvas
        
        # Убираем заголовок - он уже есть в DraggablePanel
        
        # Кнопки управления БД
        db_buttons_frame = ttk.Frame(left_frame, style='TPanel.TFrame')
        db_buttons_frame.pack(fill=tk.X, padx=8, pady=8)
        
        ttk.Button(
            db_buttons_frame,
            text="Найти БД в проекте",
            command=self._auto_discover_databases,
            style='TDefault.TButton'
        ).pack(fill=tk.X, pady=3)
        
        ttk.Button(
            db_buttons_frame,
            text="Открыть внешнюю БД",
            command=self._open_external_database,
            style='TDefault.TButton'
        ).pack(fill=tk.X, pady=3)
        
        ttk.Button(
            db_buttons_frame,
            text="Обновить список",
            command=self._refresh_databases,
            style='TDefault.TButton'
        ).pack(fill=tk.X, pady=3)
        
        # Список найденных БД
        db_list_frame = ttk.LabelFrame(left_frame, text="Найденные БД", padding=8, style='TPanel.TLabelframe')
        db_list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Контейнер для Treeview и Scrollbar с правильным layout
        db_tree_container = ttk.Frame(db_list_frame)
        db_tree_container.pack(fill=tk.BOTH, expand=True)
        
        # Treeview для списка БД
        db_tree = ttk.Treeview(db_tree_container, show="tree", height=15, style='TModern.Treeview')
        db_tree.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbar для списка
        db_scroll = ttk.Scrollbar(db_tree_container, orient=tk.VERTICAL, command=db_tree.yview)
        db_scroll.grid(row=0, column=1, sticky="ns")
        db_tree.configure(yscrollcommand=db_scroll.set)
        
        # Настраиваем grid weights
        db_tree_container.grid_rowconfigure(0, weight=1)
        db_tree_container.grid_columnconfigure(0, weight=1)
        
        self.db_tree = db_tree
        
        # Привязываем клики
        db_tree.bind("<Double-1>", lambda e: self._on_tree_item_double_click())
        db_tree.bind("<Button-1>", lambda e: self._on_tree_item_click())
        db_tree.bind("<Button-3>", lambda e: self._on_tree_item_right_click(e))  # Правый клик для контекстного меню
        db_tree.bind("<<TreeviewOpen>>", lambda e: self._on_tree_item_expand())  # Раскрытие узла
        
        # Информация о текущей БД
        info_frame = ttk.LabelFrame(left_frame, text="Информация о БД", padding=8, style='TPanel.TLabelframe')
        info_frame.pack(fill=tk.X, padx=8, pady=8)
        
        self.db_info_text = tk.Text(info_frame, height=5, wrap=tk.WORD, state=tk.DISABLED,
                                    bg='white', fg='#1a1a1a', font=('Segoe UI', 9),
                                    relief='flat', borderwidth=1, highlightthickness=1,
                                    highlightbackground='#d0d0d0', highlightcolor='#0078d4')
        self.db_info_text.pack(fill=tk.X)
        
        self.panels['left'] = left_frame
        
        # === ПАНЕЛЬ: SQL редактор ===
        sql_frame_container = ttk.Frame(self.main_canvas)
        sql_frame = DraggablePanel(sql_frame_container, None, text="SQL Редактор")
        sql_frame.pack(fill=tk.BOTH, expand=True)
        
        # SQL редактор справа вверху
        sql_id = self.main_canvas.create_window(320, 10, window=sql_frame_container, anchor="nw", width=580, height=280)
        sql_frame.canvas_id = sql_id
        sql_frame.parent_canvas = self.main_canvas
        
        # SQL редактор
        sql_text = scrolledtext.ScrolledText(
            sql_frame,
            wrap=tk.NONE,
            font=("Consolas", 10),
            height=10,
            bg='white',
            fg='#1a1a1a',
            relief='flat',
            borderwidth=1,
            highlightthickness=1,
            highlightbackground='#d0d0d0',
            highlightcolor='#0078d4',
            padx=8,
            pady=8
        )
        sql_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.sql_text = sql_text
        
        # Кнопки SQL
        sql_buttons_frame = ttk.Frame(sql_frame, style='TPanel.TFrame')
        sql_buttons_frame.pack(fill=tk.X, padx=8, pady=8)
        
        ttk.Button(
            sql_buttons_frame,
            text="Выполнить запрос (F5)",
            command=self._execute_sql,
            style='TPrimary.TButton'
        ).pack(side=tk.LEFT, padx=4)
        
        ttk.Button(
            sql_buttons_frame,
            text="Очистить",
            command=lambda: self.sql_text.delete(1.0, tk.END),
            style='TDefault.TButton'
        ).pack(side=tk.LEFT, padx=4)
        
        ttk.Button(
            sql_buttons_frame,
            text="Загрузить SQL файл",
            command=self._load_sql_file,
            style='TDefault.TButton'
        ).pack(side=tk.LEFT, padx=4)
        
        ttk.Button(
            sql_buttons_frame,
            text="Сохранить SQL запрос",
            command=self._save_sql_query,
            style='TDefault.TButton'
        ).pack(side=tk.LEFT, padx=4)
        
        # Привязываем F5 к выполнению запроса
        self.bind('<F5>', lambda e: self._execute_sql())
        
        self.panels['sql'] = sql_frame
        
        # === ПАНЕЛЬ: Таблицы и данные ===
        tables_frame_container = ttk.Frame(self.main_canvas)
        tables_frame = DraggablePanel(tables_frame_container, None, text="Таблицы и данные")
        tables_frame.pack(fill=tk.BOTH, expand=True)
        
        # Таблицы и данные справа внизу, занимают всю ширину справа
        tables_id = self.main_canvas.create_window(320, 300, window=tables_frame_container, anchor="nw", width=1020, height=550)
        tables_frame.canvas_id = tables_id
        tables_frame.parent_canvas = self.main_canvas
        
        # Список таблиц
        tables_list_frame = ttk.Frame(tables_frame, style='TPanel.TFrame')
        tables_list_frame.pack(fill=tk.X, padx=8, pady=8)
        
        ttk.Label(tables_list_frame, text="Таблицы:", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=4)
        tables_combo = ttk.Combobox(
            tables_list_frame,
            textvariable=self.table_var,
            state="readonly",
            width=30,
            font=('Segoe UI', 9)
        )
        tables_combo.pack(side=tk.LEFT, padx=4)
        tables_combo.bind("<<ComboboxSelected>>", lambda e: self._load_table_data())
        self.tables_combo = tables_combo
        
        # Поиск/фильтр
        search_frame = ttk.Frame(tables_frame, style='TPanel.TFrame')
        search_frame.pack(fill=tk.X, padx=8, pady=8)
        
        ttk.Label(search_frame, text="Поиск:", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=4)
        search_entry = ttk.Entry(
            search_frame,
            textvariable=self.search_var,
            width=30,
            style='TSearch.TEntry'
        )
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        
        # Привязываем поиск при вводе
        self.search_var.trace_add('write', lambda *args: self._filter_table_data())
        
        # Фильтр количества записей
        records_filter_frame = ttk.Frame(search_frame)
        records_filter_frame.pack(side=tk.LEFT, padx=8)
        
        ttk.Label(records_filter_frame, text="Записей:", font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=2)
        
        # Выпадающий список с предустановленными значениями
        records_per_page_str = 'Все' if self.records_per_page == 0 else str(self.records_per_page)
        if records_per_page_str not in ['10', '50', '100', '500', '1000', 'Все']:
            records_per_page_str = str(self.records_per_page)
        self.records_per_page_var = tk.StringVar(value=records_per_page_str)
        records_combo = ttk.Combobox(
            records_filter_frame,
            textvariable=self.records_per_page_var,
            values=['10', '50', '100', '500', '1000', 'Все'],
            width=8,
            state='readonly'
        )
        records_combo.pack(side=tk.LEFT, padx=2)
        records_combo.bind('<<ComboboxSelected>>', self._on_records_per_page_changed)
        
        # Поле для ввода произвольного количества
        self.custom_records_var = tk.StringVar()
        custom_records_entry = ttk.Entry(
            records_filter_frame,
            textvariable=self.custom_records_var,
            width=8
        )
        custom_records_entry.pack(side=tk.LEFT, padx=2)
        custom_records_entry.bind('<Return>', self._on_custom_records_entered)
        custom_records_entry.bind('<FocusOut>', self._on_custom_records_entered)
        
        ttk.Label(records_filter_frame, text="(вручную)", font=('Segoe UI', 8), foreground='gray').pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            search_frame,
            text="Очистить",
            command=self._clear_search,
            style='TDefault.TButton'
        ).pack(side=tk.LEFT, padx=4)
        
        # Кнопки управления таблицей
        table_buttons_frame = ttk.Frame(tables_frame, style='TPanel.TFrame')
        table_buttons_frame.pack(fill=tk.X, padx=8, pady=8)
        
        ttk.Button(
            table_buttons_frame,
            text="Обновить",
            command=self._load_table_data,
            style='TDefault.TButton'
        ).pack(side=tk.LEFT, padx=3)
        
        ttk.Button(
            table_buttons_frame,
            text="Добавить запись",
            command=self._add_record,
            style='TPrimary.TButton'
        ).pack(side=tk.LEFT, padx=3)
        
        ttk.Button(
            table_buttons_frame,
            text="Редактировать",
            command=self._edit_record,
            style='TDefault.TButton'
        ).pack(side=tk.LEFT, padx=3)
        
        ttk.Button(
            table_buttons_frame,
            text="Удалить",
            command=self._delete_record,
            style='TDefault.TButton'
        ).pack(side=tk.LEFT, padx=3)
        
        # Данные таблицы
        data_frame = ttk.Frame(tables_frame, style='TPanel.TFrame')
        data_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Treeview для отображения данных
        data_tree = ttk.Treeview(data_frame, style='TModern.Treeview')
        data_tree.grid(row=0, column=0, sticky="nsew")
        
        # Вертикальный Scrollbar
        v_scroll = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=data_tree.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        
        # Горизонтальный Scrollbar
        h_scroll = ttk.Scrollbar(data_frame, orient=tk.HORIZONTAL, command=data_tree.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        
        data_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # Настраиваем grid weights
        data_frame.grid_rowconfigure(0, weight=1)
        data_frame.grid_columnconfigure(0, weight=1)
        
        self.data_tree = data_tree
        
        # Информация о записях и пагинация
        pagination_frame = ttk.Frame(tables_frame)
        pagination_frame.pack(fill=tk.X, padx=8, pady=4)
        
        records_info = ttk.Label(pagination_frame, text="Записей: 0", font=('Segoe UI', 9))
        records_info.pack(side=tk.LEFT, padx=4)
        self.records_info = records_info
        
        # Пагинация
        pagination_controls = ttk.Frame(pagination_frame)
        pagination_controls.pack(side=tk.RIGHT, padx=4)
        
        ttk.Button(
            pagination_controls,
            text="◀◀",
            command=lambda: self._go_to_page(1),
            width=3
        ).pack(side=tk.LEFT, padx=1)
        
        ttk.Button(
            pagination_controls,
            text="◀",
            command=self._prev_page,
            width=3
        ).pack(side=tk.LEFT, padx=1)
        
        self.page_info_var = tk.StringVar(value="Стр. 1/1")
        page_info_label = ttk.Label(pagination_controls, textvariable=self.page_info_var, font=('Segoe UI', 9))
        page_info_label.pack(side=tk.LEFT, padx=4)
        
        # Поле для ввода номера страницы
        self.page_entry_var = tk.StringVar()
        page_entry = ttk.Entry(pagination_controls, textvariable=self.page_entry_var, width=5)
        page_entry.pack(side=tk.LEFT, padx=2)
        page_entry.bind('<Return>', self._on_page_entered)
        
        ttk.Button(
            pagination_controls,
            text="▶",
            command=self._next_page,
            width=3
        ).pack(side=tk.LEFT, padx=1)
        
        ttk.Button(
            pagination_controls,
            text="▶▶",
            command=lambda: self._go_to_page(self.total_pages),
            width=3
        ).pack(side=tk.LEFT, padx=1)
        
        self.panels['tables'] = tables_frame
        
        # === ПАНЕЛЬ: Результаты SQL ===
        results_frame_container = ttk.Frame(self.main_canvas)
        results_frame = DraggablePanel(results_frame_container, None, text="Результаты SQL")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Результаты SQL справа вверху, рядом с SQL редактором
        results_id = self.main_canvas.create_window(910, 10, window=results_frame_container, anchor="nw", width=430, height=280)
        results_frame.canvas_id = results_id
        results_frame.parent_canvas = self.main_canvas
        
        # Treeview для результатов
        results_tree = ttk.Treeview(results_frame, style='TModern.Treeview')
        results_tree.grid(row=0, column=0, sticky="nsew")
        
        # Вертикальный Scrollbar для результатов
        res_v_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=results_tree.yview)
        res_v_scroll.grid(row=0, column=1, sticky="ns")
        
        # Горизонтальный Scrollbar для результатов
        res_h_scroll = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=results_tree.xview)
        res_h_scroll.grid(row=1, column=0, sticky="ew")
        
        results_tree.configure(yscrollcommand=res_v_scroll.set, xscrollcommand=res_h_scroll.set)
        
        # Настраиваем grid weights
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        self.results_tree = results_tree
        
        self.panels['results'] = results_frame
        
        # Обновляем размер Canvas при изменении окна
        def update_canvas_size(event=None):
            self.main_canvas.config(scrollregion=self.main_canvas.bbox("all"))
        
        self.main_canvas.bind('<Configure>', update_canvas_size)
        self.bind('<Configure>', update_canvas_size)
        
        # Обновляем список панелей для snap
        panels_list = list(self.panels.values())
        for panel in panels_list:
            panel.set_panels_list(panels_list)
        
        # Обновляем scrollregion при изменении размера
        def update_scrollregion(event=None):
            self.main_canvas.config(scrollregion=self.main_canvas.bbox("all"))
        
        self.main_canvas.bind('<Configure>', update_scrollregion)
        self.bind('<Configure>', update_scrollregion)
        
        # === СТРОКА СТАТУСА ВНИЗУ ===
        status_frame = tk.Frame(self, bg='#0078d4', height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        status_label = tk.Label(
            status_frame,
            text="Готов к работе",
            anchor=tk.W,
            bg='#0078d4',
            fg='white',
            font=('Segoe UI', 9),
            padx=10,
            pady=6
        )
        status_label.pack(fill=tk.BOTH, expand=True)
        self.status_label = status_label
    
    def _update_status(self, message: str, status_type: str = "info"):
        """
        Обновляет строку статуса
        
        Args:
            message: Текст сообщения
            status_type: Тип статуса - "info", "success", "warning", "error"
        """
        self.status_label.config(text=message)
        
        # Цвета для разных типов статуса
        colors = {
            "info": "#0078d4",      # Синий
            "success": "#107c10",   # Зеленый
            "warning": "#ffaa44",   # Оранжевый
            "error": "#d13438"      # Красный
        }
        
        bg_color = colors.get(status_type, "#0078d4")
        self.status_label.config(background=bg_color)
        
        # Автоматически очищаем статус через 5 секунд (кроме ошибок)
        if status_type != "error":
            self.after(5000, lambda: self.status_label.config(text="Готов к работе", background="#0078d4", fg='white'))
    
    def _auto_discover_databases(self):
        """Автоматически находит все БД в проекте"""
        databases = []
        found_paths = set()  # Для отслеживания уже добавленных путей
        
        # Список известных путей к БД (показываем даже если файлы еще не созданы)
        known_paths = [
            ROOT / "data" / "bots_data.db",
            ROOT / "data" / "app_data.db",
            ROOT / "data" / "ai_data.db",
            ROOT / "license_generator" / "licenses.db",
        ]
        
        # Добавляем известные пути
        for db_path in known_paths:
            if db_path.exists():
                found_paths.add(str(db_path))
                databases.append({
                    'name': db_path.name,
                    'path': str(db_path),
                    'relative_path': str(db_path.relative_to(ROOT)),
                    'size': db_path.stat().st_size,
                    'exists': True
                })
            else:
                # Добавляем даже если файл не существует (с пометкой)
                databases.append({
                    'name': db_path.name,
                    'path': str(db_path),
                    'relative_path': str(db_path.relative_to(ROOT)),
                    'size': 0,
                    'exists': False
                })
        
        # Добавляем все .db файлы из проекта (кроме уже добавленных)
        # Исключаем служебные папки
        excluded_dirs = {'.git', '.venv', '__pycache__', 'node_modules', '.idea', '.vscode'}
        
        for db_file in ROOT.rglob("*.db"):
            # Пропускаем файлы .db-wal и .db-shm
            if db_file.name.endswith(('-wal', '-shm')):
                continue
            
            # Пропускаем если путь уже добавлен
            db_path_str = str(db_file)
            if db_path_str in found_paths:
                continue
            
            # Пропускаем если в пути есть исключенные директории
            if any(excluded in db_path_str for excluded in excluded_dirs):
                continue
            
            found_paths.add(db_path_str)
            databases.append({
                'name': db_file.name,
                'path': db_path_str,
                'relative_path': str(db_file.relative_to(ROOT)),
                'size': db_file.stat().st_size if db_file.exists() else 0,
                'exists': db_file.exists()
            })
        
        # Сортируем: сначала существующие, затем по имени
        databases.sort(key=lambda x: (not x.get('exists', True), x['name']))
        
        # Обновляем дерево БД
        self._update_database_tree(databases)
        
        # Обновляем статус (если метод вызван не из __init__)
        if hasattr(self, 'status_label'):
            count = len([db for db in databases if db.get('exists', True)])
            total = len(databases)
            self._update_status(f"Найдено баз данных: {count} существующих из {total} известных", "info")
    
    def _update_database_tree(self, databases: List[Dict]):
        """Обновляет дерево со списком БД"""
        # Очищаем дерево
        for item in self.db_tree.get_children():
            self.db_tree.delete(item)
        
        # Добавляем БД в дерево
        root_id = self.db_tree.insert("", tk.END, text="Проект", open=True)
        
        # Хранилище для связи item_id -> db_path
        self.db_tree_items = {}  # item_id -> {'type': 'db'|'table', 'db_path': str, 'table_name': str|None}
        
        for db in databases:
            exists = db.get('exists', True)
            if exists:
                size_mb = db['size'] / 1024 / 1024
                display_text = f"{db['name']} ({size_mb:.2f} MB)"
            else:
                display_text = f"{db['name']} (не создана)"
            
            item_id = self.db_tree.insert(
                root_id,
                tk.END,
                text=display_text,
                values=(db['path'], db['relative_path'], '1' if exists else '0', 'db')
            )
            
            # Сохраняем информацию о БД
            self.db_tree_items[item_id] = {
                'type': 'db',
                'db_path': db['path'],
                'table_name': None,
                'exists': exists
            }
            
            # Добавляем placeholder для таблиц (чтобы можно было раскрыть)
            if exists:
                placeholder_id = self.db_tree.insert(
                    item_id,
                    tk.END,
                    text="Загрузка таблиц...",
                    values=('', '', '', 'placeholder')
                )
                self.db_tree_items[placeholder_id] = {
                    'type': 'placeholder',
                    'db_path': db['path'],
                    'table_name': None
                }
    
    def _on_tree_item_click(self):
        """Обработчик клика на элемент дерева"""
        selection = self.db_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        item_info = self.db_tree_items.get(item)
        
        if not item_info:
            return
        
        # Если кликнули на таблицу - открываем её
        if item_info['type'] == 'table':
            db_path = item_info['db_path']
            table_name = item_info['table_name']
            
            # Открываем БД если она не открыта
            if not self.db_conn or self.db_conn.db_path != db_path:
                self._open_database(db_path)
            
            # Выбираем таблицу (не устанавливаем current_table здесь, чтобы _load_table_data мог определить изменение)
            if self.db_conn:
                self.table_var.set(table_name)
                self._load_table_data()
    
    def _on_tree_item_double_click(self):
        """Обработчик двойного клика на элемент дерева"""
        selection = self.db_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        item_info = self.db_tree_items.get(item)
        
        if not item_info:
            return
        
        # Если двойной клик на БД - открываем её
        if item_info['type'] == 'db':
            db_path = item_info['db_path']
            self._open_database(db_path)
        # Если двойной клик на таблицу - открываем её данные
        elif item_info['type'] == 'table':
            self._on_tree_item_click()
    
    def _on_tree_item_expand(self, event=None):
        """Обработчик раскрытия узла БД - загружает таблицы"""
        # Получаем раскрытый элемент из события
        item = self.db_tree.focus()
        if not item:
            # Пробуем найти раскрытый элемент другим способом
            for item_id in self.db_tree_items.keys():
                try:
                    if self.db_tree.item(item_id, 'open'):
                        item = item_id
                        break
                except:
                    continue
        
        if not item:
            return
        
        item_info = self.db_tree_items.get(item)
        
        # Если это БД - загружаем таблицы
        if item_info and item_info['type'] == 'db':
            db_path = item_info['db_path']
            # Проверяем, есть ли уже загруженные таблицы
            has_tables = False
            for child in self.db_tree.get_children(item):
                child_info = self.db_tree_items.get(child)
                if child_info and child_info['type'] == 'table':
                    has_tables = True
                    break
                elif child_info and child_info['type'] == 'placeholder':
                    # Удаляем placeholder
                    self.db_tree.delete(child)
                    if child in self.db_tree_items:
                        del self.db_tree_items[child]
            
            # Загружаем таблицы если их еще нет
            if not has_tables:
                self._load_tables_into_tree(item, db_path)
    
    def _load_tables_into_tree(self, db_item_id, db_path: str):
        """Загружает таблицы в дерево для указанной БД"""
        try:
            # Подключаемся к БД
            temp_conn = DatabaseConnection(db_path)
            temp_conn.connect()
            
            # Получаем список таблиц
            tables = temp_conn.get_tables()
            
            # Удаляем старые дочерние элементы (если есть)
            for child in self.db_tree.get_children(db_item_id):
                child_info = self.db_tree_items.get(child)
                if child_info and child_info['type'] == 'table':
                    self.db_tree.delete(child)
                    if child in self.db_tree_items:
                        del self.db_tree_items[child]
            
            # Добавляем таблицы с размерами
            for table_name in tables:
                # Получаем размер таблицы
                table_size = temp_conn.get_table_size(table_name)
                
                # Форматируем размер
                if table_size >= 1024 * 1024:  # MB
                    size_str = f"{table_size / (1024 * 1024):.2f} MB"
                elif table_size >= 1024:  # KB
                    size_str = f"{table_size / 1024:.2f} KB"
                else:
                    size_str = f"{table_size} B"
                
                # Формируем текст для отображения
                display_text = f"📋 {table_name} ({size_str})"
                
                table_item_id = self.db_tree.insert(
                    db_item_id,
                    tk.END,
                    text=display_text,
                    values=(db_path, table_name, '', 'table')
                )
                
                self.db_tree_items[table_item_id] = {
                    'type': 'table',
                    'db_path': db_path,
                    'table_name': table_name
                }
            
            temp_conn.disconnect()
            
        except Exception as e:
            self._update_status(f"Ошибка загрузки таблиц: {e}", "error")
    
    def _on_tree_item_right_click(self, event):
        """Обработчик правого клика - показывает контекстное меню"""
        # Определяем элемент под курсором
        item = self.db_tree.identify_row(event.y)
        if not item:
            return
        
        item_info = self.db_tree_items.get(item)
        if not item_info:
            return
        
        # Создаем контекстное меню
        menu = tk.Menu(self, tearoff=0)
        
        if item_info['type'] == 'table':
            # Меню для таблицы
            menu.add_command(label="Открыть таблицу", command=lambda: self._on_tree_item_click())
            menu.add_separator()
            menu.add_command(label="Редактировать таблицу", command=lambda: self._edit_table(item_info))
            menu.add_command(label="Обнулить таблицу", command=lambda: self._truncate_table(item_info))
            menu.add_command(label="Удалить таблицу", command=lambda: self._delete_table(item_info))
        elif item_info['type'] == 'db':
            # Меню для БД
            menu.add_command(label="Открыть БД", command=lambda: self._open_database(item_info['db_path']))
            menu.add_separator()
            menu.add_command(label="Добавить таблицу", command=lambda: self._create_table(item_info))
            menu.add_command(label="Обновить список таблиц", command=lambda: self._refresh_tables_in_tree(item))
        
        # Показываем меню
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
    
    def _open_external_database(self):
        """Открывает внешнюю БД через диалог выбора файла"""
        self._update_status("Выбор базы данных...", "info")
        db_path = filedialog.askopenfilename(
            title="Выберите базу данных",
            filetypes=[("SQLite databases", "*.db"), ("All files", "*.*")]
        )
        
        if db_path:
            self._open_database(db_path)
            # Добавляем в дерево
            self._refresh_databases()
        else:
            self._update_status("Выбор базы данных отменен", "info")
    
    def _open_database(self, db_path: str):
        """Открывает базу данных"""
        # Проверяем существование файла
        if not os.path.exists(db_path):
            if messagebox.askyesno(
                "База данных не найдена",
                f"Файл базы данных не существует:\n{db_path}\n\nСоздать новую базу данных?"
            ):
                # Создаем директорию если её нет
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                # Создаем пустую БД
                try:
                    conn = sqlite3.connect(db_path)
                    conn.close()
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось создать базу данных:\n{e}")
                    return
            else:
                return
        
        # Закрываем текущее подключение
        if self.db_conn:
            self.db_conn.disconnect()
        
        # Создаем новое подключение
        try:
            self.db_conn = DatabaseConnection(db_path)
            self.db_conn.connect()
        except Exception as e:
            self._update_status(f"Ошибка подключения: {e}", "error")
            self.db_conn = None
            return
        
        self.db_path_var.set(db_path)
        
        # Обновляем информацию о БД
        self._update_database_info()
        
        # Загружаем список таблиц
        self._load_tables_list()
        
        # Обновляем статус
        db_name = os.path.basename(db_path)
        self._update_status(f"База данных открыта: {db_name}", "success")
        
        # Обновляем список БД (чтобы обновился статус "не создана" -> "существует")
        self._auto_discover_databases()
        
        # Обновляем таблицы в дереве для открытой БД
        self._refresh_tables_for_opened_db(db_path)
    
    def _update_database_info(self):
        """Обновляет информацию о текущей БД"""
        if not self.db_conn or not self.db_conn.conn:
            self.db_info_text.config(state=tk.NORMAL)
            self.db_info_text.delete(1.0, tk.END)
            self.db_info_text.config(state=tk.DISABLED)
            return
        
        try:
            # Получаем информацию о БД
            db_path = self.db_conn.db_path
            file_size = os.path.getsize(db_path) / 1024 / 1024
            tables = self.db_conn.get_tables()
            
            info_text = f"Путь: {db_path}\n"
            info_text += f"Размер: {file_size:.2f} MB\n"
            info_text += f"Таблиц: {len(tables)}\n"
            
            # Получаем статистику
            if tables:
                cursor = self.db_conn.conn.cursor()
                stats = []
                for table in tables[:5]:  # Показываем только первые 5 таблиц
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        stats.append(f"{table}: {count} записей")
                    except:
                        pass
                if stats:
                    info_text += "\n".join(stats)
            
            self.db_info_text.config(state=tk.NORMAL)
            self.db_info_text.delete(1.0, tk.END)
            self.db_info_text.insert(1.0, info_text)
            self.db_info_text.config(state=tk.DISABLED)
        except Exception as e:
            self.db_info_text.config(state=tk.NORMAL)
            self.db_info_text.delete(1.0, tk.END)
            self.db_info_text.insert(1.0, f"Ошибка: {e}")
            self.db_info_text.config(state=tk.DISABLED)
    
    def _load_tables_list(self):
        """Загружает список таблиц в комбобокс"""
        if not self.db_conn:
            return
        
        self._update_status("Загрузка списка таблиц...", "info")
        tables = self.db_conn.get_tables()
        self.tables_combo['values'] = tables
        
        if tables:
            self.tables_combo.current(0)
            # Не устанавливаем current_table здесь, чтобы _load_table_data мог определить изменение
            self._update_status(f"Найдено таблиц: {len(tables)}", "success")
            self._load_table_data()
        else:
            self._update_status("База данных не содержит таблиц", "warning")
    
    def _load_table_data(self):
        """Загружает данные из выбранной таблицы"""
        if not self.db_conn:
            return
        
        table_name = self.table_var.get()
        if not table_name:
            return
        
        # Проверяем, изменилась ли таблица
        table_changed = (self.current_table != table_name)
        self.current_table = table_name
        
        # Обновляем статус
        self._update_status(f"Загрузка данных из таблицы '{table_name}'...", "info")
        
        # Очищаем treeview
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Получаем схему таблицы
        schema = self.db_conn.get_table_schema(table_name)
        if not schema:
            self._update_status(f"Ошибка: Не удалось получить схему таблицы '{table_name}'", "error")
            return
        
        # Настраиваем колонки
        columns = [col['name'] for col in schema]
        self.data_tree['columns'] = columns
        self.data_tree['show'] = 'headings'
        
        # Настраиваем заголовки
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=150, minwidth=100)
        
        # Сбрасываем пагинацию при загрузке новой таблицы
        if table_changed:
            self.current_page = 1
        
        # Загружаем данные с пагинацией
        limit = self.records_per_page if self.records_per_page > 0 else None
        offset = (self.current_page - 1) * self.records_per_page if self.records_per_page > 0 else 0
        rows, total_count = self.db_conn.get_table_data(table_name, limit=limit, offset=offset)
        
        # Сохраняем общее количество записей
        self.total_records = total_count
        self.total_pages = max(1, (total_count + self.records_per_page - 1) // self.records_per_page) if self.records_per_page > 0 else 1
        
        # Если текущая страница больше общего количества страниц (может быть при смене таблицы), сбрасываем на 1
        if self.current_page > self.total_pages:
            self.current_page = 1
            # Перезагружаем данные с первой страницы
            offset = 0
            rows, _ = self.db_conn.get_table_data(table_name, limit=limit, offset=offset)
        
        # Сохраняем загруженные данные
        self.all_table_data = rows
        self.all_table_columns = columns
        
        # Очищаем фильтр только при загрузке новой таблицы
        if table_changed:
            self.search_var.set("")
        
        # Отображаем данные (с учетом текущего фильтра)
        self._display_filtered_data()
        
        # Обновляем пагинацию
        self._update_pagination_info()
        
        # Обновляем статус
        self._update_status(f"Загружено {len(rows)} записей из таблицы '{table_name}' (всего: {total_count})", "success")
    
    def _filter_table_data(self):
        """Фильтрует данные таблицы по поисковому запросу"""
        if not self.all_table_data or not self.all_table_columns:
            return
        
        self._display_filtered_data()
    
    def _display_filtered_data(self):
        """Отображает отфильтрованные данные в таблице"""
        # Очищаем treeview
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        if not self.all_table_data or not self.all_table_columns:
            self.records_info.config(text="Записей: 0")
            return
        
        # Получаем поисковый запрос
        search_text = self.search_var.get().strip().lower()
        
        # Фильтруем данные
        if search_text:
            filtered_rows = []
            for row in self.all_table_data:
                # Проверяем, содержится ли поисковый запрос в любой из колонок
                match = False
                for col in self.all_table_columns:
                    value = str(row.get(col, '')).lower()
                    if search_text in value:
                        match = True
                        break
                
                if match:
                    filtered_rows.append(row)
        else:
            # Если поисковый запрос пуст, показываем все данные
            filtered_rows = self.all_table_data
        
        # Добавляем отфильтрованные данные
        for row in filtered_rows:
            values = [str(row.get(col, '')) for col in self.all_table_columns]
            self.data_tree.insert("", tk.END, values=values)
        
        # Обновляем информацию о записях
        total_count = self.total_records
        shown_count = len(filtered_rows)
        if search_text:
            self.records_info.config(text=f"Записей: {total_count} (найдено: {shown_count})")
        else:
            start_record = (self.current_page - 1) * self.records_per_page + 1 if self.records_per_page > 0 else 1
            end_record = min(start_record + len(filtered_rows) - 1, total_count) if self.records_per_page > 0 else total_count
            self.records_info.config(text=f"Записей: {total_count} (показано: {start_record}-{end_record})")
    
    def _clear_search(self):
        """Очищает поле поиска"""
        self.search_var.set("")
        self._display_filtered_data()
    
    def _execute_sql(self):
        """Выполняет SQL запрос"""
        if not self.db_conn:
            self._update_status("Ошибка: База данных не открыта", "error")
            return
        
        query = self.sql_text.get(1.0, tk.END).strip()
        if not query:
            self._update_status("Предупреждение: SQL запрос пуст", "warning")
            return
        
        # Обновляем статус
        self._update_status("Выполнение SQL запроса...", "info")
        
        # Выполняем запрос
        results, error = self.db_conn.execute_query(query)
        
        if error:
            self._update_status(f"Ошибка SQL: {error}", "error")
            return
        
        # Очищаем результаты
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Отображаем результаты
        if results:
            # Получаем колонки из первой записи
            columns = list(results[0].keys())
            self.results_tree['columns'] = columns
            self.results_tree['show'] = 'headings'
            
            # Настраиваем заголовки
            for col in columns:
                self.results_tree.heading(col, text=col)
                self.results_tree.column(col, width=150, minwidth=100)
            
            # Добавляем данные
            for row in results:
                values = [str(row.get(col, '')) for col in columns]
                self.results_tree.insert("", tk.END, values=values)
            
            self._update_status(f"Запрос выполнен. Найдено записей: {len(results)}", "success")
        else:
            self._update_status("Запрос выполнен успешно", "success")
            # Обновляем данные таблицы, если была выбрана таблица
            if self.current_table:
                self._load_table_data()
    
    def _load_sql_file(self):
        """Загружает SQL файл в редактор"""
        self._update_status("Выбор SQL файла...", "info")
        
        sql_file = filedialog.askopenfilename(
            title="Выберите SQL файл",
            filetypes=[
                ("SQL файлы", "*.sql"),
                ("Текстовые файлы", "*.txt"),
                ("Все файлы", "*.*")
            ],
            initialdir=str(ROOT)
        )
        
        if not sql_file:
            self._update_status("Загрузка SQL файла отменена", "info")
            return
        
        try:
            # Читаем файл с поддержкой разных кодировок
            encodings = ['utf-8', 'utf-8-sig', 'cp1251', 'latin-1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(sql_file, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                self._update_status(f"Ошибка: Не удалось прочитать файл '{sql_file}'", "error")
                return
            
            # Заменяем содержимое редактора
            self.sql_text.delete(1.0, tk.END)
            self.sql_text.insert(1.0, content)
            
            file_name = os.path.basename(sql_file)
            self._update_status(f"SQL файл загружен: {file_name}", "success")
            
        except Exception as e:
            self._update_status(f"Ошибка загрузки SQL файла: {e}", "error")
    
    def _save_sql_query(self):
        """Сохраняет текущий SQL запрос в файл"""
        query = self.sql_text.get(1.0, tk.END).strip()
        
        if not query:
            self._update_status("Предупреждение: SQL запрос пуст, нечего сохранять", "warning")
            return
        
        self._update_status("Сохранение SQL запроса...", "info")
        
        sql_file = filedialog.asksaveasfilename(
            title="Сохранить SQL запрос",
            defaultextension=".sql",
            filetypes=[
                ("SQL файлы", "*.sql"),
                ("Текстовые файлы", "*.txt"),
                ("Все файлы", "*.*")
            ],
            initialdir=str(ROOT)
        )
        
        if not sql_file:
            self._update_status("Сохранение SQL запроса отменено", "info")
            return
        
        try:
            with open(sql_file, 'w', encoding='utf-8') as f:
                f.write(query)
            
            file_name = os.path.basename(sql_file)
            self._update_status(f"SQL запрос сохранен: {file_name}", "success")
            
        except Exception as e:
            self._update_status(f"Ошибка сохранения SQL запроса: {e}", "error")
    
    def _add_record(self):
        """Открывает диалог для добавления новой записи"""
        if not self.db_conn or not self.current_table:
            self._update_status("Предупреждение: Выберите таблицу", "warning")
            return
        
        self._update_status("Открытие диалога добавления записи...", "info")
        RecordDialog(self, self.db_conn, self.current_table, mode='add', callback=self._load_table_data)
    
    def _edit_record(self):
        """Открывает диалог для редактирования записи"""
        if not self.db_conn or not self.current_table:
            self._update_status("Предупреждение: Выберите таблицу", "warning")
            return
        
        selection = self.data_tree.selection()
        if not selection:
            self._update_status("Предупреждение: Выберите запись для редактирования", "warning")
            return
        
        # Получаем данные выбранной записи
        item = selection[0]
        values = self.data_tree.item(item, "values")
        columns = self.data_tree['columns']
        
        record = {col: values[i] for i, col in enumerate(columns)}
        
        self._update_status("Открытие диалога редактирования записи...", "info")
        RecordDialog(self, self.db_conn, self.current_table, mode='edit', record=record, callback=self._load_table_data)
    
    def _delete_record(self):
        """Удаляет выбранную запись"""
        if not self.db_conn or not self.current_table:
            self._update_status("Предупреждение: Выберите таблицу", "warning")
            return
        
        selection = self.data_tree.selection()
        if not selection:
            self._update_status("Предупреждение: Выберите запись для удаления", "warning")
            return
        
        # Подтверждение
        if not messagebox.askyesno("Подтверждение", "Вы уверены, что хотите удалить эту запись?"):
            self._update_status("Удаление отменено", "info")
            return
        
        # Получаем данные выбранной записи
        item = selection[0]
        values = self.data_tree.item(item, "values")
        columns = self.data_tree['columns']
        
        # Получаем схему таблицы для определения первичного ключа
        schema = self.db_conn.get_table_schema(self.current_table)
        pk_columns = [col['name'] for col in schema if col['pk']]
        
        if not pk_columns:
            messagebox.showerror("Ошибка", "Не найдено первичного ключа для удаления записи")
            return
        
        # Формируем WHERE условие
        conditions = []
        params = []
        for pk_col in pk_columns:
            col_idx = columns.index(pk_col)
            conditions.append(f"{pk_col} = ?")
            params.append(values[col_idx])
        
        where_clause = " AND ".join(conditions)
        query = f"DELETE FROM {self.current_table} WHERE {where_clause}"
        
        # Выполняем удаление
        results, error = self.db_conn.execute_query(query, tuple(params))
        
        if error:
            self._update_status(f"Ошибка удаления записи: {error}", "error")
        else:
            self._update_status("Запись удалена", "success")
            self._load_table_data()
    
    def _refresh_databases(self):
        """Обновляет список БД"""
        self._update_status("Поиск баз данных...", "info")
        self._auto_discover_databases()
        self._update_status("Список баз данных обновлен", "success")
    
    def _refresh_tables_in_tree(self, db_item_id):
        """Обновляет список таблиц в дереве для указанной БД"""
        item_info = self.db_tree_items.get(db_item_id)
        if not item_info or item_info['type'] != 'db':
            return
        
        db_path = item_info['db_path']
        self._load_tables_into_tree(db_item_id, db_path)
        self._update_status("Список таблиц обновлен", "success")
    
    def _create_table(self, db_info: Dict):
        """Создает новую таблицу"""
        db_path = db_info['db_path']
        
        # Открываем БД если она не открыта
        if not self.db_conn or self.db_conn.db_path != db_path:
            self._open_database(db_path)
        
        if not self.db_conn:
            self._update_status("Ошибка: Не удалось открыть базу данных", "error")
            return
        
        # Открываем диалог создания таблицы
        TableDialog(self, self.db_conn, mode='create', callback=self._on_table_created)
    
    def _edit_table(self, table_info: Dict):
        """Редактирует таблицу"""
        db_path = table_info['db_path']
        table_name = table_info['table_name']
        
        # Открываем БД если она не открыта
        if not self.db_conn or self.db_conn.db_path != db_path:
            self._open_database(db_path)
        
        if not self.db_conn:
            self._update_status("Ошибка: Не удалось открыть базу данных", "error")
            return
        
        # Получаем схему таблицы
        schema = self.db_conn.get_table_schema(table_name)
        if not schema:
            self._update_status(f"Ошибка: Не удалось получить схему таблицы '{table_name}'", "error")
            return
        
        # Открываем диалог редактирования таблицы
        TableDialog(self, self.db_conn, mode='edit', table_name=table_name, schema=schema, callback=self._on_table_modified)
    
    def _truncate_table(self, table_info: Dict):
        """Обнуляет таблицу (удаляет все данные, пересоздавая таблицу)"""
        db_path = table_info['db_path']
        table_name = table_info['table_name']
        
        # Подтверждение
        if not messagebox.askyesno(
            "Подтверждение обнуления",
            f"Вы уверены, что хотите обнулить таблицу '{table_name}'?\n\n"
            f"Все данные в таблице будут удалены, но структура таблицы сохранится.\n"
            f"Это действие необратимо!\n\n"
            f"Для больших таблиц операция может занять некоторое время."
        ):
            self._update_status("Обнуление таблицы отменено", "info")
            return
        
        # Открываем БД если она не открыта
        if not self.db_conn or self.db_conn.db_path != db_path:
            self._open_database(db_path)
        
        if not self.db_conn:
            self._update_status("Ошибка: Не удалось открыть базу данных", "error")
            return
        
        # Запускаем операцию в отдельном потоке с диалогом прогресса
        self._truncate_table_async(db_path, table_name)
    
    def _truncate_table_async(self, db_path: str, table_name: str):
        """Обнуляет таблицу в отдельном потоке"""
        # Создаем диалог прогресса
        progress_dialog = TruncateProgressDialog(self, table_name)
        progress_dialog.update()
        
        def truncate_worker():
            """Рабочий поток для обнуления таблицы"""
            try:
                # Создаем отдельное подключение для фонового потока
                temp_conn = sqlite3.connect(db_path, timeout=300.0)  # Увеличиваем timeout для больших операций
                temp_conn.execute("PRAGMA journal_mode=WAL")  # Используем WAL для лучшей производительности
                temp_conn.execute("PRAGMA synchronous=NORMAL")  # Ускоряем запись
                
                cursor = temp_conn.cursor()
                
                # Обновляем прогресс
                self.after(0, lambda: progress_dialog.update_status("Получение структуры таблицы..."))
                
                # Получаем CREATE TABLE запрос
                cursor.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,)
                )
                create_sql = cursor.fetchone()
                
                if not create_sql or not create_sql[0]:
                    raise Exception(f"Не удалось получить CREATE TABLE запрос для '{table_name}'")
                
                # Для очень больших таблиц используем DELETE FROM вместо DROP/CREATE
                # Это быстрее, так как не требует пересоздания индексов
                self.after(0, lambda: progress_dialog.update_status("Удаление данных из таблицы..."))
                
                # Выполняем в транзакции
                temp_conn.execute("BEGIN IMMEDIATE TRANSACTION")
                
                try:
                    # Используем DELETE FROM для больших таблиц (быстрее чем DROP)
                    delete_query = f"DELETE FROM {table_name}"
                    cursor.execute(delete_query)
                    
                    # Обновляем прогресс
                    self.after(0, lambda: progress_dialog.update_status("Завершение операции..."))
                    
                    # Коммитим транзакцию
                    temp_conn.commit()
                    
                    temp_conn.close()
                    
                    # Закрываем диалог и обновляем UI в главном потоке
                    self.after(0, lambda: progress_dialog.close())
                    self.after(0, lambda: self._truncate_table_completed(db_path, table_name, None))
                    
                except Exception as e:
                    temp_conn.rollback()
                    temp_conn.close()
                    raise e
                    
            except Exception as e:
                error_msg = str(e)
                # Закрываем диалог и обновляем UI в главном потоке
                self.after(0, lambda: progress_dialog.close())
                self.after(0, lambda: self._truncate_table_completed(db_path, table_name, error_msg))
        
        # Запускаем поток
        thread = threading.Thread(target=truncate_worker, daemon=True)
        thread.start()
        
        # Диалог будет закрыт автоматически после завершения операции
    
    def _truncate_table_completed(self, db_path: str, table_name: str, error: Optional[str]):
        """Обработчик завершения обнуления таблицы"""
        if error:
            self._update_status(f"Ошибка обнуления таблицы: {error}", "error")
            messagebox.showerror("Ошибка", f"Не удалось обнулить таблицу '{table_name}':\n{error}")
        else:
            self._update_status(f"Таблица '{table_name}' обнулена (все данные удалены)", "success")
            
            # Обновляем список таблиц в дереве
            self._refresh_tables_after_modification(db_path)
            
            # Обновляем список таблиц в комбобоксе
            if self.db_conn and self.db_conn.db_path == db_path:
                self._load_tables_list()
                
                # Если это текущая таблица - перезагружаем данные
                if self.current_table == table_name:
                    self._load_table_data()
    
    def _delete_table(self, table_info: Dict):
        """Удаляет таблицу"""
        db_path = table_info['db_path']
        table_name = table_info['table_name']
        
        # Подтверждение
        if not messagebox.askyesno(
            "Подтверждение удаления",
            f"Вы уверены, что хотите удалить таблицу '{table_name}'?\n\n"
            f"Это действие необратимо! Все данные в таблице будут потеряны."
        ):
            self._update_status("Удаление таблицы отменено", "info")
            return
        
        # Открываем БД если она не открыта
        if not self.db_conn or self.db_conn.db_path != db_path:
            self._open_database(db_path)
        
        if not self.db_conn:
            self._update_status("Ошибка: Не удалось открыть базу данных", "error")
            return
        
        # Удаляем таблицу
        query = f"DROP TABLE IF EXISTS {table_name}"
        results, error = self.db_conn.execute_query(query)
        
        if error:
            self._update_status(f"Ошибка удаления таблицы: {error}", "error")
        else:
            self._update_status(f"Таблица '{table_name}' удалена", "success")
            # Обновляем список таблиц в дереве
            self._refresh_tables_after_modification(db_path)
            # Обновляем список таблиц в комбобоксе
            self._load_tables_list()
    
    def _on_table_created(self, table_name: str):
        """Обработчик создания таблицы"""
        self._update_status(f"Таблица '{table_name}' создана", "success")
        # Обновляем список таблиц
        if self.db_conn:
            self._refresh_tables_after_modification(self.db_conn.db_path)
            self._load_tables_list()
    
    def _on_table_modified(self, table_name: str):
        """Обработчик изменения таблицы"""
        self._update_status(f"Таблица '{table_name}' изменена", "success")
        # Обновляем список таблиц
        if self.db_conn:
            self._refresh_tables_after_modification(self.db_conn.db_path)
            self._load_tables_list()
            # Перезагружаем данные если это текущая таблица
            if self.current_table == table_name:
                self._load_table_data()
    
    def _refresh_tables_for_opened_db(self, db_path: str):
        """Обновляет таблицы в дереве для открытой БД"""
        # Находим элемент БД в дереве
        for item_id, item_info in self.db_tree_items.items():
            if item_info['type'] == 'db' and item_info['db_path'] == db_path:
                # Раскрываем узел если он закрыт
                if not self.db_tree.item(item_id, 'open'):
                    self.db_tree.item(item_id, open=True)
                # Загружаем таблицы
                self._load_tables_into_tree(item_id, db_path)
                break
    
    def _refresh_tables_after_modification(self, db_path: str):
        """Обновляет список таблиц в дереве после изменения"""
        # Находим элемент БД в дереве
        for item_id, item_info in self.db_tree_items.items():
            if item_info['type'] == 'db' and item_info['db_path'] == db_path:
                self._load_tables_into_tree(item_id, db_path)
                break
    
    def _prev_page(self):
        """Переход на предыдущую страницу"""
        if self.current_page > 1:
            self.current_page -= 1
            self._load_table_data()
    
    def _next_page(self):
        """Переход на следующую страницу"""
        if self.current_page < self.total_pages:
            self.current_page += 1
            self._load_table_data()
    
    def _go_to_page(self, page: int):
        """Переход на указанную страницу"""
        if 1 <= page <= self.total_pages:
            self.current_page = page
            self._load_table_data()
    
    def _on_page_entered(self, event=None):
        """Обработчик ввода номера страницы"""
        try:
            page = int(self.page_entry_var.get())
            self._go_to_page(page)
        except ValueError:
            self.page_entry_var.set(str(self.current_page))
    
    def _update_pagination_info(self):
        """Обновляет информацию о пагинации"""
        self.page_info_var.set(f"Стр. {self.current_page}/{self.total_pages}")
        self.page_entry_var.set(str(self.current_page))
    
    def _on_records_per_page_changed(self, event=None):
        """Обработчик изменения количества записей на странице"""
        value = self.records_per_page_var.get()
        if value == 'Все':
            self.records_per_page = 0  # 0 означает "все записи"
        else:
            try:
                self.records_per_page = int(value)
            except ValueError:
                self.records_per_page = 100
        
        self.current_page = 1
        self._save_settings()
        self._load_table_data()
    
    def _on_custom_records_entered(self, event=None):
        """Обработчик ввода произвольного количества записей"""
        try:
            value = int(self.custom_records_var.get())
            if value > 0:
                self.records_per_page = value
                self.records_per_page_var.set(str(value))
                self.current_page = 1
                self._save_settings()
                self._load_table_data()
        except ValueError:
            self.custom_records_var.set("")
    
    def _load_settings(self) -> Dict:
        """Загружает настройки из файла"""
        if not self.settings_file.exists():
            return {}
        
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки настроек: {e}")
            return {}
    
    def _save_settings(self):
        """Сохраняет настройки в файл"""
        try:
            # Создаем директорию если её нет
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем текущие настройки
            self.settings = {
                'records_per_page': self.records_per_page,
                'geometry': self.geometry(),
                'panels': {}  # TODO: сохранять положение панелей
            }
            
            # Сохраняем положение панелей
            for panel_name, panel in self.panels.items():
                if hasattr(panel, 'canvas_id') and panel.canvas_id:
                    try:
                        coords = self.main_canvas.coords(panel.canvas_id)
                        size = self.main_canvas.itemcget(panel.canvas_id, 'width'), self.main_canvas.itemcget(panel.canvas_id, 'height')
                        self.settings['panels'][panel_name] = {
                            'x': float(coords[0]) if coords and len(coords) > 0 else 0,
                            'y': float(coords[1]) if coords and len(coords) > 1 else 0,
                            'width': float(size[0]) if size[0] != '' else 300,
                            'height': float(size[1]) if size[1] != '' else 200
                        }
                    except:
                        pass
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения настроек: {e}")
    
    def on_closing(self):
        """Обработчик закрытия окна"""
        self._save_settings()
        if self.db_conn:
            self.db_conn.disconnect()
        self.destroy()


class RecordDialog(tk.Toplevel):
    """Диалог для добавления/редактирования записи"""
    
    def __init__(self, parent, db_conn: DatabaseConnection, table_name: str, mode: str = 'add', record: Dict = None, callback=None):
        super().__init__(parent)
        
        self.db_conn = db_conn
        self.table_name = table_name
        self.mode = mode
        self.record = record or {}
        self.callback = callback
        
        self.title(f"{'Редактирование' if mode == 'edit' else 'Добавление'} записи: {table_name}")
        self.geometry("600x500")
        self.resizable(True, True)
        
        # Переменные для полей
        self.field_vars = {}
        
        # Создаем интерфейс
        self._build_ui()
        
        # Фокусируемся на этом окне
        self.transient(parent)
        self.grab_set()
    
    def _build_ui(self):
        """Создает интерфейс диалога"""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Получаем схему таблицы
        schema = self.db_conn.get_table_schema(self.table_name)
        
        # Контейнер с прокруткой для полей
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Создаем поля для каждой колонки (кроме AUTOINCREMENT)
        fields_frame = scrollable_frame
        
        row = 0
        for col in schema:
            col_name = col['name']
            col_type = col['type'].upper()
            
            # Пропускаем AUTOINCREMENT колонки при добавлении
            if self.mode == 'add' and col['pk'] and 'INTEGER' in col_type:
                continue
            
            # Создаем label
            label = ttk.Label(fields_frame, text=f"{col_name}:")
            label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
            
            # Создаем поле ввода
            if 'TEXT' in col_type or 'VARCHAR' in col_type or 'CHAR' in col_type:
                # Текстовое поле
                if self.mode == 'edit' and col['pk']:
                    # Первичный ключ - только чтение при редактировании
                    entry = ttk.Entry(fields_frame, state='readonly')
                    entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
                    var = tk.StringVar(value=str(self.record.get(col_name, '')))
                    entry.config(textvariable=var)
                else:
                    entry = ttk.Entry(fields_frame, width=40)
                    entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
                    var = tk.StringVar(value=str(self.record.get(col_name, '')))
                    entry.config(textvariable=var)
            elif 'INTEGER' in col_type or 'REAL' in col_type or 'NUMERIC' in col_type:
                # Числовое поле
                entry = ttk.Entry(fields_frame, width=40)
                entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
                var = tk.StringVar(value=str(self.record.get(col_name, '')))
                entry.config(textvariable=var)
            else:
                # По умолчанию текстовое поле
                entry = ttk.Entry(fields_frame, width=40)
                entry.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
                var = tk.StringVar(value=str(self.record.get(col_name, '')))
                entry.config(textvariable=var)
            
            fields_frame.columnconfigure(1, weight=1)
            self.field_vars[col_name] = var
            row += 1
        
        # Упаковываем canvas и scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Привязываем прокрутку колесом мыши
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Кнопки
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            buttons_frame,
            text="Сохранить",
            command=self._save_record
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="Отмена",
            command=self.destroy
        ).pack(side=tk.RIGHT, padx=5)
    
    def _save_record(self):
        """Сохраняет запись"""
        schema = self.db_conn.get_table_schema(self.table_name)
        
        if self.mode == 'add':
            # INSERT
            columns = []
            values = []
            params = []
            
            for col in schema:
                col_name = col['name']
                col_type = col['type'].upper()
                
                # Пропускаем AUTOINCREMENT
                if col['pk'] and 'INTEGER' in col_type:
                    continue
                
                # Проверяем наличие поля (может быть пропущено для AUTOINCREMENT)
                if col_name not in self.field_vars:
                    continue
                
                value = self.field_vars[col_name].get().strip()
                
                # Если поле пустое и NOT NULL - пропускаем (будет ошибка от БД)
                if not value and col['notnull']:
                    continue
                
                columns.append(col_name)
                values.append("?")
                
                # Преобразуем значение по типу
                if value:
                    if 'INTEGER' in col_type or 'INT' in col_type:
                        try:
                            params.append(int(value))
                        except:
                            params.append(value)
                    elif 'REAL' in col_type or 'FLOAT' in col_type or 'DOUBLE' in col_type:
                        try:
                            params.append(float(value))
                        except:
                            params.append(value)
                    else:
                        params.append(value)
                else:
                    params.append(None)
            
            query = f"INSERT INTO {self.table_name} ({', '.join(columns)}) VALUES ({', '.join(values)})"
        else:
            # UPDATE
            pk_columns = [col['name'] for col in schema if col['pk']]
            
            if not pk_columns:
                messagebox.showerror("Ошибка", "Не найдено первичного ключа для обновления")
                return
            
            set_clauses = []
            params = []
            
            for col in schema:
                col_name = col['name']
                col_type = col['type'].upper()
                
                # Пропускаем первичный ключ
                if col_name in pk_columns:
                    continue
                
                # Проверяем наличие поля
                if col_name not in self.field_vars:
                    continue
                
                value = self.field_vars[col_name].get().strip()
                set_clauses.append(f"{col_name} = ?")
                
                # Преобразуем значение по типу
                if value:
                    if 'INTEGER' in col_type or 'INT' in col_type:
                        try:
                            params.append(int(value))
                        except:
                            params.append(value)
                    elif 'REAL' in col_type or 'FLOAT' in col_type or 'DOUBLE' in col_type:
                        try:
                            params.append(float(value))
                        except:
                            params.append(value)
                    else:
                        params.append(value)
                else:
                    params.append(None)
            
            # WHERE условие
            where_clauses = []
            for pk_col in pk_columns:
                where_clauses.append(f"{pk_col} = ?")
                params.append(self.record.get(pk_col))
            
            query = f"UPDATE {self.table_name} SET {', '.join(set_clauses)} WHERE {' AND '.join(where_clauses)}"
        
        # Выполняем запрос
        results, error = self.db_conn.execute_query(query, tuple(params))
        
        if error:
            messagebox.showerror("Ошибка", f"Ошибка сохранения записи:\n{error}")
        else:
            # Обновляем статус в родительском окне
            if self.callback:
                parent = self.master
                if hasattr(parent, '_update_status'):
                    parent._update_status("Запись сохранена", "success")
            if self.callback:
                self.callback()
            self.destroy()


class TableDialog(tk.Toplevel):
    """Диалог для создания/редактирования таблицы"""
    
    def __init__(self, parent, db_conn: DatabaseConnection, mode: str = 'create', table_name: str = None, schema: List[Dict] = None, callback=None):
        super().__init__(parent)
        
        self.db_conn = db_conn
        self.mode = mode
        self.table_name = table_name
        self.schema = schema or []
        self.callback = callback
        
        self.title(f"{'Редактирование' if mode == 'edit' else 'Создание'} таблицы")
        self.geometry("700x600")
        self.resizable(True, True)
        
        # Переменные для полей
        self.table_name_var = tk.StringVar(value=table_name or "")
        self.columns_data = []  # Список словарей с данными колонок
        
        # Если редактирование - загружаем существующие колонки
        if mode == 'edit' and schema:
            for col in schema:
                self.columns_data.append({
                    'name': col['name'],
                    'type': col['type'],
                    'notnull': col['notnull'],
                    'dflt_value': col['dflt_value'],
                    'pk': col['pk']
                })
        
        # Создаем интерфейс
        self._build_ui()
        
        # Фокусируемся на этом окне
        self.transient(parent)
        self.grab_set()
    
    def _build_ui(self):
        """Создает интерфейс диалога"""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Название таблицы
        name_frame = ttk.Frame(main_frame)
        name_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(name_frame, text="Название таблицы:", font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        name_entry = ttk.Entry(name_frame, textvariable=self.table_name_var, width=30, font=('Segoe UI', 9))
        name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        if self.mode == 'edit':
            name_entry.config(state='readonly')
        
        # Колонки
        columns_label = ttk.Label(main_frame, text="Колонки:", font=('Segoe UI', 9, 'bold'))
        columns_label.pack(anchor=tk.W, pady=(10, 5))
        
        # Контейнер с прокруткой для колонок
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        canvas = tk.Canvas(canvas_frame, bg='white')
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Заголовки колонок
        headers_frame = ttk.Frame(scrollable_frame)
        headers_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(headers_frame, text="Название", font=('Segoe UI', 8, 'bold'), width=15).grid(row=0, column=0, padx=2)
        ttk.Label(headers_frame, text="Тип", font=('Segoe UI', 8, 'bold'), width=12).grid(row=0, column=1, padx=2)
        ttk.Label(headers_frame, text="NOT NULL", font=('Segoe UI', 8, 'bold'), width=8).grid(row=0, column=2, padx=2)
        ttk.Label(headers_frame, text="PK", font=('Segoe UI', 8, 'bold'), width=5).grid(row=0, column=3, padx=2)
        ttk.Label(headers_frame, text="По умолчанию", font=('Segoe UI', 8, 'bold'), width=15).grid(row=0, column=4, padx=2)
        ttk.Label(headers_frame, text="", font=('Segoe UI', 8, 'bold'), width=5).grid(row=0, column=5, padx=2)
        
        self.columns_frame = scrollable_frame
        self.columns_widgets = []  # Список виджетов для каждой колонки
        
        # Загружаем существующие колонки или добавляем одну пустую
        if self.columns_data:
            for col_data in self.columns_data:
                self._add_column_row(col_data)
        else:
            self._add_column_row()
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Привязываем прокрутку колесом мыши
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Кнопки управления колонками
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            buttons_frame,
            text="+ Добавить колонку",
            command=self._add_column_row,
            style='TDefault.TButton'
        ).pack(side=tk.LEFT, padx=5)
        
        # Кнопки сохранения/отмены
        save_frame = ttk.Frame(main_frame)
        save_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            save_frame,
            text="Сохранить",
            command=self._save_table,
            style='TPrimary.TButton'
        ).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            save_frame,
            text="Отмена",
            command=self.destroy,
            style='TDefault.TButton'
        ).pack(side=tk.RIGHT, padx=5)
    
    def _add_column_row(self, col_data: Dict = None):
        """Добавляет строку для редактирования колонки"""
        row_frame = ttk.Frame(self.columns_frame)
        row_frame.pack(fill=tk.X, padx=5, pady=2)
        
        widgets = {}
        
        # Название
        name_var = tk.StringVar(value=col_data.get('name', '') if col_data else '')
        name_entry = ttk.Entry(row_frame, textvariable=name_var, width=15)
        name_entry.grid(row=0, column=0, padx=2, sticky="ew")
        widgets['name'] = name_var
        
        # Тип
        type_var = tk.StringVar(value=col_data.get('type', 'TEXT') if col_data else 'TEXT')
        type_combo = ttk.Combobox(
            row_frame,
            textvariable=type_var,
            values=['TEXT', 'INTEGER', 'REAL', 'BLOB', 'NUMERIC', 'BOOLEAN', 'DATE', 'DATETIME'],
            width=12,
            state='readonly'
        )
        type_combo.grid(row=0, column=1, padx=2, sticky="ew")
        widgets['type'] = type_var
        
        # NOT NULL
        notnull_var = tk.BooleanVar(value=col_data.get('notnull', False) if col_data else False)
        notnull_check = ttk.Checkbutton(row_frame, variable=notnull_var)
        notnull_check.grid(row=0, column=2, padx=2)
        widgets['notnull'] = notnull_var
        
        # PRIMARY KEY
        pk_var = tk.BooleanVar(value=col_data.get('pk', False) if col_data else False)
        pk_check = ttk.Checkbutton(row_frame, variable=pk_var)
        pk_check.grid(row=0, column=3, padx=2)
        widgets['pk'] = pk_var
        
        # По умолчанию
        default_var = tk.StringVar(value=str(col_data.get('dflt_value', '')) if col_data and col_data.get('dflt_value') else '')
        default_entry = ttk.Entry(row_frame, textvariable=default_var, width=15)
        default_entry.grid(row=0, column=4, padx=2, sticky="ew")
        widgets['default'] = default_var
        
        # Кнопка удаления
        def remove_row():
            row_frame.destroy()
            self.columns_widgets.remove(widgets)
        
        remove_btn = ttk.Button(row_frame, text="×", command=remove_row, width=3)
        remove_btn.grid(row=0, column=5, padx=2)
        
        # Настраиваем веса колонок
        row_frame.columnconfigure(0, weight=1)
        row_frame.columnconfigure(4, weight=1)
        
        self.columns_widgets.append(widgets)
    
    def _save_table(self):
        """Сохраняет таблицу"""
        table_name = self.table_name_var.get().strip()
        
        if not table_name:
            messagebox.showerror("Ошибка", "Введите название таблицы")
            return
        
        # Проверяем валидность названия таблицы
        if not table_name.replace('_', '').isalnum():
            messagebox.showerror("Ошибка", "Название таблицы может содержать только буквы, цифры и подчеркивания")
            return
        
        # Собираем данные колонок
        columns = []
        pk_columns = []
        
        for widgets in self.columns_widgets:
            col_name = widgets['name'].get().strip()
            if not col_name:
                continue  # Пропускаем пустые колонки
            
            col_type = widgets['type'].get()
            notnull = widgets['notnull'].get()
            pk = widgets['pk'].get()
            default = widgets['default'].get().strip()
            
            # Формируем определение колонки
            col_def = f"{col_name} {col_type}"
            
            if pk:
                col_def += " PRIMARY KEY"
                pk_columns.append(col_name)
            
            if notnull and not pk:
                col_def += " NOT NULL"
            
            if default:
                # Экранируем значение по умолчанию
                if col_type in ['TEXT', 'VARCHAR', 'CHAR']:
                    default = f"'{default.replace("'", "''")}'"
                col_def += f" DEFAULT {default}"
            
            columns.append(col_def)
        
        if not columns:
            messagebox.showerror("Ошибка", "Добавьте хотя бы одну колонку")
            return
        
        # Формируем SQL запрос
        if self.mode == 'create':
            query = f"CREATE TABLE {table_name} (\n    {',\n    '.join(columns)}\n)"
        else:
            # Для редактирования - показываем предупреждение
            if messagebox.askyesno(
                "Внимание",
                "Редактирование таблицы через ALTER TABLE ограничено.\n\n"
                "Можно только:\n"
                "- Добавить новую колонку\n"
                "- Переименовать таблицу\n\n"
                "Для изменения существующих колонок рекомендуется:\n"
                "1. Создать новую таблицу с нужной структурой\n"
                "2. Скопировать данные\n"
                "3. Удалить старую таблицу\n"
                "4. Переименовать новую\n\n"
                "Продолжить с добавлением колонок?"
            ):
                # Добавляем только новые колонки (упрощенная версия)
                new_columns = []
                existing_column_names = [col['name'] for col in self.schema]
                
                for widgets in self.columns_widgets:
                    col_name = widgets['name'].get().strip()
                    if not col_name or col_name in existing_column_names:
                        continue
                    
                    col_type = widgets['type'].get()
                    notnull = widgets['notnull'].get()
                    default = widgets['default'].get().strip()
                    
                    col_def = f"{col_name} {col_type}"
                    if notnull:
                        col_def += " NOT NULL"
                    if default:
                        if col_type in ['TEXT', 'VARCHAR', 'CHAR']:
                            default = f"'{default.replace("'", "''")}'"
                        col_def += f" DEFAULT {default}"
                    
                    new_columns.append(col_def)
                
                if not new_columns:
                    messagebox.showinfo("Информация", "Нет новых колонок для добавления")
                    return
                
                # Выполняем ALTER TABLE для каждой новой колонки
                for col_def in new_columns:
                    alter_query = f"ALTER TABLE {table_name} ADD COLUMN {col_def}"
                    results, error = self.db_conn.execute_query(alter_query)
                    if error:
                        messagebox.showerror("Ошибка", f"Ошибка добавления колонки:\n{error}")
                        return
                
                # Обновляем статус
                if hasattr(self.master, '_update_status'):
                    self.master._update_status(f"В таблицу '{table_name}' добавлены колонки", "success")
                
                if self.callback:
                    self.callback(table_name)
                self.destroy()
                return
        
        # Выполняем запрос
        results, error = self.db_conn.execute_query(query)
        
        if error:
            messagebox.showerror("Ошибка", f"Ошибка создания таблицы:\n{error}")
        else:
            # Обновляем статус в родительском окне
            if hasattr(self.master, '_update_status'):
                self.master._update_status(f"Таблица '{table_name}' {'создана' if self.mode == 'create' else 'изменена'}", "success")
            
            if self.callback:
                self.callback(table_name)
            self.destroy()


class TruncateProgressDialog(tk.Toplevel):
    """Диалог прогресса для обнуления таблицы"""
    
    def __init__(self, parent, table_name: str):
        super().__init__(parent)
        
        self.table_name = table_name
        self.is_closed = False
        
        self.title(f"Обнуление таблицы: {table_name}")
        self.geometry("500x150")
        self.resizable(False, False)
        
        # Центрируем окно
        self.transient(parent)
        self.grab_set()
        
        # Запрещаем закрытие во время операции
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # Создаем интерфейс
        self._build_ui()
        
        # Центрируем окно
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (self.winfo_width() // 2)
        y = (self.winfo_screenheight() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")
    
    def _build_ui(self):
        """Создает интерфейс диалога"""
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        title_label = ttk.Label(
            main_frame,
            text=f"Обнуление таблицы: {self.table_name}",
            font=('Segoe UI', 10, 'bold')
        )
        title_label.pack(pady=(0, 10))
        
        # Статус
        self.status_label = ttk.Label(
            main_frame,
            text="Подготовка...",
            font=('Segoe UI', 9)
        )
        self.status_label.pack(pady=5)
        
        # Прогресс-бар
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=400
        )
        self.progress.pack(pady=10)
        self.progress.start(10)
        
        # Информация
        info_label = ttk.Label(
            main_frame,
            text="Пожалуйста, подождите. Операция может занять некоторое время...",
            font=('Segoe UI', 8),
            foreground='gray'
        )
        info_label.pack(pady=5)
    
    def update_status(self, message: str):
        """Обновляет статус операции"""
        if not self.is_closed:
            try:
                self.status_label.config(text=message)
                self.update()
            except:
                pass  # Окно уже закрыто
    
    def close(self):
        """Закрывает диалог"""
        if not self.is_closed:
            self.is_closed = True
            try:
                self.progress.stop()
                self.grab_release()
                self.destroy()
            except:
                pass


def main():
    """Главная функция"""
    app = DatabaseGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()

