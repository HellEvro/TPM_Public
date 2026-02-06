#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI интерфейс для синхронизации и коммита в Git репозитории"""

import os
import subprocess
import sys
import threading
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
except ImportError:
    print("Ошибка: требуется tkinter. Установите python3-tk или используйте Python с поддержкой tkinter.")
    sys.exit(1)

# Настройка кодировки для Windows консоли
if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        try:
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
        except:
            pass

ROOT = Path(__file__).parent.parent
PUBLIC = ROOT.parent / "InfoBot_Public"
SYNC_SCRIPT = ROOT / "sync_to_public.py"
GIT_COMMIT_SCRIPT = ROOT / "scripts" / "git_commit_push.py"


class GitCommitGUI(tk.Tk):
    """GUI для коммита и пуша в Git"""
    
    def __init__(self):
        super().__init__()
        
        self.title("Git Commit & Push")
        self.geometry("800x600")
        self.minsize(600, 400)
        
        # Переменные
        self.commit_message = tk.StringVar()
        self.is_running = False
        self.commit_to_public = tk.BooleanVar(value=True)  # По умолчанию включено
        
        self._build_ui()
        
    def _build_ui(self):
        """Создает интерфейс"""
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        title_label = ttk.Label(
            main_frame,
            text="Синхронизация и коммит в Git",
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Описание коммита
        message_frame = ttk.LabelFrame(main_frame, text="Описание изменений", padding=10)
        message_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        message_text = scrolledtext.ScrolledText(
            message_frame,
            wrap=tk.WORD,
            height=8,
            font=("Consolas", 10)
        )
        message_text.pack(fill=tk.BOTH, expand=True)
        self.message_text = message_text
        
        # Включаем стандартные горячие клавиши для вставки/копирования/вырезания
        def paste_text(event=None):
            try:
                message_text.event_generate('<<Paste>>')
                return 'break'
            except:
                return 'break'
        
        def copy_text(event=None):
            try:
                message_text.event_generate('<<Copy>>')
                return 'break'
            except:
                return 'break'
        
        def cut_text(event=None):
            try:
                message_text.event_generate('<<Cut>>')
                return 'break'
            except:
                return 'break'
        
        # Универсальная обработка Ctrl+клавиша (работает в любой раскладке)
        def handle_ctrl_key(event):
            # Проверяем код клавиши, а не символ (работает в любой раскладке)
            # V = 86, C = 67, X = 88 (коды клавиш не зависят от раскладки)
            if event.state & 0x4:  # Ctrl нажат (0x4 = Control modifier)
                keycode = event.keycode
                if keycode == 86:  # V (или М в русской раскладке)
                    paste_text(event)
                    return 'break'
                elif keycode == 67:  # C (или С в русской раскладке)
                    copy_text(event)
                    return 'break'
                elif keycode == 88:  # X (или Ч в русской раскладке)
                    cut_text(event)
                    return 'break'
            return None
        
        # Горячие клавиши - обработка по коду клавиши (работает в любой раскладке)
        # Это перехватывает Ctrl+V/C/X независимо от раскладки
        message_text.bind('<KeyPress>', handle_ctrl_key, add='+')
        
        # Также оставляем стандартные биндинги для совместимости (английская раскладка)
        message_text.bind('<Control-v>', paste_text)
        message_text.bind('<Control-V>', paste_text)
        message_text.bind('<Control-c>', copy_text)
        message_text.bind('<Control-C>', copy_text)
        message_text.bind('<Control-x>', cut_text)
        message_text.bind('<Control-X>', cut_text)
        
        # Контекстное меню
        context_menu = tk.Menu(message_text, tearoff=0)
        context_menu.add_command(label="Вставить", command=lambda: paste_text())
        context_menu.add_separator()
        context_menu.add_command(label="Копировать", command=lambda: copy_text())
        context_menu.add_command(label="Вырезать", command=lambda: cut_text())
        
        def show_context_menu(event):
            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()
        
        message_text.bind('<Button-3>', show_context_menu)  # Правая кнопка мыши
        if sys.platform == 'darwin':  # Mac
            message_text.bind('<Button-2>', show_context_menu)  # Средняя кнопка мыши
        
        # Галочка для выбора коммита в Public
        options_frame = ttk.Frame(main_frame)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        public_checkbox = ttk.Checkbutton(
            options_frame,
            text="Записать в Public",
            variable=self.commit_to_public
        )
        public_checkbox.pack(side=tk.LEFT)
        
        # Кнопка
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.commit_button = ttk.Button(
            button_frame,
            text="Записать в Git",
            command=self._start_commit_process,
            state=tk.NORMAL
        )
        self.commit_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.cancel_button = ttk.Button(
            button_frame,
            text="Отмена",
            command=self._cancel_process,
            state=tk.DISABLED
        )
        self.cancel_button.pack(side=tk.LEFT)
        
        # Лог выполнения
        log_frame = ttk.LabelFrame(main_frame, text="Лог выполнения", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            height=15,
            font=("Consolas", 9),
            state=tk.DISABLED,
            bg="#f5f5f5"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def _log(self, message: str):
        """Добавляет сообщение в лог"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.update_idletasks()
        
    def _start_commit_process(self):
        """Запускает процесс синхронизации и коммита"""
        message = self.message_text.get("1.0", tk.END).strip()
        
        if not message:
            messagebox.showwarning("Предупреждение", "Введите описание изменений для коммита!")
            return
            
        self.is_running = True
        self.commit_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Запускаем в отдельном потоке
        thread = threading.Thread(target=self._commit_process, args=(message,), daemon=True)
        thread.start()
        
    def _cancel_process(self):
        """Отменяет процесс"""
        if self.is_running:
            self._log("\n[ОТМЕНА] Процесс отменен пользователем")
            self.is_running = False
            self._reset_ui()
            
    def _reset_ui(self):
        """Сбрасывает UI в исходное состояние"""
        self.commit_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
        
    def _commit_process(self, message: str):
        """Выполняет процесс синхронизации и коммита"""
        try:
            # Шаг 1: Синхронизация в Public
            self._log("=" * 80)
            self._log("ШАГ 1: Синхронизация файлов в InfoBot_Public")
            self._log("=" * 80)
            
            if not SYNC_SCRIPT.exists():
                self._log(f"[ОШИБКА] Скрипт синхронизации не найден: {SYNC_SCRIPT}")
                self._show_error("Ошибка", f"Скрипт синхронизации не найден: {SYNC_SCRIPT}")
                self._reset_ui()
                return
                
            result = self._run_script(SYNC_SCRIPT, [])
            if result['returncode'] != 0:
                self._log(f"[ОШИБКА] Синхронизация завершилась с ошибкой")
                self._show_error("Ошибка синхронизации", result.get('stderr', 'Неизвестная ошибка'))
                self._reset_ui()
                return
                
            self._log(result.get('stdout', ''))
            self._log("[OK] Синхронизация завершена успешно\n")
            
            # Шаг 2: Коммит в основном репозитории
            self._log("=" * 80)
            self._log("ШАГ 2: Коммит в основном репозитории")
            self._log("=" * 80)
            
            if not GIT_COMMIT_SCRIPT.exists():
                self._log(f"[ОШИБКА] Скрипт git_commit_push не найден: {GIT_COMMIT_SCRIPT}")
                self._show_error("Ошибка", f"Скрипт git_commit_push не найден: {GIT_COMMIT_SCRIPT}")
                self._reset_ui()
                return
                
            result = self._run_script(GIT_COMMIT_SCRIPT, [message], cwd=ROOT)
            if result['returncode'] != 0:
                self._log(f"[ОШИБКА] Коммит в основном репозитории завершился с ошибкой")
                self._show_error("Ошибка коммита", result.get('stderr', 'Неизвестная ошибка'))
                self._reset_ui()
                return
                
            self._log(result.get('stdout', ''))
            self._log("[OK] Коммит в основном репозитории выполнен успешно\n")
            
            # Шаг 3: Коммит в Public репозитории (только если галочка установлена)
            if self.commit_to_public.get():
                self._log("=" * 80)
                self._log("ШАГ 3: Коммит в InfoBot_Public репозитории")
                self._log("=" * 80)
                
                if not PUBLIC.exists():
                    self._log(f"[ОШИБКА] Папка InfoBot_Public не найдена: {PUBLIC}")
                    self._show_error("Ошибка", f"Папка InfoBot_Public не найдена: {PUBLIC}")
                    self._reset_ui()
                    return
                    
                result = self._run_script(GIT_COMMIT_SCRIPT, [message], cwd=PUBLIC)
                if result['returncode'] != 0:
                    self._log(f"[ОШИБКА] Коммит в Public репозитории завершился с ошибкой")
                    self._show_error("Ошибка коммита", result.get('stderr', 'Неизвестная ошибка'))
                    self._reset_ui()
                    return
                    
                self._log(result.get('stdout', ''))
                self._log("[OK] Коммит в Public репозитории выполнен успешно\n")
            else:
                self._log("[ПРОПУЩЕНО] Коммит в Public репозитории пропущен (галочка не установлена)\n")
            
            # Успешное завершение
            self._log("=" * 80)
            self._log("[УСПЕХ] Все операции выполнены успешно!")
            self._log("=" * 80)
            
            success_msg = "Все операции выполнены успешно!\n\n- Синхронизация в InfoBot_Public\n- Коммит в основном репозитории"
            if self.commit_to_public.get():
                success_msg += "\n- Коммит в InfoBot_Public репозитории"
            else:
                success_msg += "\n- Коммит в InfoBot_Public репозитории (пропущен)"
            
            messagebox.showinfo("Успех", success_msg)
            
        except Exception as e:
            self._log(f"[КРИТИЧЕСКАЯ ОШИБКА] {str(e)}")
            self._show_error("Критическая ошибка", str(e))
        finally:
            self.is_running = False
            self._reset_ui()
            
    def _run_script(self, script_path: Path, args: list, cwd: Path = None) -> dict:
        """Запускает скрипт и возвращает результат"""
        if cwd is None:
            cwd = ROOT
            
        cmd = [sys.executable, str(script_path)] + args
        
        try:
            process = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            return {
                'returncode': process.returncode,
                'stdout': process.stdout,
                'stderr': process.stderr
            }
        except Exception as e:
            return {
                'returncode': 1,
                'stdout': '',
                'stderr': str(e)
            }
            
    def _show_error(self, title: str, message: str):
        """Показывает диалог ошибки"""
        self.after(0, lambda: messagebox.showerror(title, message))


def main():
    """Главная функция"""
    app = GitCommitGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

