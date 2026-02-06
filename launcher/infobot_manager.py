#!/usr/bin/env python3
"""
InfoBot Manager GUI

Provides a cross-platform control panel to:
- Install/upgrade dependencies inside the project virtual environment
- Check for git updates
- Retrieve hardware ID and license instructions
- Launch/stop app.py, bots.py and ai.py services
- Manage license files (.lic) and open configuration folders
"""

from __future__ import annotations

import atexit
import importlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:  # pragma: no cover - tkinter should be available on all supported systems
    print("tkinter is required to run the InfoBot Manager GUI.")
    print("Install the python3-tk package or a Python distribution with tkinter support.")
    raise SystemExit(str(exc))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / ".venv"
# Подавление UserWarning sklearn delayed/Parallel в дочерних процессах (app, bots, ai)
_pw = os.environ.get("PYTHONWARNINGS", "").strip()
_add = "ignore::UserWarning:sklearn.utils.parallel"
os.environ["PYTHONWARNINGS"] = f"{_pw},{_add}" if _pw else _add
# Управляем только .venv. .venv_gpu не создаём и не используем (PyTorch ставится в .venv).
DEFAULT_REMOTE_URL = "https://github.com/HellEvro/TPM_Public.git"
STATE_FILE = PROJECT_ROOT / "launcher" / ".infobot_manager_state.json"
DEFAULT_GEOMETRY = "850x874"


def _check_python_version(python_exec: str) -> tuple[int, int] | None:
    """Проверяет версию Python и возвращает (major, minor) или None"""
    try:
        cmd = python_exec.split() if " " in python_exec else [python_exec]
        result = subprocess.run(
            cmd + ["-c", "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_str = result.stdout.strip()
            parts = version_str.split('.')
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None

def _install_python314_plus() -> bool:
    """Автоматически устанавливает Python 3.14.2+ через системные менеджеры пакетов"""
    if os.name == "nt":
        # Windows: используем winget
        try:
            result = subprocess.run(
                ["winget", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Пробуем установить Python 3.14.2 или выше
                print("[INFO] Установка Python 3.14.2+ через winget...")
                install_result = subprocess.run(
                    ["winget", "install", "--id", "Python.Python.3.14", "--silent", "--accept-package-agreements", "--accept-source-agreements"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if install_result.returncode == 0:
                    print("[OK] Python 3.14.2+ установлен")
                    time.sleep(3)  # Даем время системе обновить PATH
                    return True
                else:
                    print(f"[WARNING] Ошибка установки через winget: {install_result.stderr[:200]}")
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            print(f"[WARNING] Не удалось установить Python через winget: {e}")
    else:
        # Linux/macOS: используем системные менеджеры пакетов
        import platform
        system = platform.system()
        
        if system == "Linux":
            # Определяем менеджер пакетов
            for pkg_mgr, install_cmd in [
                ("apt-get", ["sudo", "apt-get", "update", "-qq", "&&", "sudo", "apt-get", "install", "-y", "python3.14", "python3.14-venv"]),
                ("yum", ["sudo", "yum", "install", "-y", "python3.14"]),
                ("dnf", ["sudo", "dnf", "install", "-y", "python3.14"]),
                ("pacman", ["sudo", "pacman", "-S", "--noconfirm", "python"]),
            ]:
                if shutil.which(pkg_mgr):
                    try:
                        print(f"[INFO] Установка Python 3.14.2+ через {pkg_mgr}...")
                        # Для apt-get нужно разделить команды
                        if pkg_mgr == "apt-get":
                            subprocess.run(["sudo", "apt-get", "update", "-qq"], timeout=60, check=True)
                            subprocess.run(["sudo", "apt-get", "install", "-y", "python3.14", "python3.14-venv"], timeout=300, check=True)
                        else:
                            # Для yum, dnf, pacman
                            subprocess.run(install_cmd, timeout=300, check=True)
                        print("[OK] Python 3.14.2+ установлен")
                        return True
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
                        print(f"[WARNING] Ошибка установки через {pkg_mgr}: {e}")
                        continue
        elif system == "Darwin":  # macOS
            if shutil.which("brew"):
                try:
                    print("[INFO] Установка Python 3.14.2+ через brew...")
                    subprocess.run(["brew", "install", "python@3.14"], timeout=600, check=True)
                    print("[OK] Python 3.14.2+ установлен")
                    return True
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
                    print(f"[WARNING] Ошибка установки через brew: {e}")
    
    return False

def _find_python314_plus() -> str | None:
    """Находит Python 3.14+ в системе, при необходимости устанавливает"""
    commands = [
        ["py", "-3.14"],
        ["python3.14"],
        ["python314"],
        ["python"],
        ["python3"],
    ]
    
    for cmd in commands:
        try:
            # Проверяем версию через Python код
            check_cmd = cmd[0] if len(cmd) == 1 else cmd
            result = subprocess.run(
                [check_cmd] + (cmd[1:] if len(cmd) > 1 else []) + ["-c", "import sys; exit(0 if sys.version_info[:2] >= (3, 14) else 1)"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return " ".join(cmd) if len(cmd) > 1 else cmd[0]
        except Exception:
            continue
    
    # Python 3.14+ не найден - пробуем установить
    print("[INFO] Python 3.14.2+ не найден, пытаемся установить автоматически...")
    if _install_python314_plus():
        # После установки пробуем найти снова
        time.sleep(2)
        for cmd in commands:
            try:
                check_cmd = cmd[0] if len(cmd) == 1 else cmd
                result = subprocess.run(
                    [check_cmd] + (cmd[1:] if len(cmd) > 1 else []) + ["-c", "import sys; exit(0 if sys.version_info[:2] >= (3, 14) else 1)"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return " ".join(cmd) if len(cmd) > 1 else cmd[0]
            except Exception:
                continue
    
    return None

def _detect_python_executable() -> str:
    if os.name == "nt":
        venv_python = VENV_DIR / "Scripts" / "python.exe"
    else:
        venv_python = VENV_DIR / "bin" / "python"

    if venv_python.exists():
        # Проверяем версию Python в venv
        version = _check_python_version(str(venv_python))
        if version and version[0] == 3 and version[1] >= 14:
            return str(venv_python)
        # Если версия не 3.14+, нужно пересоздать venv
        # Но возвращаем текущий для совместимости, пересоздание будет в _ensure_venv_with_dependencies

    # Fallbacks - ищем Python 3.14+
    if os.name == "nt":
        launcher = shutil.which("py")
        if launcher:
            # Проверяем что py -3.14 действительно 3.14+
            try:
                result = subprocess.run(
                    [launcher, "-3.14", "-c", "import sys; exit(0 if sys.version_info[:2] >= (3, 14) else 1)"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return f"{launcher} -3.14"
            except:
                pass
        # Проверяем системный python
        try:
            result = subprocess.run(
                ["python", "-c", "import sys; exit(0 if sys.version_info[:2] >= (3, 14) else 1)"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return "python"
        except:
            pass
        return "python"  # Fallback, но будет ошибка если не 3.14+

    # Linux/macOS
    for cmd in ["python3.14", "python3", "python"]:
        python_path = shutil.which(cmd)
        if python_path:
            try:
                result = subprocess.run(
                    [python_path, "-c", "import sys; exit(0 if sys.version_info[:2] >= (3, 14) else 1)"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return python_path
            except:
                continue
    return sys.executable  # Последний fallback


PYTHON_EXECUTABLE = _detect_python_executable()


ERROR_KEYWORDS = (
    "error",
    "exception",
    "traceback",
    "critical",
    "fatal",
    "fail",
    "failed",
    "stderr",
    "не удалось",
    "невозможно",
    "ошибк",
    "критич",
    "аварийн",
    "stacktrace",
    "panic",
    "cannot",
    "can't",
    "refused",
    "denied",
    "permission denied",
    "timeout",
    "timed out",
    "errno",
    "traceback (most recent call last",
)

SUPPRESSED_SEVERITIES = (
    "warning",
    "warn",
    "предупрежд",
    "caution",
    "info",
    "инфо",
    "inform",
    "success",
    "успех",
    "debug",
    "trace",
    "verbose",
    "notice",
    "ℹ️",
    "⚠️",
    "✅",
)

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class ManagedProcess:
    """Wraps a subprocess and streams its output to a Tkinter-safe queue."""

    def __init__(self, name: str, command: List[str], channel: str):
        self.name = name
        self.command = command
        self.channel = channel
        self.process: Optional[subprocess.Popen[str]] = None
        self._reader_thread: Optional[threading.Thread] = None
        self.child_pids: Set[int] = set()

    def start(self, log_queue: "queue.Queue[Tuple[str, str]]") -> None:
        if self.is_running:
            raise RuntimeError(f"{self.name} already running")

        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{pythonpath}" if pythonpath else str(PROJECT_ROOT)

        # On Windows we can optionally create a new console window.
        popen_kwargs: Dict[str, object] = {
            "cwd": str(PROJECT_ROOT),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "env": env,
        }
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True

        self.process = subprocess.Popen(self.command, **popen_kwargs)  # type: ignore[arg-type]

        def _safe_put(item: Tuple[str, str]) -> None:
            while True:
                try:
                    log_queue.put_nowait(item)
                    break
                except queue.Full:
                    try:
                        log_queue.get_nowait()
                    except queue.Empty:
                        break

        def _reader() -> None:
            assert self.process and self.process.stdout
            self._snapshot_children()
            startup_message = f"{self.name} запущен и работает. Все ошибки будут показаны здесь."
            _safe_put((self.channel, startup_message))
            _safe_put(("system", startup_message))
            last_snapshot = time.monotonic()
            for line in self.process.stdout:
                now = time.monotonic()
                if now - last_snapshot >= 1.0:
                    self._snapshot_children()
                    last_snapshot = now
                stripped = line.strip()
                if not stripped:
                    continue
                if self._is_error_line(stripped, service_channel=self.channel):
                    message = f"[{self.name}] {stripped}"
                    _safe_put((self.channel, message))
                    _safe_put(("system", message))
            self.process.stdout.close()
            self._snapshot_children()

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        if not self.process or self.process.poll() is not None:
            return

        self._snapshot_children()
        self._kill_children()

        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

        if self._reader_thread:
            self._reader_thread.join(timeout=1)

        self.process = None
        self._reader_thread = None
        self.child_pids.clear()

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid if self.process else None

    def _kill_process_tree_win(self) -> None:
        if not self.process:
            return
        try:
            subprocess.run(
                ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            pass

    def _kill_process_tree_posix(self) -> None:
        if not self.process:
            return
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except Exception:
            pass

    def _kill_children(self) -> None:
        self._snapshot_children()
        if os.name == "nt":
            self._kill_process_tree_win()
        else:
            self._kill_process_tree_posix()
        if not self.child_pids:
            return
        for pid in list(self.child_pids):
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                else:
                    os.kill(pid, signal.SIGTERM)
            except Exception:
                continue
        self.child_pids.clear()

    def _snapshot_children(self) -> None:
        if not self.process:
            return
        try:
            import psutil  # type: ignore import
        except Exception:
            return

        try:
            parent = psutil.Process(self.process.pid)
            descendants = parent.children(recursive=True)
            current = {proc.pid for proc in descendants if proc.is_running()}
            self.child_pids = current
        except Exception:
            pass

    def _is_error_line(self, text: str, service_channel: Optional[str] = None) -> bool:
        cleaned = ANSI_ESCAPE_RE.sub("", text)
        lowered = cleaned.lower()

        if any(marker in lowered for marker in SUPPRESSED_SEVERITIES):
            return False
        if "ошибок: 0" in lowered or "ошибок 0" in lowered:
            return False

        if service_channel == "system":
            return any(keyword in lowered for keyword in ERROR_KEYWORDS)

        if " warning " in lowered or lowered.startswith("warning"):
            return False

        return any(keyword in lowered for keyword in ERROR_KEYWORDS)


class InfoBotManager(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("InfoBot Manager")
        self._default_geometry = self._load_saved_geometry()
        self.geometry(self._default_geometry)
        self.minsize(820, 600)

        self.log_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue(maxsize=3000)
        self.processes: Dict[str, ManagedProcess] = {}
        self.log_text_widgets: Dict[str, tk.Text] = {}
        self.log_tab_ids: Dict[str, str] = {}
        self.log_notebook: Optional[ttk.Notebook] = None
        self._temp_requirements_path: Optional[Path] = None
        self.status_var = tk.StringVar(value="Готово")
        self._active_tasks: Set[str] = set()
        self.max_log_lines = 2000
        self.active_log_channel = "system"
        self.pending_logs: Dict[str, List[str]] = defaultdict(list)
        self._bootstrap_ready = False
        self._service_control_buttons: List[ttk.Button] = []
        self._git_upstream_warned = False
        self._git_log_warned = False
        atexit.register(self._cleanup_processes)

        self._ensure_utf8_console()

        self.env_status_var = tk.StringVar()
        self.venv_enabled_var = tk.BooleanVar()
        self.git_status_var = tk.StringVar()
        self.license_status_var = tk.StringVar()
        # Проверяем текущее состояние venv (если .venv существует, значит включен)
        self.venv_enabled_var.set(VENV_DIR.exists())
        self.update_environment_status()
        self.ensure_git_repository()
        self.update_git_status(initial=True)
        self.update_license_status()
        self._ensure_required_app_files()

        self.service_status_vars: Dict[str, tk.StringVar] = {}

        self._build_ui()
        self._set_service_controls_enabled(False)
        self.after(0, self._start_bootstrap)
        self.after(0, self._apply_saved_geometry)
        self.after(200, self._flush_logs)
        self.after(1200, self._refresh_service_statuses)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI builder
    def _build_ui(self) -> None:
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.configure(yscrollcommand=scrollbar.set)

        scrollable = ttk.Frame(canvas)
        def _on_scrollable_configure(event: tk.Event) -> None:
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=bbox)
                content_height = bbox[3] - bbox[1]
                if content_height <= canvas.winfo_height():
                    canvas.yview_moveto(0)

        scrollable.bind("<Configure>", _on_scrollable_configure)

        window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")

        def _resize_canvas(event: tk.Event) -> None:  # type: ignore[override]
            canvas.itemconfigure(window_id, width=event.width)

        canvas.bind("<Configure>", _resize_canvas)

        main = ttk.Frame(scrollable, padding=(12, 0, 12, 12))
        main.grid(row=0, column=0, sticky="nsew")

        scrollable.columnconfigure(0, weight=1)
        scrollable.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(8, weight=1)

        style = ttk.Style(self)
        services_bg = "#dff5d8"
        style.configure("Services.TLabelframe", background=services_bg)
        style.configure("Services.TLabelframe.Label", background=services_bg)
        style.configure("Services.TFrame", background=services_bg)
        style.configure("Services.TLabel", background=services_bg)

        self._enable_mousewheel(canvas)

        status_frame = ttk.Frame(main, padding=(0, 0, 0, 6))
        status_frame.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        status_frame.columnconfigure(1, weight=1)
        ttk.Label(status_frame, text="Статус операций:").grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=1, sticky="w")
        self.loader = ttk.Progressbar(status_frame, mode="indeterminate", length=150)
        self.loader.grid(row=0, column=2, sticky="e")
        self.loader.stop()
        self.loader.grid_remove()

        separator = ttk.Separator(main, orient="horizontal")
        separator.grid(row=1, column=0, sticky="ew", padx=4)

        venv_frame = ttk.LabelFrame(main, text="1. Виртуальное окружение", padding=10)
        venv_frame.grid(row=2, column=0, sticky="new", padx=4, pady=(4, 4))
        venv_frame.columnconfigure(1, weight=1)

        ttk.Label(venv_frame, text="Статус:").grid(row=0, column=0, sticky="w")
        ttk.Label(venv_frame, textvariable=self.env_status_var).grid(row=0, column=1, sticky="w")
        
        venv_control_frame = ttk.Frame(venv_frame)
        venv_control_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        
        # Переключатель включения/отключения venv
        venv_toggle = ttk.Checkbutton(
            venv_control_frame,
            text="Использовать виртуальное окружение",
            variable=self.venv_enabled_var,
            command=self.toggle_venv
        )
        venv_toggle.pack(side=tk.LEFT)
        
        venv_buttons = ttk.Frame(venv_frame)
        venv_buttons.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(venv_buttons, text="Обновить venv", command=self.update_venv).pack(side=tk.LEFT)
        git_frame = ttk.LabelFrame(main, text="2. Обновления из Git", padding=10)
        git_frame.grid(row=3, column=0, sticky="new", padx=4, pady=4)
        git_frame.columnconfigure(1, weight=1)

        ttk.Label(git_frame, text="Статус репозитория:").grid(row=0, column=0, sticky="w")
        ttk.Label(git_frame, textvariable=self.git_status_var).grid(row=0, column=1, sticky="w")
        
        git_buttons = ttk.Frame(git_frame)
        git_buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(git_buttons, text="Проверить Git", command=self.check_git_status).pack(side=tk.LEFT)
        ttk.Button(git_buttons, text="Получить обновления", command=self.sync_with_remote).pack(side=tk.LEFT, padx=(8, 0))

        license_frame = ttk.LabelFrame(main, text="3. Лицензия и ключи (опционально)", padding=10)
        license_frame.grid(row=5, column=0, sticky="new", padx=4, pady=4)
        license_frame.columnconfigure(1, weight=1)

        ttk.Label(license_frame, text="Статус лицензии:").grid(row=0, column=0, sticky="w")
        ttk.Label(license_frame, textvariable=self.license_status_var).grid(row=0, column=1, sticky="w")

        btn_hwid = ttk.Button(license_frame, text="Получить Hardware ID")
        btn_hwid.grid(row=1, column=0, sticky="w", pady=(6, 0))
        btn_hwid.configure(command=lambda b=btn_hwid: self.run_license_activation(b))
        ttk.Button(license_frame, text="Импортировать .lic файл", command=self.import_license_file).grid(
            row=1, column=1, sticky="w", pady=(6, 0)
        )
        ttk.Button(
            license_frame,
            text="Как настроить ключи?",
            command=lambda: self.open_path(PROJECT_ROOT / "docs" / "INSTALL.md"),
        ).grid(row=1, column=2, sticky="w", pady=(6, 0))

        services_frame = ttk.LabelFrame(main, text="4. Запуск сервисов", padding=10, style="Services.TLabelframe")
        services_frame.grid(row=6, column=0, sticky="new", padx=4, pady=4)
        services_frame.columnconfigure(1, weight=1)

        header_frame = ttk.Frame(services_frame, style="Services.TFrame")
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        header_frame.columnconfigure(0, weight=1)

        ttk.Label(header_frame, text="Управление сервисами:", style="Services.TLabel").grid(row=0, column=0, sticky="w")
        btn_start_all = ttk.Button(header_frame, text="Запустить все", command=self.start_all_services)
        btn_start_all.grid(row=0, column=1, padx=(8, 4), sticky="e")
        self._service_control_buttons.append(btn_start_all)
        btn_stop_all = ttk.Button(header_frame, text="Остановить все", command=self.stop_all_services)
        btn_stop_all.grid(row=0, column=2, sticky="e")
        self._service_control_buttons.append(btn_stop_all)

        config_frame = ttk.Frame(services_frame, style="Services.TFrame")
        config_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Button(config_frame, text="Редактировать конфиг (configs/app_config.py)", command=self.open_config_file).pack(side=tk.LEFT)
        ttk.Button(config_frame, text="Редактировать ключи (configs/keys.py)", command=self.open_keys_file).pack(side=tk.LEFT, padx=(8, 0))

        for idx, (service_id, meta) in enumerate(self._services().items(), start=2):
            status_var = tk.StringVar(value="Не запущен")
            self.service_status_vars[service_id] = status_var
            ttk.Label(services_frame, text=meta["title"], style="Services.TLabel").grid(row=idx, column=0, sticky="w")
            ttk.Label(services_frame, textvariable=status_var, style="Services.TLabel").grid(row=idx, column=1, sticky="w")
            button_frame = ttk.Frame(services_frame, style="Services.TFrame")
            button_frame.grid(row=idx, column=2, sticky="w")
            start_btn = ttk.Button(button_frame, text="Запустить", command=lambda sid=service_id: self.start_service(sid))
            start_btn.pack(side=tk.LEFT, padx=(0, 4))
            stop_btn = ttk.Button(button_frame, text="Остановить", command=lambda sid=service_id: self.stop_service(sid))
            stop_btn.pack(side=tk.LEFT)
            self._service_control_buttons.extend([start_btn, stop_btn])

        docs_frame = ttk.LabelFrame(main, text="5. Документация и файлы", padding=10)
        docs_frame.grid(row=7, column=0, sticky="new", padx=4, pady=4)
        docs_frame.columnconfigure(0, weight=1)

        docs_buttons = ttk.Frame(docs_frame)
        docs_buttons.pack(anchor="w")
        ttk.Button(
            docs_buttons,
            text="Открыть README",
            command=lambda: self.open_path(PROJECT_ROOT / "README.md"),
        ).pack(side=tk.LEFT)
        ttk.Button(
            docs_buttons,
            text="Открыть лог ботов",
            command=self.open_bots_log,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            docs_buttons,
            text="Открыть каталог проекта",
            command=lambda: self.open_path(PROJECT_ROOT),
        ).pack(side=tk.LEFT, padx=(8, 0))

        log_frame = ttk.LabelFrame(main, text="6. Логи и вывод команд", padding=10)
        log_frame.grid(row=8, column=0, sticky="nsew", padx=4, pady=4)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(log_frame)
        notebook.grid(row=0, column=0, sticky="nsew")
        self.log_notebook = notebook
        self.log_tab_ids = {}
        notebook.bind("<<NotebookTabChanged>>", self._on_log_tab_changed)

        log_tabs = [
            ("system", "Системные события"),
            ("app", "Web UI (app.py)"),
            ("bots", "Bots Service (bots.py)"),
            ("ai", "AI Engine (ai.py)"),
        ]

        for channel, title in log_tabs:
            tab = ttk.Frame(notebook)
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)
            notebook.add(tab, text=title)
            tab_id = notebook.tabs()[-1]
            self.log_tab_ids[tab_id] = channel

            text_widget = tk.Text(tab, wrap="word", height=12)
            text_widget.grid(row=0, column=0, sticky="nsew")
            scrollbar = ttk.Scrollbar(tab, command=text_widget.yview)
            scrollbar.grid(row=0, column=1, sticky="ns")
            text_widget["yscrollcommand"] = scrollbar.set
            text_widget.bind("<Key>", self._log_text_key_handler)
            text_widget.bind("<<Paste>>", lambda event: "break")
            text_widget.bind("<<Cut>>", lambda event: "break")
            text_widget.bind("<Button-3>", lambda event, widget=text_widget: self._show_log_context_menu(event, widget))
            self.log_text_widgets[channel] = text_widget

        log_controls = ttk.Frame(log_frame)
        log_controls.grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Button(
            log_controls,
            text="Скопировать текущий лог",
            command=self.copy_current_log,
        ).pack(side=tk.LEFT)
        ttk.Button(
            log_controls,
            text="Очистить текущий лог",
            command=self.clear_current_log,
        ).pack(side=tk.LEFT, padx=(8, 0))

        contacts_frame = ttk.Frame(main, padding=(10, 0, 10, 12))
        contacts_frame.grid(row=9, column=0, sticky="ew", padx=4, pady=(0, 0))

        link_style = {"fg": "#0a66c2", "cursor": "hand2"}

        repo_label = tk.Label(
            contacts_frame,
            text="Проект: github.com/HellEvro/TPM_Public",
            **link_style,
        )
        repo_label.grid(row=0, column=0, sticky="w")
        repo_label.config(font=(repo_label.cget("font"), 10))
        repo_label.bind("<Button-1>", lambda _event: self.open_link("https://github.com/HellEvro/TPM_Public"))

        telegram_label = tk.Label(
            contacts_frame,
            text="Telegram: h3113vr0",
            **link_style,
        )
        telegram_label.grid(row=0, column=1, sticky="w", padx=(18, 0))
        telegram_label.config(font=(telegram_label.cget("font"), 10))
        telegram_label.bind("<Button-1>", lambda _event: self.open_link("https://t.me/H3113vr0"))

        email_label = tk.Label(
            contacts_frame,
            text="Email: gci.company.ou@gmail.com",
            **link_style,
        )
        email_label.grid(row=0, column=2, sticky="w", padx=(18, 0))
        email_label.config(font=(email_label.cget("font"), 10))
        email_label.bind("<Button-1>", lambda _event: self.open_link("mailto:gci.company.ou@gmail.com"))

        # Обновляем статус окружения после создания элементов UI,
        # чтобы кнопка глобальной установки корректно отразила состояние.
        self.update_environment_status()

    def _load_saved_geometry(self) -> str:
        if STATE_FILE.exists():
            try:
                with STATE_FILE.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                geometry = data.get("geometry")
                if isinstance(geometry, str) and geometry:
                    return geometry
            except Exception:
                pass
        return DEFAULT_GEOMETRY

    def _apply_saved_geometry(self) -> None:
        try:
            self.update_idletasks()
            self.geometry(self._default_geometry)
        except Exception:
            pass

    def _save_window_geometry(self) -> None:
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with STATE_FILE.open("w", encoding="utf-8") as fh:
                json.dump({"geometry": self.geometry()}, fh)
        except Exception:
            pass

    def _enable_mousewheel(self, widget: tk.Widget) -> None:
        if sys.platform == "darwin":
            widget.bind_all("<MouseWheel>", lambda event: widget.yview_scroll(int(-event.delta), "units"))
            widget.bind_all("<Button-4>", lambda event: widget.yview_scroll(-1, "units"))
            widget.bind_all("<Button-5>", lambda event: widget.yview_scroll(1, "units"))
        else:
            widget.bind_all("<MouseWheel>", lambda event: widget.yview_scroll(int(-event.delta / 120), "units"))
            widget.bind_all("<Button-4>", lambda event: widget.yview_scroll(-1, "units"))
            widget.bind_all("<Button-5>", lambda event: widget.yview_scroll(1, "units"))

    def _log_text_key_handler(self, event: tk.Event) -> Optional[str]:
        nav_keys = {"Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next"}
        if event.keysym in nav_keys:
            return None

        ctrl_pressed = (event.state & 0x4) != 0
        if ctrl_pressed:
            lowered = event.keysym.lower()
            if lowered == "c":
                return None
            if lowered == "a":
                event.widget.tag_add("sel", "1.0", tk.END)
                return "break"

        return "break"

    def _show_log_context_menu(self, event: tk.Event, widget: tk.Text) -> None:
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label="Копировать", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="Выделить всё", command=lambda: widget.tag_add("sel", "1.0", tk.END))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    # ------------------------------------------------------------------ Helpers
    def _services(self) -> Dict[str, Dict[str, str]]:
        python = PYTHON_EXECUTABLE.split() if " " in PYTHON_EXECUTABLE else [PYTHON_EXECUTABLE]
        return {
            "app": {
                "title": "Web UI (app.py, порт 5000)",
                "command": python + ["app.py"],
            },
            "bots": {
                "title": "Bots Service (bots.py, порт 5001)",
                "command": python + ["bots.py"],
            },
            "ai": {
                "title": "AI Engine (ai.py) (не обязателен)",
                "command": python + ["ai.py"],
            },
        }

    def update_environment_status(self) -> None:
        """Обновляет статус виртуального окружения"""
        global PYTHON_EXECUTABLE
        PYTHON_EXECUTABLE = _detect_python_executable()
        
        if VENV_DIR.exists():
            python = "python.exe" if os.name == "nt" else "python"
            self.env_status_var.set(f"Готово: .venv найден (используется {python})")
            self.venv_enabled_var.set(True)
        else:
            # Проверяем, есть ли ._venv (отключенное окружение)
            disabled_venv = PROJECT_ROOT / "._venv"
            if disabled_venv.exists():
                self.env_status_var.set("Виртуальное окружение отключено (._venv)")
            else:
                self.env_status_var.set("Виртуальное окружение не создано")
            self.venv_enabled_var.set(False)
    
    def toggle_venv(self) -> None:
        """Переключает виртуальное окружение (включает/отключает через переименование)"""
        global PYTHON_EXECUTABLE
        
        enabled = self.venv_enabled_var.get()
        venv_dir = PROJECT_ROOT / ".venv"
        disabled_venv_dir = PROJECT_ROOT / "._venv"
        
        # Останавливаем все сервисы перед переключением
        running_services = [sid for sid, proc in self.processes.items() if proc.is_running]
        if running_services:
            if not messagebox.askyesno(
                "Остановить сервисы?",
                f"Для переключения виртуального окружения необходимо остановить все запущенные сервисы.\n\n"
                f"Запущено сервисов: {len(running_services)}\n\n"
                f"Остановить все сервисы и продолжить?"
            ):
                # Отменяем переключение
                self.venv_enabled_var.set(not enabled)
                return
            self.stop_all_services()
            self.log("[venv] Все сервисы остановлены для переключения окружения", channel="system")
        
        try:
            if enabled:
                # Включаем venv: переименовываем ._venv -> .venv
                if disabled_venv_dir.exists():
                    if venv_dir.exists():
                        messagebox.showerror(
                            "Ошибка",
                            "Не удалось включить venv: .venv уже существует. Удалите один из каталогов вручную."
                        )
                        self.venv_enabled_var.set(False)
                        return
                    
                    try:
                        disabled_venv_dir.rename(venv_dir)
                        self.log("[venv] ✅ Виртуальное окружение включено (.venv)", channel="system")
                    except OSError as exc:
                        messagebox.showerror("Ошибка", f"Не удалось переименовать ._venv в .venv: {exc}")
                        self.venv_enabled_var.set(False)
                        return
                elif not venv_dir.exists():
                    # Если нет ни .venv, ни ._venv, предлагаем создать
                    if messagebox.askyesno(
                        "Создать venv?",
                        "Виртуальное окружение не найдено. Создать новое?"
                    ):
                        self._ensure_venv_with_dependencies(update_existing=False)
                    else:
                        self.venv_enabled_var.set(False)
                        self.update_environment_status()
                        return
            else:
                # Отключаем venv: переименовываем .venv -> ._venv
                if venv_dir.exists():
                    if disabled_venv_dir.exists():
                        messagebox.showerror(
                            "Ошибка",
                            "Не удалось отключить venv: ._venv уже существует. Удалите один из каталогов вручную."
                        )
                        self.venv_enabled_var.set(True)
                        return
                    
                    try:
                        venv_dir.rename(disabled_venv_dir)
                        self.log("[venv] ⏸️ Виртуальное окружение отключено (._venv)", channel="system")
                    except OSError as exc:
                        messagebox.showerror("Ошибка", f"Не удалось переименовать .venv в ._venv: {exc}")
                        self.venv_enabled_var.set(True)
                        return
                else:
                    # Если .venv не существует, просто обновляем статус
                    self.log("[venv] Виртуальное окружение уже отключено", channel="system")
            
            # Обновляем Python executable и статус
            PYTHON_EXECUTABLE = _detect_python_executable()
            self.update_environment_status()
            
            if enabled:
                self.log(f"[venv] Используется Python из venv: {PYTHON_EXECUTABLE}", channel="system")
            else:
                self.log(f"[venv] Используется системный Python: {PYTHON_EXECUTABLE}", channel="system")
                
        except Exception as exc:  # pylint: disable=broad-except
            error_msg = f"Ошибка при переключении venv: {exc}"
            self.log(f"[venv] ❌ {error_msg}", channel="system")
            messagebox.showerror("Ошибка", error_msg)
            # Восстанавливаем состояние
            self.venv_enabled_var.set(not enabled)
            self.update_environment_status()

    def _set_service_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in self._service_control_buttons:
            try:
                button.config(state=state)
            except tk.TclError:
                continue

    def _start_bootstrap(self) -> None:
        self.ensure_git_repository()
        self._run_task("auto_bootstrap", None, "Автонастройка проекта", self._bootstrap_worker)

    def _bootstrap_worker(self) -> None:
        success = True
        try:
            self._git_sync_worker()
            success = self._ensure_venv_with_dependencies(update_existing=False)
        except Exception as exc:  # pylint: disable=broad-except
            success = False
            self.log(f"Ошибка автоматической подготовки: {exc}", channel="system")
        finally:
            self.after(0, lambda: self._on_bootstrap_completed(success))

    def _on_bootstrap_completed(self, success: bool) -> None:
        self.update_environment_status()
        self.update_git_status()
        if success and VENV_DIR.exists():
            self._bootstrap_ready = True
            self._set_service_controls_enabled(True)
            self.log("Автоматическая подготовка завершена. Сервисы готовы к запуску.", channel="system")
        else:
            self._bootstrap_ready = False
            self._set_service_controls_enabled(False)
            self.log(
                "Автоматическая подготовка завершена с ошибками. Проверьте логи и повторите попытку.",
                channel="system",
            )

    def ensure_git_repository(self) -> None:
        git_dir = PROJECT_ROOT / ".git"
        if git_dir.exists():
            try:
                result = subprocess.run(
                    ["git", "symbolic-ref", "--short", "HEAD"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=True,
                )
                current_branch = result.stdout.strip()
                if current_branch == "master":
                    subprocess.run(
                        ["git", "branch", "-m", "main"],
                        cwd=str(PROJECT_ROOT),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        check=True,
                    )
                    self.log("Переименована ветка master → main", channel="system")
                    self.update_git_status()
                self._configure_git_upstream()
            except subprocess.CalledProcessError:
                pass
            return
        if not shutil.which("git"):
            self.git_status_var.set("git не найден (обновления недоступны)")
            self.log("Git не установлен: обновления репозитория отключены.")
            return
        try:
            init_result = subprocess.run(
                ["git", "init"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            subprocess.run(
                ["git", "branch", "-m", "main"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            if init_result.stdout.strip():
                self.log(init_result.stdout.strip())
            remote_result = subprocess.run(
                ["git", "remote", "add", "origin", DEFAULT_REMOTE_URL],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            if remote_result.stdout.strip():
                self.log(remote_result.stdout.strip())
            
            # Проверяем, есть ли коммиты, и создаем первый коммит, если его нет
            commit_check = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if commit_check.returncode != 0 or commit_check.stdout.strip() == "0":
                # Настраиваем Git пользователя для коммита (если не настроен)
                subprocess.run(
                    ["git", "config", "user.name", "InfoBot User"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                subprocess.run(
                    ["git", "config", "user.email", "infobot@local"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                # Добавляем все файлы и делаем первый коммит
                subprocess.run(
                    ["git", "add", "-A"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                commit_result = subprocess.run(
                    ["git", "commit", "-m", "Initial commit: InfoBot Public repository"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                if commit_result.returncode == 0:
                    self.log("Создан первый коммит в репозитории.", channel="system")
            
            self._configure_git_upstream()
            self.log(f"Git репозиторий инициализирован. origin → {DEFAULT_REMOTE_URL}")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else str(exc)
            self.log(f"[git] Не удалось инициализировать репозиторий: {stderr}")

    def update_git_status(self, initial: bool = False) -> None:
        if not shutil.which("git"):
            self.git_status_var.set("git не найден (обновления недоступны)")
            return
        try:
            # Проверяем, есть ли коммиты
            commit_check = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if commit_check.returncode != 0 or commit_check.stdout.strip() == "0":
                self.git_status_var.set("Репозиторий инициализирован (нет коммитов)")
                return
            
            result = subprocess.run(
                ["git", "status", "-sb"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            line = result.stdout.strip().splitlines()[0] if result.stdout else "Репозиторий"
            self.git_status_var.set(line)
        except subprocess.CalledProcessError as exc:
            if initial:
                self.git_status_var.set(f"Ошибка git status: {exc.returncode}")
            else:
                self.log(f"[git] Ошибка git status: {exc}")

    def update_license_status(self) -> None:
        lic_files = sorted(PROJECT_ROOT.glob("*.lic"))
        if lic_files:
            lic_name = lic_files[0].name
            self.license_status_var.set(f"Файл найден: {lic_name} (проверка...)")

            def worker() -> None:
                os.environ.pop("INFOBOT_LICENSE_STATUS", None)
                
                # Убеждаемся, что корень проекта в sys.path
                if str(PROJECT_ROOT) not in sys.path:
                    sys.path.insert(0, str(PROJECT_ROOT))
                
                try:
                    # Используем прямой импорт license_checker вместо модуля ai
                    # Это более надежно и не зависит от наличия модуля ai.py
                    from bot_engine.ai.license_checker import get_license_checker
                    
                    license_checker = get_license_checker(project_root=PROJECT_ROOT)
                    is_valid, info = license_checker.check_license()
                    
                    # Устанавливаем статус в переменную окружения для совместимости
                    if is_valid:
                        status_payload = {
                            'state': 'valid',
                            'license_type': info.get('type', 'premium'),
                            'expires_at': info.get('expires_at', 'N/A')
                        }
                    else:
                        status_payload = {
                            'state': 'invalid',
                            'reason': info if isinstance(info, str) else 'Лицензия не валидна'
                        }
                    
                    os.environ['INFOBOT_LICENSE_STATUS'] = json.dumps(status_payload, ensure_ascii=False)
                    
                except ImportError as import_exc:
                    error_msg = str(import_exc)
                    # Проверяем, что именно не импортировалось
                    if "license_checker" in error_msg or "bot_engine" in error_msg:
                        message = f"Модуль лицензирования не найден: {error_msg}. Проверьте наличие bot_engine/ai/license_checker.pyc"
                    else:
                        message = f"Ошибка импорта: {error_msg}"
                    self.after(0, lambda: self.license_status_var.set(message))
                    return
                except SystemExit:
                    pass
                except Exception as exc:  # pylint: disable=broad-except
                    error_msg = str(exc)
                    # Показываем более понятное сообщение об ошибке
                    if "No module named" in error_msg:
                        message = f"Ошибка импорта модулей: {error_msg}. Проверьте установку."
                    else:
                        message = f"Ошибка проверки лицензии: {error_msg}"
                    self.after(0, lambda: self.license_status_var.set(message))
                    return
                
                # Читаем результат из переменной окружения
                raw_status = os.environ.get("INFOBOT_LICENSE_STATUS")
                try:
                    status_payload = json.loads(raw_status) if raw_status else {}
                except json.JSONDecodeError:
                    status_payload = {}

                message = self._format_license_status(status_payload, lic_name)
                self.after(0, lambda: self.license_status_var.set(message))

            threading.Thread(target=worker, daemon=True).start()
        else:
            self.license_status_var.set("Лицензия не найдена (.lic файл в корне проекта)")

    def _format_license_status(self, status_payload: Dict[str, Any], filename: str) -> str:
        state = status_payload.get("state")
        expires_at_raw = status_payload.get("expires_at", "N/A")
        
        # Форматируем дату в читаемый формат
        expires_at = expires_at_raw
        if expires_at_raw != "N/A" and expires_at_raw:
            try:
                from datetime import datetime
                # Пробуем распарсить ISO формат
                if 'T' in expires_at_raw:
                    dt = datetime.fromisoformat(expires_at_raw.replace('Z', '+00:00'))
                    expires_at = dt.strftime("%d.%m.%Y %H:%M:%S")
                else:
                    # Если уже в другом формате, оставляем как есть
                    expires_at = expires_at_raw
            except (ValueError, AttributeError):
                # Если не удалось распарсить, оставляем как есть
                expires_at = expires_at_raw
        
        if state == "valid":
            source = status_payload.get("time_source", "unknown")
            source_text = "онлайн" if source == "internet" else "локально"
            return f"Лицензия активна (до {expires_at}, проверено {source_text})"
        if state == "expired":
            return f"Лицензия недействительна (истекла {expires_at})"
        if state == "invalid":
            reason = status_payload.get("reason") or "файл не найден или поврежден"
            return f"Лицензия недействительна ({reason})"
        if state == "time_unavailable":
            return "Проверка лицензии недоступна: нет доступа к серверам времени"
        return f"Ошибка проверки лицензии (файл {filename})"

    def _enqueue_log(self, channel: str, message: str, broadcast: bool = True) -> None:
        if broadcast and channel != "system":
            self._safe_put_log(("system", message))
        self._safe_put_log((channel, message))

    def log(self, message: str, channel: str = "system", broadcast: bool = False) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        self._enqueue_log(channel, formatted, broadcast=broadcast)

    def _flush_logs(self) -> None:
        max_lines_per_tick = 150
        processed = 0
        while processed < max_lines_per_tick:
            try:
                channel, line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            widget = self.log_text_widgets.get(channel) or self.log_text_widgets.get("system")
            if widget is None:
                continue
            self._append_log_line(widget, line, channel)
            processed += 1
        delay = 50 if not self.log_queue.empty() else 200
        self.after(delay, self._flush_logs)

    def _append_log_line(self, widget: tk.Text, line: str, channel: str) -> None:
        if channel != self.active_log_channel:
            buffer = self.pending_logs[channel]
            buffer.append(line)
            if len(buffer) > self.max_log_lines:
                del buffer[: len(buffer) - self.max_log_lines]
            return
        widget.insert(tk.END, line + "\n")
        if channel == self.active_log_channel:
            widget.see(tk.END)
        self._trim_text_widget(widget)

    def _trim_text_widget(self, widget: tk.Text) -> None:
        max_lines = getattr(self, "max_log_lines", 2000)
        try:
            end_index = widget.index("end-1c")
        except tk.TclError:
            return
        if not end_index:
            return
        try:
            total_lines = int(end_index.split(".")[0])
        except (ValueError, IndexError):
            return
        if total_lines <= max_lines:
            return
        lines_to_remove = total_lines - max_lines
        try:
            widget.delete("1.0", f"{lines_to_remove + 1}.0")
        except tk.TclError:
            pass

    def _safe_put_log(self, item: Tuple[str, str]) -> None:
        while True:
            try:
                self.log_queue.put_nowait(item)
                break
            except queue.Full:
                try:
                    self.log_queue.get_nowait()
                except queue.Empty:
                    break

    def _cleanup_processes(self) -> None:
        try:
            self.stop_all_services()
        except Exception:
            pass

    def _on_log_tab_changed(self, event: tk.Event) -> None:
        widget = event.widget
        if not isinstance(widget, ttk.Notebook):
            return
        selected = widget.select()
        channel = self.log_tab_ids.get(selected)
        if channel:
            self.active_log_channel = channel
            self._flush_pending_logs(channel)

    def _flush_pending_logs(self, channel: str) -> None:
        pending = self.pending_logs.get(channel)
        if not pending:
            return
        widget = self.log_text_widgets.get(channel)
        if not widget:
            return
        text = "".join(f"{line}\n" for line in pending)
        widget.insert(tk.END, text)
        widget.see(tk.END)
        self._trim_text_widget(widget)
        pending.clear()

    def _refresh_service_statuses(self) -> None:
        for service_id, status_var in self.service_status_vars.items():
            proc = self.processes.get(service_id)
            if proc and proc.is_running:
                status_var.set(f"Запущен (PID {proc.pid})")
            else:
                status_var.set("Не запущен")
                if proc and not proc.is_running:
                    self.processes.pop(service_id, None)
        self.after(1200, self._refresh_service_statuses)

    # ------------------------------------------------------------------ Command execution
    def _stream_command(self, title: str, command: List[str], channel: str = "system") -> None:
        self.log(f"[{title}] Запуск: {' '.join(command)}", channel=channel)
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:
            self.log(f"[{title}] Команда не найдена: {command[0]}", channel=channel)
            return

        assert proc.stdout
        for line in proc.stdout:
            self._enqueue_log(channel, f"[{title}] {line.rstrip()}")
        return_code = proc.wait()
        if return_code == 0:
            self.log(f"[{title}] Успешно завершено.", channel=channel)
        else:
            self.log(f"[{title}] Завершено с ошибкой (код {return_code}).", channel=channel)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

    def _ensure_venv_with_dependencies(self, update_existing: bool) -> bool:
        global PYTHON_EXECUTABLE
        try:
            need_install = update_existing
            need_recreate = False
            
            # Проверяем версию Python в существующем venv
            if VENV_DIR.exists():
                if os.name == "nt":
                    venv_python = VENV_DIR / "Scripts" / "python.exe"
                else:
                    venv_python = VENV_DIR / "bin" / "python"
                
                if venv_python.exists():
                    version = _check_python_version(str(venv_python))
                    if version:
                        major, minor = version
                        if major != 3 or minor < 14:
                            self.log(
                                f"Версия Python в .venv ({major}.{minor}) не соответствует требованиям (требуется 3.14+). Пересоздаем venv...",
                                channel="system",
                            )
                            need_recreate = True
                            need_install = True
            
            # Пересоздаем venv если нужно
            if need_recreate or not VENV_DIR.exists():
                if VENV_DIR.exists():
                    try:
                        self.log("Удаление старого .venv...", channel="system")
                        shutil.rmtree(VENV_DIR)
                    except Exception as e:
                        self.log(f"Ошибка при удалении старого venv: {e}", channel="system")
                        return False
                
                # Ищем Python 3.14+ (автоматически установит если не найден)
                python314 = _find_python314_plus()
                if not python314:
                    self.log(
                        "Python 3.14.2+ не найден и автоматическая установка не удалась.",
                        channel="system",
                    )
                    self.log(
                        "Установите Python 3.14.2 или выше вручную: https://www.python.org/downloads/",
                        channel="system",
                    )
                    return False
                else:
                    python_cmd = python314
                    version = _check_python_version(python_cmd.split()[0] if " " in python_cmd else python_cmd)
                    if version:
                        major, minor = version
                        self.log(f"Используем Python {major}.{minor}: {python_cmd}", channel="system")
                    else:
                        self.log(f"Используем Python 3.14.2+: {python_cmd}", channel="system")
                
                try:
                    cmd = python_cmd.split() if " " in python_cmd else [python_cmd]
                    self._stream_command(
                        "Создание окружения с Python 3.14.2+",
                        cmd + ["-m", "venv", str(VENV_DIR)],
                        channel="system",
                    )
                    need_install = True
                except subprocess.CalledProcessError as exc:
                    self.log(
                        f"Ошибка при создании виртуального окружения (.venv): {exc.returncode}",
                        channel="system",
                    )
                    return False
            
            if not need_install and not need_recreate:
                PYTHON_EXECUTABLE = _detect_python_executable()
                self.log(".venv уже существует с правильной версией Python. Проверяем зависимости...", channel="system")
                # Все равно обновляем зависимости для совместимости
                need_install = True

            python_exec = _detect_python_executable()
            if not python_exec:
                self.log("Не удалось определить Python для установки зависимостей.", channel="system")
                return False

            # Проверка зависимостей при запуске: pip install (быстро, если уже установлено) + verify
            self.log("Проверка зависимостей при запуске (установка при необходимости)...", channel="system")
            pip_cmd = _split_command(python_exec) + ["-m", "pip"]
            self._preinstall_ccxt_without_coincurve(pip_cmd)
            requirements_file = self._prepare_requirements_file()
            commands = [
                ("Обновление pip", pip_cmd + ["install", "--upgrade", "pip", "setuptools", "wheel", "--no-warn-script-location"]),
                ("Установка/обновление зависимостей", pip_cmd + ["install", "-r", requirements_file, "--upgrade", "--no-warn-script-location"]),
            ]
            for title, command in commands:
                try:
                    self._stream_command(title, command, channel="system")
                except subprocess.CalledProcessError as exc:
                    self.log(f"[{title}] Ошибка установки ({exc.returncode})", channel="system")
                    return False

            # Обязательно cryptography (ai_manager / license_checker) и проверка sklearn/моделей
            ch = "system"
            _stream = lambda t, c, ch=ch: self._stream_command(t, c, channel=ch)
            try:
                _ensure_cryptography_for_venv(python_exec, pip_cmd, self.log, _stream, ch)
            except Exception as e:  # pylint: disable=broad-except
                self.log(f"[venv] Ошибка проверки cryptography: {e}", channel=ch)
            try:
                _verify_ai_deps_for_venv(python_exec, pip_cmd, self.log, _stream, self._enqueue_log, ch)
            except Exception as e:  # pylint: disable=broad-except
                self.log(f"[venv] Ошибка verify_ai_deps: {e}", channel=ch)
            PYTHON_EXECUTABLE = _detect_python_executable()
            return True
        finally:
            self._cleanup_temp_requirements()

    def install_dependencies(self, button: Optional[ttk.Button] = None) -> None:
        def worker() -> None:
            self._ensure_venv_with_dependencies(update_existing=True)
            self.after(0, self.update_environment_status)

        self._run_task("install_venv", button, "Создание/обновление окружения", worker)
    
    def update_venv(self, button: Optional[ttk.Button] = None) -> None:
        """Обновляет виртуальное окружение, Python пакеты и зависимости"""
        def worker() -> None:
            self.log("[venv] Начинаю обновление виртуального окружения...", channel="system")
            try:
                global PYTHON_EXECUTABLE
                
                # Проверяем наличие venv
                if not VENV_DIR.exists():
                    self.log("[venv] ⚠️ Виртуальное окружение не найдено, создаем...", channel="system")
                    success = self._ensure_venv_with_dependencies(update_existing=True)
                    if success:
                        self.log("[venv] ✅ Виртуальное окружение создано и зависимости установлены", channel="system")
                    else:
                        self.log("[venv] ❌ Ошибка при создании виртуального окружения", channel="system")
                    self.after(0, self.update_environment_status)
                    return
                
                python_exec = _detect_python_executable()
                if not python_exec:
                    self.log("[venv] ❌ Не удалось определить Python для обновления", channel="system")
                    self.after(0, self.update_environment_status)
                    return
                
                pip_cmd = _split_command(python_exec) + ["-m", "pip"]
                
                # 1. Обновляем pip, setuptools, wheel до последних версий
                self.log("[venv] 🔄 Обновление pip, setuptools, wheel...", channel="system")
                try:
                    self._stream_command(
                        "Обновление pip и базовых пакетов",
                        pip_cmd + ["install", "--upgrade", "--upgrade-strategy", "eager", "pip", "setuptools", "wheel"],
                        channel="system",
                    )
                    self.log("[venv] ✅ pip, setuptools, wheel обновлены", channel="system")
                except subprocess.CalledProcessError as exc:
                    self.log(f"[venv] ⚠️ Ошибка обновления pip ({exc.returncode}), продолжаем...", channel="system")
                
                # 2. Обновляем все установленные пакеты до последних версий
                self.log("[venv] 🔄 Обновление всех установленных пакетов...", channel="system")
                try:
                    self._stream_command(
                        "Обновление всех пакетов",
                        pip_cmd + ["list", "--outdated", "--format=freeze"],
                        channel="system",
                    )
                except subprocess.CalledProcessError:
                    pass  # Игнорируем ошибки при проверке устаревших пакетов
                
                # 3. Обновляем зависимости из requirements.txt
                self.log("[venv] 🔄 Обновление зависимостей из requirements.txt...", channel="system")
                self._preinstall_ccxt_without_coincurve(pip_cmd)
                requirements_file = self._prepare_requirements_file()
                
                try:
                    # Обновляем пакеты из requirements.txt до последних версий
                    self._stream_command(
                        "Обновление зависимостей",
                        pip_cmd + ["install", "--upgrade", "--upgrade-strategy", "eager", "-r", requirements_file],
                        channel="system",
                    )
                    self.log("[venv] ✅ Зависимости обновлены", channel="system")
                except subprocess.CalledProcessError as exc:
                    self.log(f"[venv] ⚠️ Ошибка обновления зависимостей ({exc.returncode})", channel="system")

                # 3.1 Проверка sklearn/моделей; при несовпадении — переустановка scikit-learn 1.7.x
                ch = "system"
                _stream = lambda t, c, ch=ch: self._stream_command(t, c, channel=ch)
                try:
                    _verify_ai_deps_for_venv(python_exec, pip_cmd, self.log, _stream, self._enqueue_log, ch)
                except Exception as e:  # pylint: disable=broad-except
                    self.log(f"[venv] Ошибка verify_ai_deps: {e}", channel=ch)
                
                # 4. Проверяем версию Python
                try:
                    version_result = subprocess.run(
                        _split_command(python_exec) + ["--version"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        check=True,
                    )
                    python_version = version_result.stdout.strip()
                    self.log(f"[venv] 📌 Версия Python: {python_version}", channel="system")
                except Exception:
                    pass
                
                PYTHON_EXECUTABLE = _detect_python_executable()
                self.log("[venv] ✅ Обновление виртуального окружения завершено", channel="system")
                
            except Exception as exc:  # pylint: disable=broad-except
                self.log(f"[venv] ❌ Ошибка при обновлении: {exc}", channel="system")
                import traceback
                self.log(f"[venv] Traceback: {traceback.format_exc()}", channel="system")
            finally:
                self._cleanup_temp_requirements()
                self.after(0, self.update_environment_status)
        
        self._run_task("update_venv", button, "Обновление venv", worker)

    def install_dependencies_global(self, button: Optional[ttk.Button] = None) -> None:
        def worker() -> None:
            try:
                pip_cmd = _split_command(sys.executable) + ["-m", "pip"]
                self._preinstall_ccxt_without_coincurve(pip_cmd)
                requirements_file = self._prepare_requirements_file()
                python_cmd = pip_cmd + ["install", "-r", requirements_file]
                try:
                    self._stream_command("Установка зависимостей (глобально)", python_cmd, channel="system")
                    self.log("Глобальная установка зависимостей завершена.", channel="system")
                except subprocess.CalledProcessError as exc:
                    self.log(
                        f"[Установка зависимостей (глобально)] Ошибка ({exc.returncode}). Убедитесь, что есть права и активный pip.",
                        channel="system",
                    )
            finally:
                self._cleanup_temp_requirements()

        self._run_task("install_global", button, "Установка зависимостей", worker)

    def delete_environment(self, button: Optional[ttk.Button] = None) -> None:
        if not VENV_DIR.exists():
            messagebox.showinfo("Информация", "Виртуальное окружение (.venv) отсутствует.")
            return
        if messagebox.askyesno(
            "Удалить .venv",
            "Удалить виртуальное окружение (.venv)? Все запущенные сервисы будут остановлены.",
        ):
            def worker() -> None:
                self.stop_all_services()
                try:
                    shutil.rmtree(VENV_DIR)
                    self.log("Виртуальное окружение (.venv) удалено.", channel="system")
                except OSError as exc:
                    self.log(
                        f"Не удалось удалить .venv: {exc}",
                        channel="system",
                    )
                    self.after(
                        0,
                        lambda e=exc: messagebox.showerror(
                            "Ошибка удаления",
                            f"{e}\n\nЕсли проблема сохраняется, закройте менеджер и удалите папку .venv вручную.",
                        ),
                    )
                finally:
                    self.update_environment_status()
                    global PYTHON_EXECUTABLE
                    PYTHON_EXECUTABLE = _detect_python_executable()

            self._run_task("delete_venv", button, "Удаление окружения", worker)

    def check_git_status(self, button: Optional[ttk.Button] = None) -> None:
        """Ручная проверка статуса Git репозитория (та же проверка, что и при запуске)"""
        if not shutil.which("git"):
            messagebox.showwarning("Git не найден", "Git не установлен. Установите Git для работы с репозиторием.")
            self.git_status_var.set("git не найден (обновления недоступны)")
            return
        
        def worker() -> None:
            self.log("[git] Начинаю полную проверку Git репозитория...", channel="system")
            try:
                # Выполняем ту же проверку, что и при запуске приложения
                # 1. Проверяем и инициализируем репозиторий (если нужно)
                self.ensure_git_repository()
                
                # 2. Обновляем статус репозитория
                self.update_git_status()
                
                # 3. Настраиваем upstream (если нужно)
                self._configure_git_upstream()
                
                # 4. Показываем последние коммиты (если есть)
                self._run_git_log_preview()
                
                self.log("[git] Полная проверка Git завершена", channel="system")
            except Exception as exc:  # pylint: disable=broad-except
                self.log(f"[git] Ошибка при проверке Git: {exc}", channel="system")
            finally:
                # Обновляем статус в UI
                self.after(0, self.update_git_status)
        
        self._run_task("git_check", button, "Проверка Git", worker)

    def sync_with_remote(self, button: Optional[ttk.Button] = None) -> None:
        if not shutil.which("git"):
            messagebox.showwarning("Git не найден", "Для обновления необходимо установить Git.")
            return
        self.ensure_git_repository()
        self._run_task("git_sync", button, "Получение обновлений", self._git_sync_worker)

    def run_license_activation(self, button: Optional[ttk.Button] = None) -> None:
        python_cmd = _split_command(PYTHON_EXECUTABLE)
        command = python_cmd + ["scripts/activate_premium.py"]
        self._run_task(
            "license_activation",
            button,
            "Получение Hardware ID",
            lambda: self._license_worker(command),
        )

    def import_license_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Выберите файл лицензии",
            filetypes=[("InfoBot License", "*.lic"), ("Все файлы", "*.*")],
        )
        if not file_path:
            return

        destination = PROJECT_ROOT / Path(file_path).name
        try:
            shutil.copy2(file_path, destination)
            self.log(f"[license] Файл {destination.name} скопирован в корень проекта.")
            self.update_license_status()
        except OSError as exc:
            messagebox.showerror("Ошибка копирования", str(exc))

    def start_service(self, service_id: str) -> None:
        if not self._bootstrap_ready:
            self.log("Дождитесь завершения автоматической подготовки перед запуском сервисов.", channel="system")
            return
        if service_id in self.processes and self.processes[service_id].is_running:
            messagebox.showinfo("Уже запущено", f"Сервис {service_id} уже запущен.")
            return

        services = self._services()
        service = services[service_id]
        process = ManagedProcess(service["title"], service["command"], service_id)
        try:
            process.start(self.log_queue)
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Ошибка запуска", str(exc))
            return
        self.processes[service_id] = process
        self.log(f"{service['title']} запущен (PID {process.pid})", channel=service_id)

    def stop_service(self, service_id: str) -> None:
        process = self.processes.get(service_id)
        if not process or not process.is_running:
            self.log(f"Сервис {service_id} не запущен.", channel=service_id)
            return
        process.stop()
        self.processes.pop(service_id, None)
        services = self._services()
        title = services.get(service_id, {}).get("title", service_id)
        self.log(f"{title} остановлен.", channel=service_id)
        self.log(f"{title} остановлен.", channel="system", broadcast=False)

    def stop_all_services(self) -> None:
        for service_id in list(self.processes.keys()):
            self.stop_service(service_id)

    def start_all_services(self) -> None:
        if not self._bootstrap_ready:
            self.log("Дождитесь завершения автоматической подготовки перед запуском сервисов.", channel="system")
            return
        for service_id in self._services().keys():
            self.start_service(service_id)

    def _on_close(self) -> None:
        self._save_window_geometry()
        self.stop_all_services()
        self.destroy()

    def open_path(self, path: Path) -> None:
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if not path.exists():
            messagebox.showwarning("Файл не найден", f"Файл или папка {path} не существует.")
            return
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Ошибка открытия", str(exc))

    def open_link(self, url: str) -> None:
        try:
            webbrowser.open(url, new=2)
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Ошибка открытия ссылки", str(exc))

    def open_config_file(self) -> None:
        target = PROJECT_ROOT / "configs" / "app_config.py"
        example = PROJECT_ROOT / "configs" / "app_config.example.py"
        if not target.exists():
            if example.exists():
                if messagebox.askyesno(
                    "Создать конфиг",
                    "Файл configs/app_config.py не найден. Создать из configs/app_config.example.py?",
                ):
                    if not self._create_config_file_from_example(silent=False):
                        return
            else:
                messagebox.showwarning(
                    "Файл не найден",
                    "Файл configs/app_config.example.py отсутствует. Скопируйте шаблон вручную.",
                )
                return
        self.open_path(target)

    def open_keys_file(self) -> None:
        target = PROJECT_ROOT / "configs" / "keys.py"
        example = PROJECT_ROOT / "configs" / "keys.example.py"
        if not target.exists():
            if example.exists():
                if messagebox.askyesno(
                    "Создать файл ключей",
                    "Файл configs/keys.py не найден. Создать из configs/keys.example.py?",
                ):
                    if not self._create_keys_file_from_example(silent=False):
                        return
            else:
                messagebox.showwarning(
                    "Файл не найден",
                    "Файл configs/keys.example.py отсутствует. Скопируйте шаблон вручную.",
                )
                return
        self.open_path(target)

    def open_bots_log(self) -> None:
        target = PROJECT_ROOT / "logs" / "bots.log"
        if not target.exists():
            if messagebox.askyesno(
                "Лог не найден",
                "Файл logs/bots.log ещё не создан. Создать пустой файл?",
            ):
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.touch()
                    self.log("Создан пустой файл logs/bots.log", channel="system")
                except OSError as exc:
                    messagebox.showerror("Ошибка создания файла", str(exc))
                    return
            else:
                messagebox.showinfo(
                    "Лог отсутствует",
                    "Файл logs/bots.log появится после первого запуска сервиса bots.py.",
                )
                return
        self.open_path(target)

    def copy_current_log(self) -> None:
        if not self.log_notebook:
            return
        current_tab = self.log_notebook.select()
        channel = self.log_tab_ids.get(current_tab, "system")
        widget = self.log_text_widgets.get(channel)
        if not widget:
            messagebox.showinfo("Лог не найден", "Не удалось определить активный лог.")
            return
        text = widget.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Пусто", "В текущем логе нет данных для копирования.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Готово", "Содержимое лога скопировано в буфер обмена.")

    def clear_current_log(self) -> None:
        if not self.log_notebook:
            return
        current_tab = self.log_notebook.select()
        channel = self.log_tab_ids.get(current_tab, "system")
        widget = self.log_text_widgets.get(channel)
        if not widget:
            return
        try:
            widget.delete("1.0", tk.END)
        except tk.TclError:
            pass
        self.pending_logs[channel].clear()

    def _git_sync_worker(self) -> None:
        if not shutil.which("git"):
            return
        try:
            # Проверяем, есть ли коммиты перед синхронизацией
            commit_check = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if commit_check.returncode != 0 or commit_check.stdout.strip() == "0":
                self.log("[git] Нет локальных коммитов. Синхронизация пропущена.", channel="system")
                self.after(0, self.update_git_status)
                return
            
            self._stream_command("git fetch", ["git", "fetch", "--all", "--prune"])
            self.log(
                "[git] Локальные изменения будут перезаписаны состоянием origin/main (игнорируются только файлы из .gitignore).",
                channel="system",
            )
            # Снимаем skip-worktree с configs/bot_config.py, иначе "Entry not uptodate" при reset
            _skip_worktree_path = "configs/bot_config.py"
            try:
                subprocess.run(
                    ["git", "update-index", "--no-skip-worktree", _skip_worktree_path],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                    timeout=5,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass
            try:
                subprocess.run(
                    ["git", "update-index", "--refresh"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                    timeout=10,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                pass
            try:
                self._stream_command("git reset (подготовка)", ["git", "reset", "--hard", "HEAD"])
            except subprocess.CalledProcessError:
                pass
            try:
                self._stream_command("git reset", ["git", "reset", "--hard", "origin/main"])
            except subprocess.CalledProcessError:
                self.log(
                    "[git reset] Не удалось сбросить на origin/main. Если было «Entry not uptodate», "
                    "в терминале в каталоге репозитория выполните:\n"
                    "  git update-index --no-skip-worktree configs/bot_config.py\n"
                    "  git update-index --refresh\n"
                    "  git reset --hard origin/main",
                    channel="system",
                )
                return
            try:
                self._stream_command("git clean", ["git", "clean", "-fd"])
            except subprocess.CalledProcessError:
                pass
        except subprocess.CalledProcessError:
            pass
        finally:
            self.after(0, self.update_git_status)
        self._configure_git_upstream()
        self._run_git_log_preview()

    def _license_worker(self, command: List[str]) -> None:
        try:
            self._stream_command("license", command)
        except subprocess.CalledProcessError:
            pass
        self.update_license_status()

    def _set_status(self, text: str, busy: bool) -> None:
        self.status_var.set(text)
        if busy:
            if not self.loader.winfo_ismapped():
                self.loader.grid()
            self.loader.start(10)
        else:
            self.loader.stop()
            if self.loader.winfo_ismapped():
                self.loader.grid_remove()

    def _run_task(
        self,
        task_id: str,
        button: Optional[ttk.Button],
        description: str,
        worker: Callable[[], None],
    ) -> None:
        if task_id in self._active_tasks:
            self.log(f"{description} уже выполняется...", channel="system")
            return

        original_text = button.cget("text") if button else None
        if button:
            button.config(state=tk.DISABLED, text=f"{description}…")

        self._active_tasks.add(task_id)
        self._set_status(f"{description}…", busy=True)

        def run() -> None:
            try:
                worker()
            finally:
                def finish() -> None:
                    if button and original_text is not None:
                        button.config(state=tk.NORMAL, text=original_text)
                    self._active_tasks.discard(task_id)
                    if self._active_tasks:
                        self._set_status("Выполняется…", busy=True)
                    else:
                        self._set_status("Готово", busy=False)

                self.after(0, finish)

        threading.Thread(target=run, daemon=True).start()

    def _preinstall_ccxt_without_coincurve(self, pip_cmd: List[str]) -> None:
        if os.name != "nt" or sys.version_info < (3, 13):
            return
        try:
            self._stream_command(
                "Подготовка ccxt (без optional зависимостей)",
                pip_cmd + ["install", "--upgrade", "ccxt", "--no-deps"],
                channel="system",
            )
        except subprocess.CalledProcessError:
            self.log(
                "[Подготовка ccxt] Не удалось установить ccxt без дополнительных зависимостей. Продолжаем стандартную установку.",
                channel="system",
            )

    def _configure_git_upstream(self) -> None:
        if not shutil.which("git"):
            return
        try:
            # Сначала проверяем, есть ли коммиты
            commit_check = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if commit_check.returncode != 0 or commit_check.stdout.strip() == "0":
                # Нет коммитов - upstream настроить нельзя
                return
            
            branch_result = subprocess.run(
                ["git", "symbolic-ref", "--short", "HEAD"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            current_branch = branch_result.stdout.strip()
            if current_branch != "main":
                return
            tracking_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "main@{upstream}"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if tracking_result.returncode != 0:
                subprocess.run(
                    ["git", "branch", "--set-upstream-to=origin/main", "main"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=True,
                )
                self._git_upstream_warned = False
                self.log("Ветка main теперь отслеживает origin/main.", channel="system")
        except subprocess.CalledProcessError as exc:
            if not self._git_upstream_warned:
                detail = ""
                if isinstance(exc, subprocess.CalledProcessError):
                    detail = (exc.stderr or "").strip() if getattr(exc, "stderr", None) else ""
                # Не показываем ошибку, если просто нет коммитов
                if "no commit" not in detail.lower() and "unknown revision" not in detail.lower():
                    message = "[git] Не удалось настроить upstream для ветки main."
                    if detail:
                        message = f"{message} {detail}"
                    self.log(message, channel="system")
                self._git_upstream_warned = True

    def _run_git_log_preview(self) -> None:
        if not shutil.which("git"):
            return
        try:
            commit_check = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            # Если команда не выполнилась или вернула 0 - нет коммитов
            if commit_check.returncode != 0 or commit_check.stdout.strip() == "0":
                self._git_log_warned = False
                return  # Не показываем ошибку, просто выходим
            
            # Показываем новый и предыдущий коммит(c) c датами
            result = subprocess.run(
                ["git", "log", "-2", "--pretty=format:%h | %cd | %s%n    Автор: %an", "--graph", "--date=format:%Y-%m-%d %H:%M:%S"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            output = result.stdout.strip()
            if output:
                self.log_queue.put(("system", f"[git log] Последние коммиты:\n{output}"))
            self._git_log_warned = False
        except subprocess.CalledProcessError as exc:
            if not self._git_log_warned:
                detail = (exc.stderr or "").strip() if getattr(exc, "stderr", None) else ""
                # Не показываем ошибку, если просто нет коммитов
                if "unknown revision" not in detail.lower() and "no commit" not in detail.lower():
                    message = "[git log] Не удалось получить историю коммитов."
                    if detail:
                        message = f"{message} {detail}"
                    self.log(message, channel="system")
                self._git_log_warned = True

    def _ensure_utf8_console(self) -> None:
        if os.name != "nt":
            return
        try:
            subprocess.run(
                "chcp 65001",
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except Exception:
            pass
        for cmd in [
            ["git", "config", "--global", "core.quotepath", "off"],
            ["git", "config", "--global", "i18n.logOutputEncoding", "utf-8"],
        ]:
            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
            except Exception:
                pass

    def _ensure_required_app_files(self) -> None:
        config_path = PROJECT_ROOT / "app" / "config.py"
        if not config_path.exists():
            self._create_config_file_from_example(silent=True)

        keys_path = PROJECT_ROOT / "app" / "keys.py"
        if not keys_path.exists():
            self._create_keys_file_from_example(silent=True)
        
        # ✅ Восстанавливаем bot_config.py из example, если он отсутствует
        bot_config_path = PROJECT_ROOT / "bot_engine" / "bot_config.py"
        if not bot_config_path.exists():
            self._create_bot_config_file_from_example(silent=True)

    def _create_config_file_from_example(self, silent: bool = False) -> bool:
        target = PROJECT_ROOT / "configs" / "app_config.py"
        example = PROJECT_ROOT / "configs" / "app_config.example.py"
        if not example.exists():
            message = "Файл configs/app_config.example.py не найден. Скопируйте шаблон вручную."
            if silent:
                self.log(message, channel="system")
            else:
                messagebox.showwarning("Файл не найден", message)
            return False

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(example, target)
            self._strip_example_header(target)
            self.log("Создан configs/app_config.py из примера", channel="system")
            return True
        except OSError as exc:
            message = f"Не удалось создать configs/app_config.py: {exc}"
            if silent:
                self.log(message, channel="system")
            else:
                messagebox.showerror("Ошибка копирования", str(exc))
            return False

    def _create_keys_file_from_example(self, silent: bool = False) -> bool:
        target = PROJECT_ROOT / "configs" / "keys.py"
        example = PROJECT_ROOT / "configs" / "keys.example.py"
        if not example.exists():
            message = "Файл configs/keys.example.py не найден. Скопируйте шаблон ключей вручную."
            if silent:
                self.log(message, channel="system")
            else:
                messagebox.showwarning("Файл не найден", message)
            return False

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(example, target)
            self.log("Создан configs/keys.py из примера", channel="system")
            self.log("Добавьте API ключи в configs/keys.py перед запуском сервисов.", channel="system")
            return True
        except OSError as exc:
            message = f"Не удалось создать configs/keys.py: {exc}"
            if silent:
                self.log(message, channel="system")
            else:
                messagebox.showerror("Ошибка копирования", str(exc))
            return False

    def _create_bot_config_file_from_example(self, silent: bool = False) -> bool:
        """Создает configs/bot_config.py из configs/bot_config.example.py, если отсутствует."""
        target = PROJECT_ROOT / "configs" / "bot_config.py"
        example = PROJECT_ROOT / "configs" / "bot_config.example.py"
        if not example.exists():
            message = "Файл configs/bot_config.example.py не найден."
            if silent:
                self.log(message, channel="system")
            else:
                messagebox.showwarning("Файл не найден", message)
            return False

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(example, target)
            self.log("Создан configs/bot_config.py из примера", channel="system")
            return True
        except OSError as exc:
            message = f"Не удалось создать configs/bot_config.py: {exc}"
            if silent:
                self.log(message, channel="system")
            else:
                messagebox.showerror("Ошибка копирования", str(exc))
            return False

    def _strip_example_header(self, config_path: Path) -> None:
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError:
            return

        stripped = text.lstrip()
        if stripped.startswith('"""'):
            start_index = text.find('"""')
            end_index = text.find('"""', start_index + 3)
            if end_index != -1:
                new_text = text[end_index + 3 :]
                # remove leading blank lines
                new_text = new_text.lstrip("\n")
                try:
                    config_path.write_text(new_text, encoding="utf-8")
                    self.log("Удалён пояснительный блок из configs/app_config.py.", channel="system")
                except OSError as exc:
                    self.log(f"[config] Не удалось очистить заголовок config.py: {exc}", channel="system")

    def _prepare_requirements_file(self) -> str:
        base_path = PROJECT_ROOT / "requirements.txt"
        if os.name != "nt" or sys.version_info < (3, 13):
            return str(base_path)

        try:
            lines = base_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            self.log(f"Не удалось прочитать requirements.txt: {exc}", channel="system")
            return str(base_path)

        filtered = []
        ccxt_skipped = False
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("ccxt"):
                ccxt_skipped = True
                continue
            filtered.append(line)

        if not ccxt_skipped:
            return str(base_path)

        fd, temp_path = tempfile.mkstemp(prefix="requirements_filtered_", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            temp_file.write("\n".join(filtered) + "\n")
        self._temp_requirements_path = Path(temp_path)
        self.log(
            "Используем requirements без ccxt (уже установлен отдельно, чтобы избежать установки coincurve).",
            channel="system",
        )
        return temp_path

    def _cleanup_temp_requirements(self) -> None:
        if self._temp_requirements_path and self._temp_requirements_path.exists():
            try:
                self._temp_requirements_path.unlink()
            except OSError:
                pass
        self._temp_requirements_path = None


def _split_command(command: str) -> List[str]:
    if os.name == "nt" and command.startswith("py "):
        return command.split()
    if " " in command:
        return command.split()
    return [command]


def _ensure_cryptography_for_venv(
    python_exec: str,
    pip_cmd: List[str],
    log_fn: Callable[..., None],
    stream_fn: Callable[..., None],
    channel: str = "system",
) -> bool:
    """Устанавливает cryptography при необходимости. Вызывается из лаунчера (модульный уровень — без self)."""
    try:
        r = subprocess.run(
            _split_command(python_exec) + ["-c", "import cryptography"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            return True
    except Exception:
        pass
    log_fn("[cryptography] Установка cryptography...", channel=channel)
    try:
        stream_fn("Установка cryptography", pip_cmd + ["install", "cryptography>=41.0.0", "--no-warn-script-location"], channel)
    except subprocess.CalledProcessError:
        log_fn("[cryptography] Ошибка установки cryptography.", channel=channel)
        return False
    return True


def _verify_ai_deps_for_venv(
    python_exec: str,
    pip_cmd: List[str],
    log_fn: Callable[..., None],
    stream_fn: Callable[..., None],
    enqueue_fn: Callable[..., None],
    channel: str = "system",
) -> bool:
    """Проверяет sklearn/joblib и модели; при ошибке ставит scikit-learn 1.7.x. Модульный уровень — без self."""
    script = PROJECT_ROOT / "scripts" / "verify_ai_deps.py"
    if not script.exists():
        return True

    def run_verify() -> bool:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        r = subprocess.run(
            _split_command(python_exec) + [str(script)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            env=env,
        )
        if r.returncode != 0 and r.stderr:
            for line in r.stderr.strip().splitlines():
                enqueue_fn(channel, line.strip() if line.strip() else line)
        return r.returncode == 0

    if run_verify():
        log_fn("[verify_ai_deps] OK", channel=channel)
        return True

    log_fn(
        "[verify_ai_deps] Ошибка версий/моделей — переустанавливаем scikit-learn>=1.7.0,<1.8...",
        channel=channel,
    )
    try:
        stream_fn(
            "Установка scikit-learn 1.7.x",
            pip_cmd + ["install", "scikit-learn>=1.7.0,<1.8", "--no-warn-script-location"],
            channel,
        )
    except subprocess.CalledProcessError:
        log_fn("[verify_ai_deps] Не удалось переустановить scikit-learn.", channel=channel)
        return False

    log_fn("[verify_ai_deps] Повторная проверка после переустановки...", channel=channel)
    if run_verify():
        log_fn("[verify_ai_deps] OK после переустановки", channel=channel)
        return True
    log_fn("[verify_ai_deps] Ошибка сохраняется. Нажмите «Обновить venv». Если не поможет — запустите AI Engine и дождитесь обучения; модели обновятся.", channel=channel)
    return False


def main() -> None:
    manager = InfoBotManager()
    try:
        manager.mainloop()
    finally:
        manager.stop_all_services()


if __name__ == "__main__":
    main()

