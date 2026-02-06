#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для отдельного запуска синхронизации времени Windows.
Использует те же настройки, что и app (configs/app_config.py → TIME_SYNC), либо значения по умолчанию.

Запуск из корня проекта:
  python scripts/run_time_sync.py           # бесконечный цикл каждые 30 мин (по умолчанию)
  python scripts/run_time_sync.py --once    # один раз синхронизировать и выйти
  python scripts/run_time_sync.py --no-require-admin  # не требовать админа (пробовать в любом случае)

Можно добавить в Планировщик заданий Windows для запуска по расписанию.
"""
from __future__ import annotations

import argparse
import datetime
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

# Корень проекта
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# UTF-8 для консоли Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Настройки по умолчанию (как в app.py)
DEFAULT_CONFIG = {
    "INTERVAL_MINUTES": 30,
    "SERVER": "time.windows.com",
    "REQUIRE_ADMIN": True,
    "RUN_ON_START": True,
}


def load_config() -> dict:
    """Загружает TIME_SYNC из configs/app_config.py или возвращает DEFAULT_CONFIG."""
    try:
        from configs.app_config import TIME_SYNC  # noqa: F401
        cfg = dict(DEFAULT_CONFIG)
        if isinstance(TIME_SYNC, dict):
            cfg.update(TIME_SYNC)
        return cfg
    except Exception:
        return dict(DEFAULT_CONFIG)


def check_admin_rights() -> bool:
    if sys.platform != "win32":
        return False
    try:
        r = subprocess.run(["net", "session"], capture_output=True, check=False)
        return r.returncode == 0
    except Exception:
        return False


# Альтернативные NTP-серверы, если time.windows.com недоступен
FALLBACK_NTP_SERVERS = ["time.windows.com", "time.google.com", "pool.ntp.org"]

# NTP epoch (1900-01-01) vs Unix epoch (1970-01-01)
NTP_EPOCH_OFFSET = (datetime.datetime(1970, 1, 1) - datetime.datetime(1900, 1, 1)).total_seconds()


def ntp_fetch_time(host: str, port: int = 123, timeout: float = 5.0) -> datetime.datetime | None:
    """
    Получает время с NTP-сервера по UDP (без внешних зависимостей).
    Возвращает datetime в локальной временной зоне или None при ошибке.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        # NTP client packet: 48 bytes (mode 3 = client)
        packet = bytearray(48)
        packet[0] = 0x1B  # li=0, vn=3, mode=3
        sock.sendto(packet, (host, port))
        data = sock.recv(48)
        sock.close()
        if len(data) < 48:
            return None
        # Transmit timestamp at offset 40 (32-bit seconds, 32-bit fraction)
        secs = struct.unpack("!I", data[40:44])[0]
        frac = struct.unpack("!I", data[44:48])[0]
        ntp_secs = secs + frac / 2**32
        unix_secs = ntp_secs - NTP_EPOCH_OFFSET
        utc = datetime.datetime.utcfromtimestamp(unix_secs)
        # Переводим в локальное время (для установки в систему)
        return utc.replace(tzinfo=datetime.timezone.utc).astimezone().replace(tzinfo=None)
    except Exception:
        return None


def set_windows_time_direct(dt: datetime.datetime) -> tuple[bool, str]:
    """
    Устанавливает системное время Windows через API SetLocalTime.
    Требует прав администратора (привилегия SeSystemtimePrivilege).
    """
    if sys.platform != "win32":
        return False, "Только для Windows"
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        advapi32 = ctypes.windll.advapi32  # type: ignore[attr-defined]

        class SYSTEMTIME(ctypes.Structure):
            _fields_ = [
                ("wYear", wintypes.WORD),
                ("wMonth", wintypes.WORD),
                ("wDayOfWeek", wintypes.WORD),
                ("wDay", wintypes.WORD),
                ("wHour", wintypes.WORD),
                ("wMinute", wintypes.WORD),
                ("wSecond", wintypes.WORD),
                ("wMilliseconds", wintypes.WORD),
            ]

        st = SYSTEMTIME()
        st.wYear = dt.year
        st.wMonth = dt.month
        st.wDayOfWeek = dt.weekday()
        st.wDay = dt.day
        st.wHour = dt.hour
        st.wMinute = dt.minute
        st.wSecond = dt.second
        st.wMilliseconds = dt.microsecond // 1000

        # Сначала пробуем без явного включения привилегии (у админа часто уже есть)
        if kernel32.SetLocalTime(ctypes.byref(st)):
            return True, "Время установлено (NTP + SetLocalTime)"

        # Включаем SeSystemtimePrivilege и повторяем
        TOKEN_ADJUST_PRIVILEGES = 0x0020
        TOKEN_QUERY = 0x0008
        SE_PRIVILEGE_ENABLED = 0x00000002

        class LUID(ctypes.Structure):
            _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", ctypes.c_long)]

        class LUID_AND_ATTRIBUTES(ctypes.Structure):
            _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]

        class TOKEN_PRIVILEGES(ctypes.Structure):
            _fields_ = [("PrivilegeCount", wintypes.DWORD), ("Privileges", LUID_AND_ATTRIBUTES)]

        token = ctypes.c_void_p()
        if not advapi32.OpenProcessToken(
            kernel32.GetCurrentProcess(),
            TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
            ctypes.byref(token),
        ):
            return False, "OpenProcessToken не удался"

        tp = TOKEN_PRIVILEGES()
        tp.PrivilegeCount = 1
        tp.Privileges.Attributes = SE_PRIVILEGE_ENABLED
        if not advapi32.LookupPrivilegeValueW(None, "SeSystemtimePrivilege", ctypes.byref(tp.Privileges.Luid)):
            kernel32.CloseHandle(token)
            return False, "LookupPrivilegeValue не удался"
        if not advapi32.AdjustTokenPrivileges(token, False, ctypes.byref(tp), 0, None, None):
            kernel32.CloseHandle(token)
            return False, "AdjustTokenPrivileges не удался"

        ok = kernel32.SetLocalTime(ctypes.byref(st))
        kernel32.CloseHandle(token)
        return (True, "Время установлено (NTP + SetLocalTime)") if ok else (False, "SetLocalTime не удался")
    except Exception as e:
        return False, str(e)


def sync_time_ntp_fallback(ntp_server: str, silent: bool = False) -> tuple[bool, str]:
    """
    Резервная синхронизация: получить время по NTP, установить через SetLocalTime.
    Не зависит от службы w32time. Требует прав администратора.
    """
    if not check_admin_rights():
        return False, "Требуются права администратора"
    dt = ntp_fetch_time(ntp_server)
    if dt is None:
        return False, f"Не удалось получить время с {ntp_server}"
    if not silent:
        print(f"[TimeSync] Время с NTP ({ntp_server}): {dt}")
    return set_windows_time_direct(dt)


def query_time_service_diagnostics() -> str:
    """Возвращает вывод w32tm /query и состояние службы для диагностики."""
    lines = []
    # Состояние службы Windows Time (часто даёт причину, если w32tm молчит)
    r_sc = subprocess.run(
        ["sc", "query", "w32time"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    lines.append("Служба w32time (sc query w32time):")
    if (r_sc.stdout or "").strip():
        lines.append(r_sc.stdout.strip())
    if (r_sc.stderr or "").strip():
        lines.append("stderr: " + r_sc.stderr.strip())
    if not (r_sc.stdout or r_sc.stderr or "").strip():
        lines.append("(пустой вывод)")
    lines.append("---")
    for q in ["/status", "/source"]:
        r = subprocess.run(
            ["w32tm", "/query", q],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        part = f"w32tm /query {q} (код {r.returncode}):"
        if (r.stdout or "").strip():
            part += "\n" + r.stdout.strip()
        if (r.stderr or "").strip():
            part += "\nstderr: " + r.stderr.strip()
        if not (r.stdout or r.stderr or "").strip():
            part += " (пустой вывод)"
        lines.append(part)
    return "\n".join(lines)


def ensure_time_service_running() -> tuple[bool, str]:
    """
    Запускает службу w32time. Если она отключена — переводит в «вручную» и запускает.
    Возвращает (успех, сообщение).
    """
    if sys.platform != "win32":
        return False, "Только для Windows"
    if not check_admin_rights():
        return False, "Нужны права администратора"
    try:
        r = subprocess.run(
            ["sc", "start", "w32time"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if r.returncode == 0:
            return True, "Служба w32time запущена"
        # Служба может быть отключена (disabled) — переводим в ручной запуск
        subprocess.run(
            ["sc", "config", "w32time", "start= demand"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        r2 = subprocess.run(
            ["sc", "start", "w32time"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if r2.returncode == 0:
            return True, "Служба w32time запущена (тип запуска установлен «вручную»)"
        err = (r2.stderr or r2.stdout or r.stderr or r.stdout or "").strip()
        return False, err or f"Не удалось запустить w32time (код {r2.returncode})"
    except Exception as e:
        return False, str(e)


def configure_time_service(server: str = "time.windows.com", silent: bool = True) -> tuple[bool, str]:
    if sys.platform != "win32":
        return False, "Только для Windows"
    if not check_admin_rights():
        return False, "Требуются права администратора"
    try:
        # Сначала убедимся, что служба запущена (иначе /config может не сработать)
        ensure_time_service_running()
        cmd = [
            "w32tm", "/config",
            f'/manualpeerlist:"{server}"',
            "/syncfromflags:manual",
            "/reliable:yes",
            "/update",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
        if r.returncode != 0:
            return False, (r.stderr or "").strip() or "Ошибка настройки"
        subprocess.run(["net", "stop", "w32time"], capture_output=True, check=False)
        time.sleep(2)
        r_start = subprocess.run(["net", "start", "w32time"], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False)
        if r_start.returncode != 0:
            ok, msg = ensure_time_service_running()
            if not ok:
                return False, f"Служба не перезапустилась: {msg}"
        return True, "Служба времени настроена"
    except Exception as e:
        return False, str(e)


def sync_time_once(
    require_admin: bool = True,
    server: str = "time.windows.com",
    silent: bool = False,
) -> tuple[bool, str]:
    if sys.platform != "win32":
        return False, "Синхронизация времени только для Windows"
    if require_admin and not check_admin_rights():
        return False, "Требуются права администратора. Запустите от имени администратора или используйте --no-require-admin."
    try:
        if not silent:
            print("[TimeSync] Синхронизация времени...")
        r = subprocess.run(
            ["w32tm", "/resync"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if r.returncode == 0:
            return True, "Время успешно синхронизировано"
        # Попытка с /rediscover (помогает после перезапуска службы)
        r_rediscover = subprocess.run(
            ["w32tm", "/resync", "/rediscover"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if r_rediscover.returncode == 0:
            return True, "Время синхронизировано (/rediscover)"
        # Настройка службы и повтор (только с админом)
        if check_admin_rights():
            ok, _ = configure_time_service(server=server, silent=True)
            if ok:
                time.sleep(5)  # даём службе времени запуститься
                r2 = subprocess.run(
                    ["w32tm", "/resync"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    check=False,
                )
                if r2.returncode == 0:
                    return True, "Время синхронизировано после настройки службы"
                r2 = subprocess.run(
                    ["w32tm", "/resync", "/rediscover"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    check=False,
                )
                if r2.returncode == 0:
                    return True, "Время синхронизировано (/rediscover после настройки)"
            # Пробуем альтернативные NTP-серверы
            for fallback in FALLBACK_NTP_SERVERS:
                if fallback == server:
                    continue
                ok_cfg, _ = configure_time_service(server=fallback, silent=True)
                if ok_cfg:
                    time.sleep(3)
                    r3 = subprocess.run(
                        ["w32tm", "/resync"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="ignore",
                        check=False,
                    )
                    if r3.returncode == 0:
                        return True, f"Время синхронизировано (сервер {fallback})"
            # Резерв: не полагаемся на w32time — получаем время по NTP и ставим через SetLocalTime
            if not silent:
                print("[TimeSync] Пробуем резервный способ (NTP + SetLocalTime)...")
            for ntp_host in [server] + [s for s in FALLBACK_NTP_SERVERS if s != server]:
                ok_ntp, msg_ntp = sync_time_ntp_fallback(ntp_host, silent=True)
                if ok_ntp:
                    return True, msg_ntp + f" (сервер {ntp_host})"
        err_msg = (r.stderr or "").strip() or f"Код выхода w32tm: {r.returncode}"
        if not silent:
            print("\n[Диагностика службы времени]")
            print(query_time_service_diagnostics())
            print()
        return False, err_msg
    except Exception as e:
        return False, str(e)


def run_once(config: dict, no_require_admin: bool) -> None:
    require_admin = config.get("REQUIRE_ADMIN", True) and not no_require_admin
    server = config.get("SERVER", "time.windows.com")
    success, msg = sync_time_once(require_admin=require_admin, server=server, silent=False)
    if success:
        print("OK:", msg)
    else:
        print("Ошибка:", msg)
    sys.exit(0 if success else 1)


def run_daemon(config: dict, no_require_admin: bool) -> None:
    if sys.platform != "win32":
        print("Режим daemon только для Windows.")
        sys.exit(1)
    require_admin = config.get("REQUIRE_ADMIN", True) and not no_require_admin
    interval_min = max(1, int(config.get("INTERVAL_MINUTES", 30)))
    interval_sec = interval_min * 60
    server = config.get("SERVER", "time.windows.com")
    admin_ok = check_admin_rights()

    def do_sync() -> tuple[bool, str]:
        return sync_time_once(require_admin=require_admin, server=server, silent=True)

    stop = []

    def on_signal(*_args: object) -> None:
        stop.append(1)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    print("=" * 60)
    print("Синхронизация времени Windows (режим daemon)")
    print(f"Интервал: {interval_min} мин. Сервер: {server}")
    print("Права администратора:", "да" if admin_ok else "нет (синхронизация может не проходить)")
    if no_require_admin:
        print("Режим: --no-require-admin (попытки синхронизации без проверки прав)")
    print("Остановка: Ctrl+C")
    print("=" * 60)

    if admin_ok:
        print("Запуск службы времени (w32time)...")
        ok_start, msg_start = ensure_time_service_running()
        if ok_start:
            print(msg_start)
        else:
            print("Внимание:", msg_start)
        print("Настройка службы времени...")
        configure_time_service(server=server, silent=True)
        time.sleep(5)  # даём службе времени применить настройки перед первой синхронизацией
    if config.get("RUN_ON_START", True):
        # Первая синхронизация с выводом диагностики при ошибке (silent=False)
        ok, msg = sync_time_once(require_admin=require_admin, server=server, silent=False)
        print("При старте:", "OK" if ok else msg)
        if not ok:
            print("(Дальше цикл продолжит попытки по расписанию.)")

    n = 0
    try:
        while not stop:
            # Ждём интервал, проверяя stop каждую секунду — тогда Ctrl+C срабатывает сразу
            for _ in range(interval_sec):
                if stop:
                    break
                time.sleep(1)
            if stop:
                break
            n += 1
            ok, msg = do_sync()
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if ok:
                print(f"[{ts}] #{n} OK: {msg}")
            else:
                print(f"[{ts}] #{n} Ошибка: {msg}")
    except KeyboardInterrupt:
        pass
    print("Остановка по Ctrl+C.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Синхронизация времени Windows: бесконечный цикл каждые 30 мин (по умолчанию)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Один раз синхронизировать и выйти (по умолчанию — бесконечный цикл каждые 30 мин)",
    )
    parser.add_argument(
        "--no-require-admin",
        action="store_true",
        help="Не требовать права администратора (всё равно пробовать w32tm)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        metavar="MIN",
        help="Интервал в минутах (переопределяет конфиг)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Показать статус службы времени (w32tm) и выйти",
    )
    args = parser.parse_args()

    if args.diagnose:
        if sys.platform != "win32":
            print("Диагностика только для Windows.")
            sys.exit(1)
        print("=== Диагностика службы времени Windows ===\n")
        print(query_time_service_diagnostics())
        sys.exit(0)

    config = load_config()
    if args.interval is not None:
        config["INTERVAL_MINUTES"] = max(1, args.interval)

    if args.once:
        run_once(config, args.no_require_admin)
    else:
        run_daemon(config, args.no_require_admin)


if __name__ == "__main__":
    main()
