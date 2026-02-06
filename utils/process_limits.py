# -*- coding: utf-8 -*-
"""
Общий модуль ограничения ОЗУ/CPU процесса для ai.py и bots.py.
Использует AI_MEMORY_PCT / AI_MEMORY_LIMIT_MB из bot_config.SystemConfig или env.
На Windows — Job Object (ProcessMemoryLimit); на Linux/macOS — setrlimit(RLIMIT_AS).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional, Tuple

# Храним handle Job Object на Windows, чтобы лимиты не снимались (не закрывать handle)
_win_job_handles: list = []


def get_total_ram_mb() -> Optional[int]:
    """Возвращает объём общей ОЗУ системы в MB или None при ошибке."""
    try:
        import psutil
        total_bytes = psutil.virtual_memory().total
        return int(total_bytes / (1024 * 1024))
    except Exception:
        pass
    if sys.platform == "linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // 1024
        except Exception:
            pass
    return None


def compute_memory_limit_mb() -> Tuple[Optional[int], Optional[str], Optional[int], Optional[float]]:
    """
    Вычисляет лимит ОЗУ в MB. Источники (приоритет):
    1) bot_config.SystemConfig: AI_MEMORY_PCT, AI_MEMORY_LIMIT_MB
    2) Переменные окружения: AI_MEMORY_PCT, AI_MEMORY_LIMIT_MB
    Возвращает (limit_mb, kind, total_mb, pct) или (None, None, None, None).
    """
    pct_val: Optional[float] = None
    limit_mb_val: Optional[int] = None
    try:
        from bot_engine.config_loader import SystemConfig
        pct_val = getattr(SystemConfig, "AI_MEMORY_PCT", 0) or 0
        limit_mb_val = getattr(SystemConfig, "AI_MEMORY_LIMIT_MB", 0) or 0
    except Exception:
        pass
    if not pct_val and not limit_mb_val:
        pct_str = os.environ.get("AI_MEMORY_PCT", "").strip()
        if pct_str:
            try:
                pct_val = float(pct_str.replace(",", "."))
                pct_val = max(1.0, min(100.0, pct_val))
            except ValueError:
                pct_val = None
        limit_mb_str = os.environ.get("AI_MEMORY_LIMIT_MB", "").strip()
        if limit_mb_str:
            try:
                limit_mb_val = int(limit_mb_str)
            except ValueError:
                limit_mb_val = 0
    if pct_val and pct_val > 0:
        total_mb = get_total_ram_mb()
        if total_mb and total_mb > 0:
            limit_mb = int(total_mb * pct_val / 100.0)
            if limit_mb > 0:
                if limit_mb_val and limit_mb_val > 0:
                    limit_mb = min(limit_mb, limit_mb_val)
                os.environ["AI_MEMORY_LIMIT_MB"] = str(limit_mb)
                return limit_mb, "pct", total_mb, pct_val
    if limit_mb_val and limit_mb_val > 0:
        os.environ["AI_MEMORY_LIMIT_MB"] = str(limit_mb_val)
        return limit_mb_val, "mb", None, None
    return None, None, None, None


def apply_memory_limit_setrlimit(limit_mb: int) -> bool:
    """Применяет лимит ОЗУ через setrlimit(RLIMIT_AS) на Linux/macOS. Возвращает True при успехе."""
    if sys.platform == "win32":
        return False
    try:
        import resource
        limit_bytes = limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        return True
    except (ValueError, OSError, Exception):
        return False


def apply_windows_job_limits(
    memory_mb: Optional[int] = None,
    cpu_pct: Optional[int] = None,
    process_name: str = "AI",
) -> None:
    """
    Windows: Job Object для лимита ОЗУ (ProcessMemoryLimit) и/или CPU (%).
    memory_mb — из env AI_MEMORY_LIMIT_MB если None.
    process_name — префикс для лога ([AI], [Bots]).
    """
    if sys.platform != "win32":
        return
    try:
        import multiprocessing
        if multiprocessing.current_process().name != "MainProcess":
            return
    except Exception:
        pass
    if memory_mb is None:
        try:
            limit_str = os.environ.get("AI_MEMORY_LIMIT_MB", "").strip()
            if limit_str:
                memory_mb = int(limit_str)
            else:
                memory_mb = 0
        except ValueError:
            memory_mb = 0
    if cpu_pct is None:
        try:
            from bot_engine.config_loader import SystemConfig
            cpu_pct = getattr(SystemConfig, "AI_CPU_PCT", 0) or 0
        except Exception:
            cpu_pct = 0
        if not cpu_pct:
            pct_str = os.environ.get("AI_CPU_PCT", "").strip()
            if pct_str:
                try:
                    cpu_pct = int(float(pct_str.replace(",", ".")))
                    cpu_pct = max(1, min(100, cpu_pct))
                except ValueError:
                    cpu_pct = 0
    if not memory_mb and not cpu_pct:
        return
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        CreateJobObjectW = kernel32.CreateJobObjectW
        CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        CreateJobObjectW.restype = wintypes.HANDLE
        AssignProcessToJobObject = kernel32.AssignProcessToJobObject
        AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        AssignProcessToJobObject.restype = wintypes.BOOL
        SetInformationJobObject = kernel32.SetInformationJobObject
        SetInformationJobObject.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p]
        SetInformationJobObject.restype = wintypes.BOOL
        job = CreateJobObjectW(None, None)
        if not job:
            return
        if not AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            kernel32.CloseHandle(job)
            return
        applied: list = []
        if memory_mb and memory_mb > 0:
            JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100
            JobObjectExtendedLimitInformation = 9
            limit_bytes = memory_mb * 1024 * 1024

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong),
                ]

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_ulonglong),
                    ("PerJobUserTimeLimit", ctypes.c_ulonglong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            basic = JOBOBJECT_BASIC_LIMIT_INFORMATION()
            basic.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
            ext = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            ext.BasicLimitInformation = basic
            ext.ProcessMemoryLimit = limit_bytes
            if SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(ext)):
                applied.append(f"ОЗУ {memory_mb} MB")
        if cpu_pct and cpu_pct > 0:
            JOB_OBJECT_CPU_RATE_CONTROL_ENABLE = 0x1
            JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x4
            JobObjectCpuRateControlInformation = 15
            cpu_rate = cpu_pct * 100

            class JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("ControlFlags", wintypes.DWORD),
                    ("CpuRate", wintypes.DWORD),
                ]

            info = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(
                ControlFlags=JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
                CpuRate=cpu_rate,
            )
            if SetInformationJobObject(job, JobObjectCpuRateControlInformation, ctypes.byref(info)):
                applied.append(f"CPU {cpu_pct}%")
        if applied:
            sys.stderr.write(f"[{process_name}] Windows Job Object: {', '.join(applied)}\n")
            _win_job_handles.append(job)
        else:
            kernel32.CloseHandle(job)
    except Exception:
        pass
