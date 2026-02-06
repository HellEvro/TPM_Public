#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диагностика компонентов HWID
Показывает какие параметры оборудования используются для генерации Hardware ID
"""

import platform
import hashlib
import subprocess
import uuid
import sys
from pathlib import Path

if platform.system() == 'Windows':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

print("=" * 80)
print("ДИАГНОСТИКА КОМПОНЕНТОВ HWID")
print("=" * 80)
print()
print(f"Операционная система: {platform.system()} {platform.release()}")
print(f"Архитектура: {platform.machine()}")
print()

components = []
component_details = {}

# 1. MAC адрес
print("1. MAC АДРЕС СЕТЕВОЙ КАРТЫ")
print("-" * 80)
try:
    mac_raw = uuid.getnode()
    mac = ':'.join(['{:02x}'.format((mac_raw >> elements) & 0xff)
                   for elements in range(0, 2*6, 2)][::-1])
    
    if mac == '00:00:00:00:00:00' or mac_raw == 0:
        print(f"   ❌ MAC адрес: {mac} (СЛУЧАЙНЫЙ/НЕДОСТУПЕН)")
        print(f"   ⚠️  UUID.getnode() вернул: {mac_raw}")
        print(f"   ⚠️  ВНИМАНИЕ: На миниПК это может меняться после перезагрузки!")
    else:
        print(f"   ✅ MAC адрес: {mac}")
        components.append(f"MAC:{mac}")
        component_details["MAC"] = mac
except Exception as e:
    print(f"   ❌ Ошибка получения MAC: {e}")
print()

# 2. UUID машины (на основе hostname)
print("2. UUID МАШИНЫ (на основе hostname)")
print("-" * 80)
try:
    hostname = platform.node()
    machine_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, hostname))
    print(f"   Hostname: {hostname}")
    print(f"   ✅ UUID: {machine_uuid}")
    components.append(f"UUID:{machine_uuid}")
    component_details["UUID"] = machine_uuid
    print(f"   ⚠️  ВНИМАНИЕ: Если hostname меняется, UUID тоже меняется!")
except Exception as e:
    print(f"   ❌ Ошибка получения UUID: {e}")
print()

# 3. Информация о платформе
print("3. ИНФОРМАЦИЯ О ПЛАТФОРМЕ")
print("-" * 80)
try:
    platform_info = f"{platform.system()}-{platform.machine()}"
    print(f"   ✅ Платформа: {platform_info}")
    components.append(f"PLATFORM:{platform_info}")
    component_details["PLATFORM"] = platform_info
except Exception as e:
    print(f"   ❌ Ошибка получения платформы: {e}")
print()

# 4. Специфичные для Windows данные
if platform.system() == 'Windows':
    print("4. WINDOWS: СЕРИЙНЫЙ НОМЕР ПРОЦЕССОРА")
    print("-" * 80)
    try:
        result = subprocess.check_output(
            'wmic cpu get processorid',
            shell=True,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        cpu_id = result.split('\n')[1].strip() if '\n' in result else result.strip()
        if cpu_id and cpu_id != 'ProcessorId':
            print(f"   ✅ CPU ID: {cpu_id}")
            components.append(f"CPU:{cpu_id}")
            component_details["CPU"] = cpu_id
        else:
            print(f"   ❌ CPU ID не найден")
    except Exception as e:
        print(f"   ❌ Ошибка получения CPU ID: {e}")
    print()
    
    print("5. WINDOWS: СЕРИЙНЫЙ НОМЕР ДИСКА")
    print("-" * 80)
    try:
        result = subprocess.check_output(
            'wmic diskdrive get serialnumber',
            shell=True,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        disk_serial = result.split('\n')[1].strip() if '\n' in result else result.strip()
        if disk_serial and disk_serial != 'SerialNumber':
            print(f"   ✅ Disk Serial: {disk_serial}")
            components.append(f"DISK:{disk_serial}")
            component_details["DISK"] = disk_serial
        else:
            print(f"   ❌ Disk Serial не найден")
    except Exception as e:
        print(f"   ❌ Ошибка получения Disk Serial: {e}")
    print()
    
    # Дополнительная диагностика для Windows
    print("6. WINDOWS: ДОПОЛНИТЕЛЬНАЯ ДИАГНОСТИКА")
    print("-" * 80)
    try:
        # BIOS Serial Number
        result = subprocess.check_output(
            'wmic bios get serialnumber',
            shell=True,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        bios_serial = result.split('\n')[1].strip() if '\n' in result else result.strip()
        if bios_serial and bios_serial != 'SerialNumber':
            print(f"   📋 BIOS Serial: {bios_serial}")
        else:
            print(f"   ❌ BIOS Serial не найден")
    except Exception as e:
        print(f"   ❌ Ошибка получения BIOS Serial: {e}")
    
    try:
        # Baseboard Serial Number
        result = subprocess.check_output(
            'wmic baseboard get serialnumber',
            shell=True,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        board_serial = result.split('\n')[1].strip() if '\n' in result else result.strip()
        if board_serial and board_serial != 'SerialNumber':
            print(f"   📋 Motherboard Serial: {board_serial}")
        else:
            print(f"   ❌ Motherboard Serial не найден")
    except Exception as e:
        print(f"   ❌ Ошибка получения Motherboard Serial: {e}")
    print()

# 5. Специфичные для Linux данные
elif platform.system() == 'Linux':
    print("4. LINUX: MACHINE ID")
    print("-" * 80)
    try:
        with open('/etc/machine-id', 'r') as f:
            machine_id = f.read().strip()
            print(f"   ✅ Machine ID: {machine_id}")
            components.append(f"MACHINE_ID:{machine_id}")
            component_details["MACHINE_ID"] = machine_id
            print(f"   ✅ Machine ID - СТАБИЛЬНЫЙ параметр (не меняется после перезагрузки)")
    except Exception as e:
        print(f"   ❌ Ошибка получения Machine ID: {e}")
    print()
    
    # Дополнительная диагностика для Linux
    print("5. LINUX: ДОПОЛНИТЕЛЬНАЯ ДИАГНОСТИКА")
    print("-" * 80)
    try:
        # CPU Info
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Serial' in cpuinfo:
                for line in cpuinfo.split('\n'):
                    if 'Serial' in line:
                        print(f"   📋 {line.strip()}")
    except Exception as e:
        print(f"   ❌ Ошибка чтения /proc/cpuinfo: {e}")
    
    try:
        # DMI Serial (если доступен)
        result = subprocess.check_output(
            ['cat', '/sys/class/dmi/id/product_serial'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        if result and result != 'Not Specified':
            print(f"   📋 DMI Product Serial: {result}")
    except:
        pass
    print()

# Итоговая комбинация
print("=" * 80)
print("ИТОГОВАЯ КОМБИНАЦИЯ ДЛЯ HWID")
print("=" * 80)
print()
combined = '|'.join(components)
print("Используемые компоненты:")
for i, comp in enumerate(components, 1):
    print(f"   {i}. {comp}")

print()
print(f"Комбинация для хэширования:")
print(f"   {combined}")
print()

hardware_id = hashlib.sha256(combined.encode()).hexdigest()
short_hwid = hardware_id[:16].upper()

print(f"✅ Полный HWID: {hardware_id}")
print(f"✅ Короткий HWID: {short_hwid}")
print()

# Анализ стабильности
print("=" * 80)
print("АНАЛИЗ СТАБИЛЬНОСТИ HWID")
print("=" * 80)
print()

unstable_params = []
stable_params = []

if "MAC" in component_details:
    mac_val = component_details["MAC"]
    if mac_val.startswith("00:00:00:00:00:00") or uuid.getnode() == 0:
        unstable_params.append("MAC - случайный адрес, может меняться после перезагрузки")
    else:
        stable_params.append("MAC - физический адрес, стабильный")

if "UUID" in component_details:
    unstable_params.append("UUID - зависит от hostname, может меняться если меняется hostname")

if "PLATFORM" in component_details:
    stable_params.append("PLATFORM - стабильная")

if "CPU" in component_details:
    stable_params.append("CPU ID - серийный номер процессора, стабильный")

if "DISK" in component_details:
    stable_params.append("DISK - серийный номер диска, стабильный")

if "MACHINE_ID" in component_details:
    stable_params.append("MACHINE_ID - системный ID Linux, стабильный")

if stable_params:
    print("✅ СТАБИЛЬНЫЕ параметры (не меняются после перезагрузки):")
    for param in stable_params:
        print(f"   • {param}")
    print()

if unstable_params:
    print("⚠️  НЕСТАБИЛЬНЫЕ параметры (могут меняться):")
    for param in unstable_params:
        print(f"   • {param}")
    print()

print("=" * 80)
print("РЕКОМЕНДАЦИИ")
print("=" * 80)
print()
print("Для стабильного HWID рекомендуется:")
print("1. Использовать только стабильные параметры оборудования")
print("2. Избегать MAC адреса если он случайный (00:00:00:00:00:00)")
print("3. На Windows: использовать CPU ID + Disk Serial + BIOS/Motherboard Serial")
print("4. На Linux: использовать Machine ID + CPU Serial + DMI Serial")
print("5. Избегать hostname и UUID на его основе")
print()

