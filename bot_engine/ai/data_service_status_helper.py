#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper модуль для перехвата сохранения data_service.json и перенаправления в БД

ВАЖНО: Этот модуль используется для перехвата сохранения в файл data_service.json
и перенаправления в БД (ai_data.db) вместо файла.
"""

import os
import json
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger('AI.DataServiceStatusHelper')

def update_data_service_status_in_db(**kwargs):
    """
    Обновить статус data-service в БД вместо файла

    ВАЖНО: Использует БД вместо файла data_service.json!

    Args:
        **kwargs: Поля статуса для обновления
    """
    try:
        from bot_engine.ai.ai_database import get_ai_database
        ai_db = get_ai_database()
        if not ai_db:
            logger.warning("⚠️ AI Database не доступна, статус не обновлен")
            return

        # Получаем текущий статус из БД
        current_status = ai_db.get_data_service_status('data_service')
        if current_status and current_status.get('status'):
            status = current_status['status']
        else:
            status = {}

        # Обновляем статус
        status.update(kwargs)
        status['timestamp'] = datetime.now().isoformat()

        # Сохраняем в БД
        ai_db.save_data_service_status('data_service', status)

    except Exception as e:
        logger.error(f"❌ Ошибка обновления статуса data-service: {e}")

def get_data_service_status_from_db() -> Optional[Dict]:
    """
    Получить статус data-service из БД вместо файла

    ВАЖНО: Использует БД вместо файла data_service.json!

    Returns:
        Словарь со статусом или None
    """
    try:
        from bot_engine.ai.ai_database import get_ai_database
        ai_db = get_ai_database()
        if not ai_db:
            logger.warning("⚠️ AI Database не доступна")
            return None

        result = ai_db.get_data_service_status('data_service')
        if result and result.get('status'):
            return result['status']
        return None
    except Exception as e:
        logger.error(f"❌ Ошибка получения статуса data-service: {e}")
        return None

# Перехватываем сохранение в файл и перенаправляем в БД
def save_data_service_status_file(filepath: str, status: Dict):
    """
    Сохранить статус data-service (перехватывает сохранение в файл и перенаправляет в БД)

    ВАЖНО: Этот метод перехватывает сохранение в файл и перенаправляет в БД!

    Args:
        filepath: Путь к файлу (игнорируется, используется для совместимости)
        status: Словарь со статусом
    """
    # Игнорируем filepath, сохраняем в БД
    update_data_service_status_in_db(**status)

def load_data_service_status_file(filepath: str) -> Optional[Dict]:
    """
    Загрузить статус data-service (перехватывает загрузку из файла и загружает из БД)

    ВАЖНО: Этот метод перехватывает загрузку из файла и загружает из БД!

    Args:
        filepath: Путь к файлу (игнорируется, используется для совместимости)

    Returns:
        Словарь со статусом или None
    """
    # Игнорируем filepath, загружаем из БД
    return get_data_service_status_from_db()
