#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Асинхронный воркер для обновления оптимальных EMA периодов
Запускает optimal_ema.py с параметром --all в фоновом режиме
"""

import os
import sys
import time
import threading
import subprocess
import logging
from datetime import datetime
from typing import Optional

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Настройка логирования
logger = logging.getLogger(__name__)

class OptimalEMAWorker:
    """Асинхронный воркер для обновления оптимальных EMA"""
    
    def __init__(self, update_interval: int = 3600):  # 1 час по умолчанию
        self.update_interval = update_interval
        self.is_running = False
        self.worker_thread = None
        self.last_update = None
        self.process = None
        self.optimal_ema_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'sync', 'optimal_ema.py')
        
        # Проверяем существование скрипта
        if not os.path.exists(self.optimal_ema_script):
            logger.error(f"Скрипт optimal_ema.py не найден: {self.optimal_ema_script}")
            raise FileNotFoundError(f"Скрипт optimal_ema.py не найден: {self.optimal_ema_script}")
    
    def start(self):
        """Запускает воркер в отдельном потоке"""
        if self.is_running:
            logger.warning("[OPTIMAL_EMA_WORKER] Воркер уже запущен")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info(f"[OPTIMAL_EMA_WORKER] 🚀 Воркер запущен (интервал: {self.update_interval} сек)")
    
    def stop(self):
        """Останавливает воркер"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Останавливаем текущий процесс если он запущен
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
                logger.info("[OPTIMAL_EMA_WORKER] Текущий процесс остановлен")
            except subprocess.TimeoutExpired:
                logger.warning("[OPTIMAL_EMA_WORKER] Принудительное завершение процесса")
                self.process.kill()
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        
        logger.info("[OPTIMAL_EMA_WORKER] ⏹️ Воркер остановлен")
    
    def _worker_loop(self):
        """Основной цикл воркера"""
        logger.info("[OPTIMAL_EMA_WORKER] 🔄 Начинаем цикл обновления оптимальных EMA")
        
        while self.is_running:
            try:
                # Запускаем обновление
                self._run_optimal_ema_update()
                
                # Ждем до следующего обновления
                for _ in range(self.update_interval):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"[OPTIMAL_EMA_WORKER] Ошибка в цикле воркера: {e}")
                # При ошибке ждем 5 минут перед повтором
                for _ in range(300):  # 5 минут
                    if not self.is_running:
                        break
                    time.sleep(1)
    
    def _run_optimal_ema_update(self):
        """Запускает обновление оптимальных EMA"""
        try:
            logger.info("[OPTIMAL_EMA_WORKER] 🔄 Запуск обновления оптимальных EMA...")
            
            # Команда для запуска optimal_ema.py с параметром --force (пересчет всех символов)
            cmd = [sys.executable, self.optimal_ema_script, '--force']
            
            # Запускаем процесс
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            
            logger.info(f"[OPTIMAL_EMA_WORKER] 📊 Процесс запущен: PID {self.process.pid}")
            
            # Ждем завершения процесса
            stdout, stderr = self.process.communicate()
            
            if self.process.returncode == 0:
                self.last_update = datetime.now()
                logger.info("[OPTIMAL_EMA_WORKER] ✅ Обновление оптимальных EMA завершено успешно")
                
                # Логируем последние строки вывода для информации
                if stdout:
                    lines = stdout.strip().split('\n')
                    for line in lines[-10:]:  # Последние 10 строк для лучшего понимания прогресса
                        if line.strip():
                            logger.info(f"[OPTIMAL_EMA_WORKER] {line}")
            else:
                logger.error(f"[OPTIMAL_EMA_WORKER] ❌ Ошибка обновления (код: {self.process.returncode})")
                if stderr:
                    logger.error(f"[OPTIMAL_EMA_WORKER] STDERR: {stderr}")
                if stdout:
                    logger.error(f"[OPTIMAL_EMA_WORKER] STDOUT: {stdout}")
            
        except Exception as e:
            logger.error(f"[OPTIMAL_EMA_WORKER] Ошибка запуска обновления: {e}")
        finally:
            self.process = None
    
    def force_update(self):
        """Принудительно запускает обновление"""
        if self.process and self.process.poll() is None:
            logger.warning("[OPTIMAL_EMA_WORKER] Обновление уже выполняется")
            return False
        
        logger.info("[OPTIMAL_EMA_WORKER] 🔄 Принудительное обновление...")
        self._run_optimal_ema_update()
        return True
    
    def get_status(self):
        """Возвращает статус воркера"""
        return {
            'is_running': self.is_running,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'update_interval': self.update_interval,
            'is_updating': self.process is not None and self.process.poll() is None,
            'script_path': self.optimal_ema_script
        }
    
    def set_update_interval(self, interval: int):
        """Устанавливает интервал обновления"""
        if interval < 300:  # Минимум 5 минут
            logger.warning("[OPTIMAL_EMA_WORKER] Интервал не может быть меньше 5 минут")
            return False
        
        self.update_interval = interval
        logger.info(f"[OPTIMAL_EMA_WORKER] Интервал обновления изменен на {interval} секунд")
        return True

# Глобальный экземпляр воркера
optimal_ema_worker = None

def start_optimal_ema_worker(update_interval: int = 3600):
    """Запускает глобальный воркер оптимальных EMA"""
    global optimal_ema_worker
    
    if optimal_ema_worker is not None:
        logger.warning("[OPTIMAL_EMA_WORKER] Воркер уже инициализирован")
        return optimal_ema_worker
    
    try:
        optimal_ema_worker = OptimalEMAWorker(update_interval)
        optimal_ema_worker.start()
        return optimal_ema_worker
    except Exception as e:
        logger.error(f"[OPTIMAL_EMA_WORKER] Ошибка инициализации воркера: {e}")
        return None

def stop_optimal_ema_worker():
    """Останавливает глобальный воркер"""
    global optimal_ema_worker
    
    if optimal_ema_worker is not None:
        optimal_ema_worker.stop()
        optimal_ema_worker = None

def get_optimal_ema_worker():
    """Возвращает глобальный воркер"""
    return optimal_ema_worker
