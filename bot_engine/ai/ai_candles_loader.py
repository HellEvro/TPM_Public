#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль загрузки ВСЕХ доступных свечей для AI обучения

Загружает максимально возможное количество свечей для всех монет
и сохраняет в БД (таблица candles_history)
"""

import os
import json
import logging
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import concurrent.futures

logger = logging.getLogger('AI.CandlesLoader')


class AICandlesLoader:
    """
    Загрузчик свечей для AI обучения
    
    Загружает ВСЕ доступные свечи для всех монет (максимальный период)
    """
    
    def __init__(self, exchange_obj=None):
        """
        Инициализация загрузчика
        
        Args:
            exchange_obj: Объект биржи (если None, получает через API)
        """
        self.exchange = exchange_obj
        
        # Подключаемся к БД
        try:
            from bot_engine.ai.ai_database import get_ai_database
            self.ai_db = get_ai_database()
            logger.debug("✅ AI Database подключена для AICandlesLoader")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось подключиться к AI Database: {e}")
            self.ai_db = None
        
        # Максимальные периоды для разных бирж
        self.max_periods = {
            'bybit': '200',  # Bybit поддерживает до 200 свечей за раз, но можно запрашивать несколько раз
            'binance': '1000',  # Binance до 1000 свечей
            'okx': '1000'  # OKX до 1000 свечей
        }
        
        logger.info("✅ AICandlesLoader инициализирован")
    
    def get_exchange(self):
        """Получить объект биржи"""
        if self.exchange:
            return self.exchange
        
        try:
            # Пробуем получить через API bots.py
            import requests
            response = requests.get('http://127.0.0.1:5001/api/bots/exchange-info', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    exchange_type = data.get('exchange_type', 'bybit')
                    # Здесь можно создать объект биржи, но проще использовать API
                    return None
        except:
            pass
        
        # Пробуем импортировать напрямую
        try:
            from bots_modules.imports_and_globals import get_exchange
            return get_exchange()
        except:
            return None
    
    def load_all_candles_full_history(self, max_workers: int = 10) -> bool:
        """
        Загружает ВСЕ доступные свечи для всех монет
        
        Использует максимальный период для получения максимального количества свечей
        
        Args:
            max_workers: Количество параллельных потоков
        
        Returns:
            True если успешно загружено
        """
        # Сокращенные логи
        logger.info("📊 Загрузка свечей для AI...")
        
        try:
            exchange = self.get_exchange()
            if not exchange:
                logger.error("❌ Не удалось получить объект биржи")
                return False
            
            # Получаем список всех пар
            logger.info("📊 Получение списка всех торговых пар...")
            try:
                pairs = exchange.get_all_pairs()
                if not pairs:
                    logger.error("=" * 80)
                    logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА: СПИСОК ПАР ПУСТОЙ!")
                    logger.error("=" * 80)
                    logger.error("   💡 Метод exchange.get_all_pairs() вернул пустой список")
                    logger.error("   💡 Проверьте что биржа правильно инициализирована")
                    logger.error("=" * 80)
                    return False
                
                logger.info(f"✅ Получено {len(pairs)} торговых пар")
                logger.info(f"📈 Загружаем максимально доступное количество свечей для каждой монеты...")
                logger.info(f"   💡 Первые 10 пар: {pairs[:10]}")
            except Exception as pairs_error:
                logger.error("=" * 80)
                logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА: НЕ УДАЛОСЬ ПОЛУЧИТЬ СПИСОК ПАР!")
                logger.error("=" * 80)
                logger.error(f"   Ошибка: {pairs_error}")
                import traceback
                logger.error(traceback.format_exc())
                logger.error("=" * 80)
                return False
            
            # Загружаем существующий кэш для инкрементального обновления
            existing_candles_data = self._load_existing_candles()
            existing_candles = {}
            if existing_candles_data:
                if 'candles' in existing_candles_data:
                    existing_candles = existing_candles_data['candles']
                elif isinstance(existing_candles_data, dict) and not existing_candles_data.get('metadata'):
                    existing_candles = existing_candles_data
            
            if existing_candles:
                logger.info(f"📊 Найдено существующих данных для {len(existing_candles)} монет")
                logger.info("💡 Используем инкрементальное обновление: загружаем только новые свечи")
            else:
                logger.info("📊 Полная загрузка: файл не найден, загружаем все свечи с нуля")
            
            # Загружаем свечи параллельно (инкрементально или полностью)
            candles_data = {}
            loaded_count = 0
            updated_count = 0
            new_count = 0
            failed_count = 0
            total_candles = 0
            total_new_candles = 0
            
            # Определяем максимальный период для биржи
            exchange_type = self._detect_exchange_type(exchange)
            max_period = self._get_max_period_for_exchange(exchange_type)
            
            logger.info(f"📊 Используем период: {max_period} для биржи {exchange_type}")
            
            def load_symbol_candles(symbol):
                """Загружает свечи для одного символа (инкрементально или полностью)"""
                try:
                    # Проверяем существующие свечи для этого символа
                    existing_symbol_data = existing_candles.get(symbol, {})
                    existing_candles_list = existing_symbol_data.get('candles', []) if isinstance(existing_symbol_data, dict) else []
                    
                    # Определяем последнюю загруженную свечу
                    last_candle_time = None
                    if existing_candles_list:
                        # Сортируем по времени и берем самую новую
                        sorted_existing = sorted(existing_candles_list, key=lambda x: x.get('time', 0))
                        if sorted_existing:
                            last_candle_time = sorted_existing[-1].get('time', 0)
                            logger.info(f"   📊 {symbol}: найдено {len(existing_candles_list)} существующих свечей, последняя: {datetime.fromtimestamp(last_candle_time/1000).strftime('%Y-%m-%d %H:%M')}")
                    
                    # Используем тот же метод что и bots.py, но с максимальным limit
                    # Для Bybit: используем прямой вызов API с limit=1000
                    all_candles = []
                    
                    # Определяем тип биржи и используем соответствующий метод
                    exchange_type = self._detect_exchange_type(exchange)
                    
                    if exchange_type == 'bybit':
                        # Для Bybit используем пагинацию для получения ВСЕХ доступных свечей
                        # Биржа может выдать максимум 2000 свечей за раз, поэтому делаем несколько запросов
                        try:
                            clean_sym = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
                            
                            # ИНКРЕМЕНТАЛЬНАЯ ЗАГРУЗКА: начинаем с последней загруженной свечи или с текущего времени
                            if last_candle_time:
                                # Загружаем только новые свечи (после последней загруженной)
                                end_time = int(time.time() * 1000)  # Текущее время
                                start_from_time = last_candle_time  # Начинаем с последней загруженной
                                logger.info(f"   🔄 {symbol}: инкрементальное обновление (после {datetime.fromtimestamp(start_from_time/1000).strftime('%Y-%m-%d %H:%M')})")
                                incremental_mode = True
                            else:
                                # Полная загрузка: начинаем с текущего времени и идем в прошлое
                                end_time = int(time.time() * 1000)  # Текущее время в миллисекундах
                                start_from_time = None
                                logger.debug(f"   📊 {symbol}: полная загрузка истории")
                                incremental_mode = False
                            
                            max_candles_per_request = 2000  # ПО 2000 свечей за запрос (максимум биржи)
                            request_count = 0
                            # ВАЖНО: Загружаем ВСЕ доступные свечи через пагинацию
                            # При полной загрузке: загружаем ВСЕ свечи пока они не закончатся
                            # При инкрементальном обновлении: загружаем только новые свечи
                            if incremental_mode:
                                max_requests = 10  # Для инкрементального обновления достаточно 10 запросов
                            else:
                                max_requests = None  # БЕЗ ОГРАНИЧЕНИЙ - загружаем ВСЕ доступные свечи!
                            
                            # Делаем запросы пока не получим все доступные свечи
                            while max_requests is None or request_count < max_requests:
                                try:
                                    response = exchange.client.get_kline(
                                        category="linear",
                                        symbol=f"{clean_sym}USDT",
                                        interval='360',  # 6H свечи
                                        limit=max_candles_per_request,
                                        end=str(end_time)  # Получаем свечи ДО этого времени
                                    )
                                    
                                    # Проверка rate limiting
                                    if response.get('retCode') == 10006:
                                        logger.debug(f"⚠️ Rate limit для {symbol}, ждем 1 секунду...")
                                        time.sleep(1)
                                        continue
                                    
                                    if response and response.get('retCode') == 0:
                                        klines = response['result']['list']
                                        
                                        if not klines or len(klines) == 0:
                                            # Больше нет свечей - это реальный конец истории
                                            break
                                        
                                        # Добавляем свечи (они уже отсортированы от новых к старым)
                                        # При инкрементальном обновлении фильтруем только новые свечи
                                        new_candles_in_batch = 0
                                        for k in klines:
                                            candle_time = int(k[0])
                                            
                                            # При инкрементальном обновлении пропускаем старые свечи
                                            if incremental_mode and start_from_time and candle_time <= start_from_time:
                                                continue  # Эта свеча уже есть в базе
                                            
                                            candle = {
                                                'time': candle_time,
                                                'open': float(k[1]),
                                                'high': float(k[2]),
                                                'low': float(k[3]),
                                                'close': float(k[4]),
                                                'volume': float(k[5])
                                            }
                                            all_candles.append(candle)
                                            new_candles_in_batch += 1
                                        
                                        # Если в инкрементальном режиме не получили новых свечей - прекращаем
                                        if incremental_mode and new_candles_in_batch == 0:
                                            logger.info(f"   ✅ {symbol}: новых свечей нет, данные актуальны")
                                            break
                                        
                                        # ВАЖНО: Получаем timestamp самой старой свечи для следующего запроса
                                        oldest_timestamp = int(klines[-1][0])  # Последняя свеча в списке - самая старая
                                        
                                        request_count += 1
                                        
                                        # Логируем прогресс каждые 10 запросов или если загрузили много свечей
                                        if request_count % 10 == 0 or len(all_candles) % 10000 == 0:
                                            logger.info(f"   📊 {symbol}: загружено {len(all_candles)} свечей за {request_count} запросов...")
                                        
                                        # ВАЖНО: Обновляем end_time для следующего запроса (идем дальше в прошлое)
                                        # Минус 1 мс чтобы не получить ту же свечу повторно
                                        end_time = oldest_timestamp - 1
                                        
                                        # ВАЖНО: Продолжаем запрашивать пока получаем свечи!
                                        # Прерываем ТОЛЬКО если:
                                        # 1. Получили 0 свечей (реальный конец истории) - уже обработано выше
                                        # 2. В инкрементальном режиме не получили новых свечей - уже обработано выше
                                        # 3. Достигли очень старой даты (больше 3 лет назад) - проверяем ниже
                                        
                                        # Проверка на очень старую дату (больше 3 лет назад)
                                        # Это защита от бесконечного цикла, если биржа возвращает старые данные
                                        oldest_date_days_ago = (int(time.time() * 1000) - oldest_timestamp) / (1000 * 60 * 60 * 24)
                                        if oldest_date_days_ago > 1095:  # Больше 3 лет назад (~1095 дней)
                                            logger.info(f"   ✅ {symbol}: достигнут конец истории (самая старая свеча {oldest_date_days_ago:.0f} дней назад, больше 3 лет)")
                                            break
                                        
                                        # ВАЖНО: НЕ прерываем если получили меньше свечей чем запросили!
                                        # Биржа может возвращать меньше свечей по разным причинам (лимиты, доступность данных)
                                        # Продолжаем запрашивать дальше пока получаем свечи!
                                        
                                        # Небольшая задержка между запросами (уменьшаем для быстрой загрузки)
                                        time.sleep(0.1)
                                    else:
                                        # Ошибка API - прекращаем загрузку для этого символа
                                        break
                                        
                                except Exception as e:
                                    logger.debug(f"⚠️ Ошибка запроса свечей для {symbol} (запрос {request_count + 1}): {e}")
                                    break
                            
                            # Объединяем существующие и новые свечи
                            if existing_candles_list and all_candles:
                                # Объединяем и удаляем дубликаты
                                all_candles_dict = {c['time']: c for c in existing_candles_list}
                                for new_candle in all_candles:
                                    all_candles_dict[new_candle['time']] = new_candle
                                
                                # Преобразуем обратно в список и сортируем
                                all_candles = sorted(all_candles_dict.values(), key=lambda x: x['time'])
                                new_candles_count = len(all_candles) - len(existing_candles_list)
                            elif existing_candles_list:
                                # Только существующие свечи (новых нет)
                                all_candles = existing_candles_list
                                new_candles_count = 0
                            else:
                                # Только новые свечи (полная загрузка)
                                new_candles_count = len(all_candles)
                            
                            # Сортируем от старых к новым
                            all_candles.sort(key=lambda x: x['time'])
                            
                            if request_count > 0 or new_candles_count > 0:
                                total_candles_count = len(all_candles)
                                days_history = total_candles_count * 6 / 24  # Примерно дней истории для 6H свечей
                                
                                # Убраны DEBUG логи - они дублировали INFO логи
                                # if incremental_mode and new_candles_count > 0:
                                #     logger.debug(f"📊 {symbol}: Обновлено! Добавлено {new_candles_count} новых свечей...")
                                # elif incremental_mode:
                                #     logger.debug(f"📊 {symbol}: Данные актуальны...")
                                
                                if not incremental_mode:
                                    logger.info(f"📊 {symbol}: Загружено ВСЕ доступные свечи: {total_candles_count} свечей за {request_count} запросов (~{days_history:.0f} дней истории)")
                                    logger.info(f"   💡 Загружали по {max_candles_per_request} свечей за запрос через пагинацию")
                                    logger.info(f"   ✅ Загружены ВСЕ доступные свечи для {symbol}")
                        except Exception as e:
                            logger.debug(f"⚠️ Ошибка пагинации для {symbol}: {e}")
                            # Fallback: используем один запрос с limit=2000 (тоже с инкрементальным обновлением)
                            try:
                                clean_sym = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
                                response = exchange.client.get_kline(
                                    category="linear",
                                    symbol=f"{clean_sym}USDT",
                                    interval='360',
                                    limit=2000  # ПО 2000 свечей за запрос
                                )
                                if response and response.get('retCode') == 0:
                                    klines = response['result']['list']
                                    fallback_new_candles = []
                                    for k in klines:
                                        candle_time = int(k[0])
                                        
                                        # При инкрементальном обновлении пропускаем старые свечи
                                        if incremental_mode and start_from_time and candle_time <= start_from_time:
                                            continue
                                        
                                        candle = {
                                            'time': candle_time,
                                            'open': float(k[1]),
                                            'high': float(k[2]),
                                            'low': float(k[3]),
                                            'close': float(k[4]),
                                            'volume': float(k[5])
                                        }
                                        fallback_new_candles.append(candle)
                                    
                                    # Объединяем с существующими
                                    if existing_candles_list and fallback_new_candles:
                                        all_candles_dict = {c['time']: c for c in existing_candles_list}
                                        for new_candle in fallback_new_candles:
                                            all_candles_dict[new_candle['time']] = new_candle
                                        all_candles = sorted(all_candles_dict.values(), key=lambda x: x['time'])
                                        new_candles_count = len(all_candles) - len(existing_candles_list)
                                    elif existing_candles_list:
                                        all_candles = existing_candles_list
                                        new_candles_count = 0
                                    else:
                                        all_candles = fallback_new_candles
                                        new_candles_count = len(fallback_new_candles)
                                    
                                    all_candles.sort(key=lambda x: x['time'])
                            except:
                                pass
                    else:
                        # Для других бирж используем стандартный метод
                        chart_response = exchange.get_chart_data(symbol, '6h', max_period)
                        if chart_response and chart_response.get('success'):
                            candles = chart_response['data'].get('candles', [])
                            if candles:
                                all_candles.extend(candles)
                    
                    if all_candles:
                        return {
                            'symbol': symbol,
                            'candles': all_candles,
                            'count': len(all_candles),
                            'new_count': new_candles_count if 'new_candles_count' in locals() else len(all_candles),
                            'timeframe': '6h',
                            'loaded_at': datetime.now().isoformat(),
                            'last_candle_time': max(c['time'] for c in all_candles) if all_candles else None,
                            'source': 'ai_full_history_loader',
                            'exchange_type': exchange_type,
                            'requests_made': request_count if exchange_type == 'bybit' else 1,
                            'incremental': incremental_mode if 'incremental_mode' in locals() else False
                        }
                    return None
                    
                except Exception as e:
                    logger.debug(f"⚠️ Ошибка загрузки свечей для {symbol}: {e}")
                    return None
            
            # Загружаем параллельно (сокращенные логи)
            logger.info(f"🚀 Параллельная загрузка: {len(pairs)} пар, {max_workers} потоков")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(load_symbol_candles, symbol): symbol for symbol in pairs}
                
                for future in concurrent.futures.as_completed(futures):
                    symbol = futures[future]
                    try:
                        result = future.result()
                        if result:
                            symbol = result['symbol']
                            candles_data[symbol] = result
                            loaded_count += 1
                            total_candles += result['count']
                            total_new_candles += result.get('new_count', 0)
                            
                            if result.get('incremental', False):
                                updated_count += 1
                            else:
                                new_count += 1
                            
                            # Логируем прогресс каждые 100 монет (реже)
                            if loaded_count % 100 == 0:
                                logger.info(f"📊 Прогресс: {loaded_count}/{len(pairs)} монет, {total_candles} свечей...")
                        else:
                            failed_count += 1
                    except Exception as e:
                        logger.debug(f"⚠️ Ошибка для {symbol}: {e}")
                        failed_count += 1
            
            # Итоговая статистика (кратко)
            logger.info(f"✅ Загрузка завершена: {loaded_count} монет, {total_candles} свечей, {failed_count} ошибок")
            
            # Объединяем с существующими данными
            if existing_candles:
                logger.info(f"📊 Объединяем с существующими данными ({len(existing_candles)} монет)...")
                for symbol, data in existing_candles.items():
                    if symbol not in candles_data:
                        candles_data[symbol] = data
            
            # Проверка данных (тихо)
            if not candles_data:
                logger.error(f"❌ Нет данных для сохранения: {loaded_count} монет загружено, {failed_count} ошибок")
                return False
            
            # Дополнительная проверка валидности
            valid_symbols = sum(1 for data in candles_data.values() 
                               if isinstance(data, dict) and data.get('candles') and len(data.get('candles', [])) > 0)
            
            if valid_symbols == 0:
                logger.error(f"❌ Нет валидных данных: {len(candles_data)} записей, но нет свечей")
                return False
            
            logger.info(f"💾 Сохранение: {len(candles_data)} монет, {total_candles} свечей")
            
            # Сохраняем в файл
            try:
                self._save_candles(candles_data)
                logger.debug("✅ Файл сохранен")
            except Exception as save_error:
                logger.error(f"❌ Ошибка сохранения файла: {save_error}")
                import traceback
                logger.debug(traceback.format_exc())
                return False
            
            # Итоговая статистика (кратко)
            logger.info(f"✅ Загрузка завершена: {loaded_count} монет, {total_candles} свечей, {total_new_candles} новых, {failed_count} ошибок")
            
            # Проверка БД
            if self.ai_db:
                count = self.ai_db.count_candles()
                logger.debug(f"📁 БД: {count:,} свечей")
                return True
            else:
                logger.error("❌ AI Database не доступна")
                return False
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки свечей: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _detect_exchange_type(self, exchange) -> str:
        """Определяет тип биржи"""
        exchange_class = type(exchange).__name__.lower()
        if 'bybit' in exchange_class:
            return 'bybit'
        elif 'binance' in exchange_class:
            return 'binance'
        elif 'okx' in exchange_class:
            return 'okx'
        return 'bybit'  # По умолчанию
    
    def _get_max_period_for_exchange(self, exchange_type: str) -> str:
        """Получить максимальный период для биржи"""
        # ВАЖНО: Загружаем ВСЕ доступные свечи через пагинацию
        # Используем максимальный limit=2000 для получения максимума свечей за запрос
        # Для 6H свечей это даст ~500 дней истории за запрос (2000 * 6 часов = 12000 часов = ~500 дней)
        # С пагинацией загружаем ВСЕ доступные свечи пока они не закончатся
        
        if exchange_type == 'bybit':
            # Bybit поддерживает limit=2000 за запрос, используем пагинацию для загрузки ВСЕХ свечей
            return '2000'  # Максимум за один запрос, загружаем ВСЕ через пагинацию
        elif exchange_type == 'binance':
            return '2000'  # До 2000 свечей за запрос
        elif exchange_type == 'okx':
            return '2000'  # До 2000 свечей за запрос
        
        return '2000'  # По умолчанию максимум
    
    def _load_existing_candles(self) -> Dict:
        """Загрузить существующие свечи из БД"""
        if not self.ai_db:
            return {}
        
        try:
            return self.ai_db.get_all_candles_dict(timeframe='6h')
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки существующих свечей из БД: {e}")
            return {}
    
    def _save_candles(self, candles_data: Dict):
        """Сохранить свечи в БД"""
        # ВАЖНО: Проверяем что есть данные для сохранения
        if not candles_data:
            logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА: candles_data пустой!")
            raise ValueError("candles_data пустой - нечего сохранять")
        
        total_candles_count = sum(info.get('count', 0) if isinstance(info, dict) else 0 for info in candles_data.values())
        logger.info(f"💾 Сохранение {len(candles_data)} монет, {total_candles_count} свечей в БД...")
        
        # Сохраняем ТОЛЬКО в БД
        if not self.ai_db:
            logger.error("❌ AI Database не подключена!")
            raise RuntimeError("AI Database не доступна")
        
        try:
            # Преобразуем формат данных для БД
            db_candles_data = {}
            for symbol, candle_info in candles_data.items():
                if isinstance(candle_info, dict):
                    candles = candle_info.get('candles', [])
                else:
                    candles = candle_info if isinstance(candle_info, list) else []
                
                if candles:
                    db_candles_data[symbol] = candles
            
            if db_candles_data:
                saved_results = self.ai_db.save_candles_batch(db_candles_data, timeframe='6h')
                total_saved = sum(saved_results.values())
                logger.info(f"✅ Сохранено {total_saved} свечей в БД для {len(saved_results)} монет")
            else:
                logger.warning("⚠️ Нет данных для сохранения в БД")
        except Exception as db_error:
            logger.error(f"❌ Ошибка сохранения свечей в БД: {db_error}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def get_candles_for_symbol(self, symbol: str) -> Optional[List[Dict]]:
        """Получить свечи для символа из БД"""
        if not self.ai_db:
            return None
        
        try:
            return self.ai_db.get_candles(symbol, timeframe='6h')
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки свечей для {symbol} из БД: {e}")
            return None

