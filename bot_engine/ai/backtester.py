"""
Backtesting Engine для тестирования торговых стратегий

Позволяет тестировать AI модули и торговую логику на исторических данных
для оценки производительности и оптимизации параметров.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger('AI.Backtest')

class BacktestEngine:
    """
    Движок для бэктестинга торговых стратегий
    """

    def __init__(self, config: Dict = None):
        """
        Инициализация backtesting engine

        Args:
            config: Конфигурация бэктеста
        """
        self.config = config or {
            'initial_balance': 10000.0,  # Начальный баланс USDT
            'leverage': 10,  # Кредитное плечо
            'maker_fee': 0.0002,  # 0.02%
            'taker_fee': 0.0006,  # 0.06%
            'slippage': 0.001,  # 0.1% проскальзывание
            'max_positions': 5,  # Максимум одновременных позиций
            'position_size_pct': 0.2  # 20% баланса на позицию
        }

        # Состояние симуляции
        self.balance = self.config['initial_balance']
        self.positions = []  # Открытые позиции
        self.trade_history = []  # История сделок
        self.equity_curve = []  # Кривая капитала

        # Статистика
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'max_equity': self.config['initial_balance']
        }

    def run_backtest(
        self,
        historical_data: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        strategy_func=None
    ) -> Dict:
        """
        Запускает бэктест на исторических данных

        Args:
            historical_data: Словарь {symbol: DataFrame с OHLCV}
            start_date: Начальная дата бэктеста
            end_date: Конечная дата бэктеста
            strategy_func: Функция стратегии, которая возвращает сигналы

        Returns:
            Результаты бэктеста
        """
        logger.info("[BACKTEST] Запуск бэктеста...")
        logger.info(f"[BACKTEST] Начальный баланс: {self.balance} USDT")
        logger.info(f"[BACKTEST] Количество монет: {len(historical_data)}")

        # Сброс состояния
        self.reset()

        # Если нет strategy_func, используем встроенную стратегию бота
        if strategy_func is None:
            strategy_func = self._default_strategy

        # Объединяем все временные метки
        all_timestamps = set()
        for symbol, df in historical_data.items():
            if 'timestamp' in df.columns:
                all_timestamps.update(df['timestamp'].values)
            elif 'time' in df.columns:
                all_timestamps.update(df['time'].values)

        timestamps = sorted(list(all_timestamps))

        # Фильтруем по датам
        if start_date:
            timestamps = [t for t in timestamps if t >= start_date.timestamp() * 1000]
        if end_date:
            timestamps = [t for t in timestamps if t <= end_date.timestamp() * 1000]

        logger.info(f"[BACKTEST] Период: {len(timestamps)} временных меток")

        # Прогоняем по временным меткам
        for i, timestamp in enumerate(timestamps):
            # Обновляем метрики каждые 10%
            if i % (len(timestamps) // 10) == 0:
                progress = (i / len(timestamps)) * 100
                logger.info(f"[BACKTEST] Прогресс: {progress:.0f}%")

            # Получаем текущие цены для всех монет
            current_prices = {}
            for symbol, df in historical_data.items():
                time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
                candle = df[df[time_col] == timestamp]
                if not candle.empty:
                    current_prices[symbol] = candle.iloc[0]['close']

            # Обновляем открытые позиции
            self._update_positions(current_prices, timestamp)

            # Генерируем сигналы стратегии
            signals = strategy_func(historical_data, timestamp, current_prices)

            # Обрабатываем сигналы
            if signals:
                self._process_signals(signals, current_prices, timestamp)

            # Сохраняем состояние капитала
            current_equity = self._calculate_equity(current_prices)
            self.equity_curve.append({
                'timestamp': timestamp,
                'balance': self.balance,
                'equity': current_equity
            })

        # Закрываем все открытые позиции
        final_prices = current_prices
        for position in list(self.positions):
            self._close_position(position, final_prices[position['symbol']], timestamps[-1], reason='Backtest ended')

        # Вычисляем финальную статистику
        results = self._calculate_results()

        logger.info("[BACKTEST] ✅ Бэктест завершен")
        logger.info(f"[BACKTEST] Финальный баланс: {self.balance:.2f} USDT")
        logger.info(f"[BACKTEST] PnL: {results['total_pnl_pct']:+.2f}%")
        logger.info(f"[BACKTEST] Win Rate: {results['win_rate']:.1f}%")

        return results

    def _default_strategy(
        self,
        historical_data: Dict[str, pd.DataFrame],
        timestamp: int,
        current_prices: Dict[str, float]
    ) -> List[Dict]:
        """
        Стандартная стратегия бота (упрощенная для бэктеста)

        Returns:
            Список сигналов [{symbol, signal, confidence}]
        """
        # TODO: Реализовать упрощенную версию логики бота
        # Пока возвращаем пустой список
        return []

    def _update_positions(self, current_prices: Dict[str, float], timestamp: int):
        """Обновляет открытые позиции и проверяет SL/TP"""
        for position in list(self.positions):
            symbol = position['symbol']

            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]

            # Проверяем Stop Loss
            if position['side'] == 'LONG':
                pnl_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100

                if current_price <= position['stop_loss']:
                    self._close_position(position, current_price, timestamp, reason='Stop Loss')
                elif current_price >= position['take_profit']:
                    self._close_position(position, current_price, timestamp, reason='Take Profit')

            elif position['side'] == 'SHORT':
                pnl_pct = ((position['entry_price'] - current_price) / position['entry_price']) * 100

                if current_price >= position['stop_loss']:
                    self._close_position(position, current_price, timestamp, reason='Stop Loss')
                elif current_price <= position['take_profit']:
                    self._close_position(position, current_price, timestamp, reason='Take Profit')

    def _process_signals(
        self,
        signals: List[Dict],
        current_prices: Dict[str, float],
        timestamp: int
    ):
        """Обрабатывает торговые сигналы"""
        # Проверяем лимит позиций
        if len(self.positions) >= self.config['max_positions']:
            return

        for signal in signals:
            symbol = signal.get('symbol')
            signal_type = signal.get('signal')  # 'LONG' or 'SHORT'
            confidence = signal.get('confidence', 0.5)

            if symbol not in current_prices:
                continue

            # Проверяем, нет ли уже позиции по этой монете
            if any(p['symbol'] == symbol for p in self.positions):
                continue

            # Открываем позицию
            if signal_type in ['LONG', 'SHORT']:
                self._open_position(
                    symbol=symbol,
                    side=signal_type,
                    price=current_prices[symbol],
                    timestamp=timestamp,
                    confidence=confidence
                )

    def _open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        timestamp: int,
        confidence: float
    ):
        """Открывает новую позицию"""
        # Вычисляем размер позиции
        position_usdt = self.balance * self.config['position_size_pct']
        position_size = (position_usdt * self.config['leverage']) / price

        # Вычисляем комиссию
        fee = position_usdt * self.config['taker_fee']

        # Учитываем проскальзывание
        entry_price = price * (1 + self.config['slippage']) if side == 'LONG' else price * (1 - self.config['slippage'])

        # Вычисляем SL и TP (упрощенно)
        if side == 'LONG':
            stop_loss = entry_price * 0.97  # -3%
            take_profit = entry_price * 1.06  # +6%
        else:  # SHORT
            stop_loss = entry_price * 1.03  # +3%
            take_profit = entry_price * 0.94  # -6%

        position = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'current_price': entry_price,
            'size': position_size,
            'value_usdt': position_usdt,
            'leverage': self.config['leverage'],
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': timestamp,
            'confidence': confidence,
            'entry_fee': fee
        }

        self.positions.append(position)
        self.balance -= fee  # Вычитаем комиссию

    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        timestamp: int,
        reason: str
    ):
        """Закрывает позицию"""
        # Вычисляем PnL
        if position['side'] == 'LONG':
            pnl_pct = ((exit_price - position['entry_price']) / position['entry_price']) * 100
        else:  # SHORT
            pnl_pct = ((position['entry_price'] - exit_price) / position['entry_price']) * 100

        # PnL в USDT с учетом плеча
        pnl_usdt = position['value_usdt'] * (pnl_pct / 100) * self.config['leverage']

        # Комиссия на выход
        exit_fee = position['value_usdt'] * self.config['taker_fee']

        # Итоговый PnL
        net_pnl = pnl_usdt - exit_fee

        # Обновляем баланс
        self.balance += net_pnl

        # Записываем в историю
        trade = {
            'symbol': position['symbol'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'duration': timestamp - position['entry_time'],
            'pnl_pct': pnl_pct,
            'pnl_usdt': net_pnl,
            'reason': reason,
            'confidence': position['confidence']
        }

        self.trade_history.append(trade)

        # Обновляем статистику
        self.stats['total_trades'] += 1
        if net_pnl > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1

        self.stats['total_pnl'] += net_pnl

        # Удаляем позицию
        self.positions.remove(position)

    def _calculate_equity(self, current_prices: Dict[str, float]) -> float:
        """Вычисляет текущий капитал (баланс + unrealized PnL)"""
        equity = self.balance

        for position in self.positions:
            symbol = position['symbol']
            if symbol in current_prices:
                current_price = current_prices[symbol]

                if position['side'] == 'LONG':
                    pnl_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100
                else:
                    pnl_pct = ((position['entry_price'] - current_price) / position['entry_price']) * 100

                pnl_usdt = position['value_usdt'] * (pnl_pct / 100) * self.config['leverage']
                equity += pnl_usdt

        # Обновляем максимальный капитал
        if equity > self.stats['max_equity']:
            self.stats['max_equity'] = equity

        # Вычисляем drawdown
        drawdown = (self.stats['max_equity'] - equity) / self.stats['max_equity'] * 100
        if drawdown > self.stats['max_drawdown']:
            self.stats['max_drawdown'] = drawdown

        return equity

    def _calculate_results(self) -> Dict:
        """Вычисляет финальные метрики производительности"""
        if not self.trade_history:
            return {
                'error': 'No trades executed',
                'total_trades': 0
            }

        # Базовые метрики
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t['pnl_usdt'] > 0)
        losing_trades = total_trades - winning_trades

        # Win Rate
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        # Total PnL
        total_pnl = sum(t['pnl_usdt'] for t in self.trade_history)
        total_pnl_pct = (total_pnl / self.config['initial_balance']) * 100

        # Average Win/Loss
        wins = [t['pnl_usdt'] for t in self.trade_history if t['pnl_usdt'] > 0]
        losses = [t['pnl_usdt'] for t in self.trade_history if t['pnl_usdt'] < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # Profit Factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Sharpe Ratio (упрощенно)
        returns = [t['pnl_pct'] for t in self.trade_history]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) if len(returns) > 1 and np.std(returns) > 0 else 0

        # Max Consecutive Wins/Losses
        max_consecutive_wins = self._calculate_max_consecutive(self.trade_history, win=True)
        max_consecutive_losses = self._calculate_max_consecutive(self.trade_history, win=False)

        # Средняя длительность сделки
        durations = [t['duration'] for t in self.trade_history if t['duration'] > 0]
        avg_duration_ms = np.mean(durations) if durations else 0
        avg_duration_hours = avg_duration_ms / (1000 * 3600)  # Конвертируем в часы

        return {
            'success': True,

            # Общие метрики
            'initial_balance': self.config['initial_balance'],
            'final_balance': self.balance,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,

            # Сделки
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,

            # Profit метрики
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,

            # Риск метрики
            'max_drawdown': self.stats['max_drawdown'],
            'sharpe_ratio': sharpe_ratio,

            # Streak метрики
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,

            # Время
            'avg_trade_duration_hours': avg_duration_hours,

            # Кривая капитала
            'equity_curve': self.equity_curve,
            'trade_history': self.trade_history
        }

    def _calculate_max_consecutive(self, trades: List[Dict], win: bool) -> int:
        """Вычисляет максимальную серию побед/поражений"""
        max_consecutive = 0
        current_consecutive = 0

        for trade in trades:
            is_win = trade['pnl_usdt'] > 0

            if is_win == win:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def reset(self):
        """Сбрасывает состояние для нового бэктеста"""
        self.balance = self.config['initial_balance']
        self.positions = []
        self.trade_history = []
        self.equity_curve = []
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'max_equity': self.config['initial_balance']
        }

    def save_results(self, results: Dict, filename: str):
        """Сохраняет результаты бэктеста в файл"""
        try:
            os.makedirs('data/ai/backtests', exist_ok=True)
            filepath = f"data/ai/backtests/{filename}"

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"[BACKTEST] Результаты сохранены: {filepath}")

        except Exception as e:
            logger.error(f"[BACKTEST] Ошибка сохранения результатов: {e}")

    def generate_report(self, results: Dict) -> str:
        """Генерирует текстовый отчет о результатах бэктеста"""
        if not results.get('success'):
            return "Backtest failed: No data"

        report = []
        report.append("=" * 60)
        report.append("BACKTEST RESULTS")
        report.append("=" * 60)
        report.append("")

        # Общие результаты
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 60)
        report.append(f"Initial Balance:     {results['initial_balance']:>12.2f} USDT")
        report.append(f"Final Balance:       {results['final_balance']:>12.2f} USDT")
        report.append(f"Total PnL:           {results['total_pnl']:>12.2f} USDT ({results['total_pnl_pct']:+.2f}%)")
        report.append("")

        # Статистика сделок
        report.append("TRADE STATISTICS")
        report.append("-" * 60)
        report.append(f"Total Trades:        {results['total_trades']:>12}")
        report.append(f"Winning Trades:      {results['winning_trades']:>12} ({results['win_rate']:.1f}%)")
        report.append(f"Losing Trades:       {results['losing_trades']:>12}")
        report.append(f"Average Win:         {results['avg_win']:>12.2f} USDT")
        report.append(f"Average Loss:        {results['avg_loss']:>12.2f} USDT")
        report.append(f"Profit Factor:       {results['profit_factor']:>12.2f}")
        report.append("")

        # Риск метрики
        report.append("RISK METRICS")
        report.append("-" * 60)
        report.append(f"Max Drawdown:        {results['max_drawdown']:>12.2f}%")
        report.append(f"Sharpe Ratio:        {results['sharpe_ratio']:>12.2f}")
        report.append(f"Max Consecutive Wins:  {results['max_consecutive_wins']:>10}")
        report.append(f"Max Consecutive Losses:{results['max_consecutive_losses']:>10}")
        report.append("")

        # Время
        report.append("TIME ANALYSIS")
        report.append("-" * 60)
        report.append(f"Avg Trade Duration:  {results['avg_trade_duration_hours']:>12.1f} hours")
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)

class ParameterOptimizer:
    """
    Оптимизатор параметров AI модулей
    """

    def __init__(self):
        self.best_params = None
        self.best_score = -float('inf')
        self.optimization_history = []

    def optimize(
        self,
        historical_data: Dict[str, pd.DataFrame],
        param_ranges: Dict[str, Tuple[float, float, float]],
        metric: str = 'sharpe_ratio',
        max_iterations: int = 50
    ) -> Dict:
        """
        Оптимизирует параметры методом случайного поиска

        Args:
            historical_data: Исторические данные
            param_ranges: Диапазоны параметров {name: (min, max, step)}
            metric: Метрика для оптимизации
            max_iterations: Максимум итераций

        Returns:
            Лучшие найденные параметры
        """
        logger.info(f"[OPTIMIZER] Запуск оптимизации...")
        logger.info(f"[OPTIMIZER] Параметры: {list(param_ranges.keys())}")
        logger.info(f"[OPTIMIZER] Метрика: {metric}")
        logger.info(f"[OPTIMIZER] Итераций: {max_iterations}")

        for i in range(max_iterations):
            # Генерируем случайные параметры
            params = {}
            for param_name, (min_val, max_val, step) in param_ranges.items():
                if step == int(step):  # Integer parameter
                    params[param_name] = np.random.randint(int(min_val), int(max_val) + 1)
                else:  # Float parameter
                    params[param_name] = np.random.uniform(min_val, max_val)

            # Запускаем бэктест с этими параметрами
            backtest = BacktestEngine(config={
                **self.config,
                **params
            })

            results = backtest.run_backtest(historical_data)

            # Оцениваем результат
            score = results.get(metric, -float('inf'))

            # Сохраняем в историю
            self.optimization_history.append({
                'iteration': i + 1,
                'params': params,
                'score': score,
                'results': results
            })

            # Обновляем лучший результат
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                logger.info(f"[OPTIMIZER] Новый лучший результат! {metric}={score:.3f}")
                logger.info(f"[OPTIMIZER] Параметры: {params}")

            # Прогресс
            if (i + 1) % 10 == 0:
                logger.info(f"[OPTIMIZER] Прогресс: {i+1}/{max_iterations}")

        logger.info(f"[OPTIMIZER] ✅ Оптимизация завершена")
        logger.info(f"[OPTIMIZER] Лучший {metric}: {self.best_score:.3f}")
        logger.info(f"[OPTIMIZER] Лучшие параметры: {self.best_params}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history
        }
