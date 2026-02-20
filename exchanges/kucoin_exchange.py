# -*- coding: utf-8 -*-
"""
Адаптер KuCoin Futures (USDT-margined perpetual) через CCXT.
API ключи создаются на https://futures.kucoin.com (отдельно от спота).
"""
from .base_exchange import BaseExchange
import ccxt
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)


def clean_symbol(symbol):
    """Приводит символ к базовой валюте (BTC, ETH, ...). CCXT: BTC/USDT:USDT -> BTC."""
    if not symbol:
        return ''
    base = (symbol.split('/') or [symbol])[0]
    base = (base.split(':') or [base])[0]
    return base.replace('USDT', '').strip()


def to_ccxt_symbol(symbol):
    """Базовый символ (BTC) -> CCXT формат KuCoin Futures: BTC/USDT:USDT."""
    return f"{symbol}/USDT:USDT"


class KucoinExchange(BaseExchange):
    def __init__(self, api_key, api_secret, position_mode='Hedge', limit_order_offset=0.01):
        super().__init__(api_key, api_secret, position_mode, limit_order_offset)
        try:
            self.client = ccxt.kucoinfutures({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'adjustForTimeDifference': True},
                'timeout': 30000,
            })
            self.daily_pnl = {}
            self.last_reset_day = None
            self.max_profit_values = {}
            self.max_loss_values = {}
            try:
                self.client.load_markets()
            except Exception as e:
                logger.warning("[KUCOIN] load_markets: %s", e)
        except Exception as e:
            logger.error("Error initializing KuCoin Futures: %s", e)
            raise

    def get_positions(self):
        try:
            positions = self.client.fetch_positions()
            processed = []
            rapid_growth = []
            for pos in positions:
                try:
                    contracts = float(pos.get('contracts') or 0)
                    if contracts == 0:
                        continue
                    symbol = clean_symbol(pos.get('symbol', ''))
                    unrealized = float(pos.get('unrealizedPnl') or 0)
                    notional = float(pos.get('notional') or 0)
                    leverage = float(pos.get('leverage') or pos.get('lever') or 1)
                    margin = (notional / leverage) if leverage > 0 else notional
                    roi = (unrealized / margin * 100) if margin > 0 else 0
                    if unrealized > 0:
                        self.max_profit_values[symbol] = max(
                            self.max_profit_values.get(symbol, 0), unrealized
                        )
                    else:
                        self.max_loss_values[symbol] = min(
                            self.max_loss_values.get(symbol, 0), unrealized
                        )
                    side = 'Long' if (pos.get('side') == 'long' or float(pos.get('contracts', 0)) > 0) else 'Short'
                    processed.append({
                        'symbol': symbol,
                        'pnl': unrealized,
                        'max_profit': self.max_profit_values.get(symbol, 0),
                        'max_loss': self.max_loss_values.get(symbol, 0),
                        'roi': roi,
                        'high_roi': roi > 100,
                        'high_loss': unrealized < -40,
                        'side': side,
                        'size': abs(contracts),
                        'realized_pnl': float(pos.get('realizedPnl') or 0),
                        'leverage': leverage,
                    })
                    if symbol in self.daily_pnl and self.daily_pnl[symbol] > 0 and unrealized > 0:
                        r = unrealized / self.daily_pnl[symbol]
                        if r >= 2.0:
                            rapid_growth.append({
                                'symbol': symbol,
                                'start_pnl': self.daily_pnl[symbol],
                                'current_pnl': unrealized,
                                'growth_ratio': r,
                            })
                    else:
                        self.daily_pnl[symbol] = unrealized
                except Exception as e:
                    logger.debug("[KUCOIN] skip position: %s", e)
                    continue
            return processed, rapid_growth
        except Exception as e:
            logger.error("[KUCOIN] get_positions: %s", e)
            return [], []

    def get_closed_pnl(self, sort_by='time', period='all', start_date=None, end_date=None):
        try:
            end_ts = int(time.time() * 1000)
            end_dt = datetime.fromtimestamp(end_ts / 1000)
            if period == 'day':
                start_ts = int(end_dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
            elif period == 'week':
                days = end_dt.weekday()
                start_dt = end_dt.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
                start_ts = int(start_dt.timestamp() * 1000)
            elif period == 'month':
                start_dt = end_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                start_ts = int(start_dt.timestamp() * 1000)
            elif period == 'year':
                start_dt = end_dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                start_ts = int(start_dt.timestamp() * 1000)
            elif period == 'custom' and start_date and end_date:
                try:
                    if isinstance(start_date, str) and '-' in start_date:
                        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
                    else:
                        start_ts = int(start_date)
                    if isinstance(end_date, str) and '-' in end_date:
                        end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
                        end_ts = int(end_dt.timestamp() * 1000)
                    else:
                        end_ts = int(end_date)
                except Exception:
                    start_ts = end_ts - (7 * 24 * 60 * 60 * 1000)
            else:
                start_ts = end_ts - (730 * 24 * 60 * 60 * 1000)
            result = []
            try:
                trades = self.client.fetch_my_trades(since=start_ts, limit=200)
                for t in trades:
                    ts = int(t.get('timestamp') or 0) * 1000 if t.get('timestamp') else 0
                    if ts < start_ts or ts > end_ts:
                        continue
                    fee = t.get('fee') or {}
                    cost = fee.get('cost')
                    if cost is None:
                        continue
                    pnl = float(cost) if isinstance(cost, (int, float)) else 0.0
                    sym = clean_symbol(t.get('symbol', ''))
                    result.append({
                        'symbol': sym,
                        'qty': float(t.get('amount') or 0),
                        'entry_price': float(t.get('price') or 0),
                        'exit_price': float(t.get('price') or 0),
                        'closed_pnl': pnl,
                        'close_time': datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                        'exchange': 'kucoin',
                        'close_timestamp': ts,
                    })
            except Exception as e:
                logger.warning("[KUCOIN] get_closed_pnl fetch_my_trades: %s", e)
            if sort_by == 'pnl':
                result.sort(key=lambda x: abs(float(x.get('closed_pnl', 0))), reverse=True)
            else:
                result.sort(key=lambda x: x.get('close_timestamp', 0), reverse=True)
            return result
        except Exception as e:
            logger.error("[KUCOIN] get_closed_pnl: %s", e)
            return []

    def get_symbol_chart_data(self, symbol):
        try:
            market = to_ccxt_symbol(symbol)
            ohlcv = self.client.fetch_ohlcv(market, '5m', limit=24)
            if not ohlcv:
                return []
            return [float(c[4]) for c in ohlcv]
        except Exception as e:
            logger.error("[KUCOIN] get_symbol_chart_data %s: %s", symbol, e)
            return []

    def get_sma200_position(self, symbol):
        try:
            market = to_ccxt_symbol(symbol)
            klines = self.client.fetch_ohlcv(market, '1d', limit=200)
            if len(klines) < 200:
                return None
            closes = [float(k[4]) for k in klines]
            sma200 = sum(closes[:200]) / 200
            return float(closes[-1]) > sma200
        except Exception as e:
            logger.error("[KUCOIN] get_sma200_position %s: %s", symbol, e)
            return None

    def get_ticker(self, symbol):
        try:
            ticker = self.client.fetch_ticker(to_ccxt_symbol(symbol))
            return {
                'symbol': symbol,
                'last': float(ticker.get('last') or 0),
                'bid': float(ticker.get('bid') or 0),
                'ask': float(ticker.get('ask') or 0),
                'timestamp': int(ticker.get('timestamp') or time.time() * 1000),
            }
        except Exception as e:
            logger.error("[KUCOIN] get_ticker %s: %s", symbol, e)
            return None

    def close_position(self, symbol, size, side, order_type="Limit"):
        try:
            market = to_ccxt_symbol(symbol)
            ticker = self.get_ticker(symbol)
            if not ticker:
                return {'success': False, 'message': 'Could not get current price'}
            reduce_side = 'sell' if side == 'Long' else 'buy'
            params = {'reduceOnly': True}
            if order_type and order_type.upper() == "LIMIT":
                pct = (100 - self.limit_order_offset) / 100 if reduce_side == 'buy' else (100 + self.limit_order_offset) / 100
                price = (ticker['ask'] * pct) if reduce_side == 'buy' else (ticker['bid'] * pct)
                order = self.client.create_order(
                    market, 'limit', reduce_side, abs(size), price, params
                )
            else:
                order = self.client.create_order(
                    market, 'market', reduce_side, abs(size), None, params
                )
            oid = order.get('id') or order.get('orderId')
            return {
                'success': True,
                'order_id': str(oid) if oid else None,
                'message': f'{order_type or "Market"} order placed',
                'close_price': float(ticker['last']),
            }
        except Exception as e:
            logger.error("[KUCOIN] close_position %s: %s", symbol, e)
            return {'success': False, 'message': str(e)}

    def get_all_pairs(self):
        try:
            markets = self.client.fetch_markets()
            pairs = [
                clean_symbol(m.get('symbol', m.get('id', '')))
                for m in markets
                if m.get('type') in ('swap', 'future') and m.get('quote') == 'USDT' and m.get('active', True)
            ]
            return sorted(set(pairs))
        except Exception as e:
            logger.error("[KUCOIN] get_all_pairs: %s", e)
            return []

    def place_order(self, symbol, side, quantity, order_type='market', price=None, leverage=None, **kwargs):
        """Размещение ордера. side: LONG/SHORT или BUY/SELL."""
        try:
            market = to_ccxt_symbol(symbol)
            if leverage is not None:
                try:
                    self.client.set_leverage(int(leverage), market)
                except Exception as le:
                    logger.warning("[KUCOIN] set_leverage %s: %s", symbol, le)
            side_lower = (side or '').lower()
            if side_lower in ('long', 'buy'):
                order_side = 'buy'
            else:
                order_side = 'sell'
            if order_type and order_type.lower() == 'limit' and price is not None:
                order = self.client.create_order(market, 'limit', order_side, quantity, price, {})
            else:
                order = self.client.create_order(market, 'market', order_side, quantity, None, {})
            return {
                'success': True,
                'order_id': str(order.get('id') or order.get('orderId') or ''),
                'message': 'Order placed',
            }
        except Exception as e:
            logger.error("[KUCOIN] place_order %s: %s", symbol, e)
            return {'success': False, 'message': str(e)}

    def get_unified_account_info(self):
        """Единый формат информации о счёте для UI и ботов."""
        try:
            # KuCoin Futures может возвращать баланс по одной валюте; пробуем USDT
            balance = self.client.fetch_balance()
            total = float(balance.get('total', {}).get('USDT', 0))
            free = float(balance.get('free', {}).get('USDT', 0))
            info_data = (balance.get('info') or {}).get('data') or {}
            if total == 0 and free == 0 and info_data:
                total = float(info_data.get('accountEquity') or info_data.get('marginBalance') or 0)
                free = float(info_data.get('availableBalance') or 0)
            positions, _ = self.get_positions()
            unrealized = sum(float(p.get('pnl', 0)) for p in positions)
            return {
                'success': True,
                'total_equity': total,
                'total_wallet_balance': total,
                'total_available_balance': free,
                'total_unrealized_pnl': unrealized,
                'total_margin_balance': total,
                'account_type': 'futures',
                'active_positions': len(positions),
                'total_position_value': 0.0,
            }
        except Exception as e:
            logger.error("[KUCOIN] get_unified_account_info: %s", e)
            return {
                'success': False,
                'error': str(e),
                'total_wallet_balance': 0,
                'total_available_balance': 0,
                'active_positions': 0,
                'total_position_value': 0,
            }
