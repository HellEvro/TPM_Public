#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Добавление методов оценки производительности в ai_continuous_learning.py
"""

def add_performance_methods():
    # Новые методы для добавления в конец файла
    new_methods = '''

    def evaluate_ai_performance(self, trades: List[Dict]) -> Dict:
        """
        Оценивает производительность AI на основе сделок

        Args:
            trades: Список сделок с результатами

        Returns:
            Словарь с метриками производительности AI
        """
        try:
            logger.info("📊 Оценка производительности AI...")

            # Разделяем сделки с AI и без AI
            ai_trades = [t for t in trades if t.get('ai_used', False)]
            non_ai_trades = [t for t in trades if not t.get('ai_used', False)]

            metrics = {
                'total_trades': len(trades),
                'ai_trades': len(ai_trades),
                'non_ai_trades': len(non_ai_trades),
                'ai_trades_percentage': (len(ai_trades) / len(trades) * 100) if trades else 0,
                'evaluation_timestamp': datetime.now().isoformat()
            }

            # Оцениваем AI сделки
            if ai_trades:
                ai_successful = len([t for t in ai_trades if t.get('pnl', 0) > 0])
                ai_win_rate = ai_successful / len(ai_trades) if ai_trades else 0
                ai_avg_pnl = np.mean([t.get('pnl', 0) for t in ai_trades]) if ai_trades else 0
                ai_total_pnl = sum([t.get('pnl', 0) for t in ai_trades])

                metrics.update({
                    'ai_win_rate': ai_win_rate,
                    'ai_avg_pnl': ai_avg_pnl,
                    'ai_total_pnl': ai_total_pnl,
                    'ai_successful_trades': ai_successful,
                    'ai_failed_trades': len(ai_trades) - ai_successful
                })

            # Оцениваем не-AI сделки для сравнения
            if non_ai_trades:
                non_ai_successful = len([t for t in non_ai_trades if t.get('pnl', 0) > 0])
                non_ai_win_rate = non_ai_successful / len(non_ai_trades) if non_ai_trades else 0
                non_ai_avg_pnl = np.mean([t.get('pnl', 0) for t in non_ai_trades]) if non_ai_trades else 0
                non_ai_total_pnl = sum([t.get('pnl', 0) for t in non_ai_trades])

                metrics.update({
                    'non_ai_win_rate': non_ai_win_rate,
                    'non_ai_avg_pnl': non_ai_avg_pnl,
                    'non_ai_total_pnl': non_ai_total_pnl,
                    'non_ai_successful_trades': non_ai_successful,
                    'non_ai_failed_trades': len(non_ai_trades) - non_ai_successful
                })

            # Сравнение AI vs не-AI
            if ai_trades and non_ai_trades:
                win_rate_diff = metrics['ai_win_rate'] - metrics['non_ai_win_rate']
                avg_pnl_diff = metrics['ai_avg_pnl'] - metrics['non_ai_avg_pnl']

                metrics.update({
                    'win_rate_difference': win_rate_diff,
                    'avg_pnl_difference': avg_pnl_diff,
                    'ai_better_win_rate': win_rate_diff > 0,
                    'ai_better_avg_pnl': avg_pnl_diff > 0
                })

                # Определяем общую оценку AI
                ai_score = 0
                if win_rate_diff > 0.05:  # AI лучше на 5%+ по win rate
                    ai_score += 1
                if avg_pnl_diff > 10:  # AI лучше на $10+ в среднем
                    ai_score += 1
                if metrics['ai_win_rate'] > 0.6:  # AI имеет win rate > 60%
                    ai_score += 1

                metrics['ai_performance_score'] = ai_score  # 0-3 шкала
                metrics['ai_performance_rating'] = self._get_performance_rating(ai_score)

                logger.info("📊 Оценка AI:")
                logger.info(f"   Win Rate AI: {metrics['ai_win_rate']:.1%} vs Без AI: {metrics['non_ai_win_rate']:.1%} (разница: {win_rate_diff:.1%})")
                logger.info(f"   Avg PnL AI: ${metrics['ai_avg_pnl']:.2f} vs Без AI: ${metrics['non_ai_avg_pnl']:.2f} (разница: ${avg_pnl_diff:.2f})")
                logger.info(f"   Рейтинг AI: {metrics['ai_performance_rating']} (балл: {ai_score}/3)")

            # Сохраняем метрики в knowledge base
            self.knowledge_base['performance_metrics'] = self.knowledge_base.get('performance_metrics', [])
            self.knowledge_base['performance_metrics'].append(metrics)

            # Ограничиваем историю (последние 100 оценок)
            if len(self.knowledge_base['performance_metrics']) > 100:
                self.knowledge_base['performance_metrics'] = self.knowledge_base['performance_metrics'][-100:]

            self._save_knowledge_base()

            return metrics

        except Exception as e:
            logger.error(f"❌ Ошибка оценки производительности AI: {e}")
            return {}

    def _get_performance_rating(self, score: int) -> str:
        """
        Получить текстовую оценку производительности AI

        Args:
            score: Числовой балл (0-3)

        Returns:
            Текстовая оценка
        """
        ratings = {
            0: "Критично низкая - требует улучшений",
            1: "Низкая - нуждается в доработке",
            2: "Средняя - работает, но можно лучше",
            3: "Высокая - отличная производительность"
        }
        return ratings.get(score, "Неизвестно")

    def get_performance_trends(self) -> Dict:
        """
        Анализирует тренды производительности AI со временем

        Returns:
            Словарь с трендами производительности
        """
        try:
            metrics_history = self.knowledge_base.get('performance_metrics', [])

            if len(metrics_history) < 2:
                return {'error': 'Недостаточно данных для анализа трендов'}

            # Анализируем последние 10 оценок
            recent_metrics = metrics_history[-10:]

            trends = {
                'period_analyzed': len(recent_metrics),
                'win_rate_trend': self._calculate_trend([m.get('ai_win_rate', 0) for m in recent_metrics]),
                'avg_pnl_trend': self._calculate_trend([m.get('ai_avg_pnl', 0) for m in recent_metrics]),
                'performance_score_trend': self._calculate_trend([m.get('ai_performance_score', 0) for m in recent_metrics]),
                'latest_performance': recent_metrics[-1] if recent_metrics else {}
            }

            # Определяем, улучшается ли AI
            improving = (
                trends['win_rate_trend'] > 0 and
                trends['avg_pnl_trend'] > 0 and
                trends['performance_score_trend'] >= 0
            )

            trends['ai_improving'] = improving
            trends['trend_summary'] = "AI улучшается" if improving else "AI стабильна или ухудшается"

            return trends

        except Exception as e:
            logger.error(f"❌ Ошибка анализа трендов производительности: {e}")
            return {'error': str(e)}

    def _calculate_trend(self, values: List[float]) -> float:
        """
        Вычисляет тренд в значениях (линейная регрессия)

        Args:
            values: Список значений

        Returns:
            Коэффициент тренда (положительный = рост, отрицательный = падение)
        """
        try:
            if len(values) < 2:
                return 0

            x = np.arange(len(values))
            y = np.array(values)

            # Линейная регрессия
            slope = np.polyfit(x, y, 1)[0]

            return slope

        except Exception:
            return 0
'''

    # Добавляем методы в конец файла
    with open('bot_engine/ai/ai_continuous_learning.py', 'a', encoding='utf-8') as f:
        f.write(new_methods)

    print("Методы оценки производительности добавлены")

if __name__ == '__main__':
    add_performance_methods()