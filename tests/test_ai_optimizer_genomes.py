#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверяет, что AIStrategyOptimizer подхватывает геномы из data/ai/optimizer_genomes.json
и конвертирует их в ожидаемые диапазоны + метаданные.
"""

import json
import unittest
from pathlib import Path

from bot_engine.ai.ai_strategy_optimizer import AIStrategyOptimizer


class TestAIOptimizerGenomes(unittest.TestCase):
    """Smoke-тест конфигурации геномов оптимизатора."""

    GENOME_PATH = Path('data') / 'ai' / 'optimizer_genomes.json'

    def setUp(self):
        self.original_payload = None
        if self.GENOME_PATH.exists():
            self.original_payload = self.GENOME_PATH.read_text(encoding='utf-8')

    def tearDown(self):
        if self.original_payload is not None:
            self.GENOME_PATH.write_text(self.original_payload, encoding='utf-8')

    def test_optimizer_reads_custom_genomes(self):
        """Оптимизатор должен считывать пользовательские диапазоны и max_tests."""
        test_payload = {
            "version": "unit-test",
            "max_tests": 12,
            "parameters": {
                "rsi_long_threshold": {"min": 10, "max": 14, "step": 2, "type": "int"},
                "take_profit_percent": {"values": [11.5, 12.5, 13.5]},
            },
        }
        self.GENOME_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.GENOME_PATH.write_text(json.dumps(test_payload, ensure_ascii=False, indent=2), encoding='utf-8')

        optimizer = AIStrategyOptimizer()

        self.assertEqual(optimizer.parameter_genomes_meta['version'], 'unit-test')
        self.assertEqual(optimizer.max_genome_tests, 12)

        rsi_range = optimizer._build_range_from_genome('rsi_long_threshold')
        self.assertEqual(rsi_range, [10, 12, 14])

        tp_range = optimizer._build_range_from_genome('take_profit_percent')
        self.assertEqual(tp_range, [11.5, 12.5, 13.5])


if __name__ == '__main__':
    unittest.main()

