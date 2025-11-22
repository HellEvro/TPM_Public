-- ============================================================================
-- Миграция 001: Перенос сделок из других баз данных в bot_trades_history
-- ============================================================================
-- 
-- Описание:
--   Переносит сделки из ai_data.db (bot_trades, exchange_trades) 
--   и app_data.db (closed_pnl) в bots_data.db -> bot_trades_history
--
-- Условия выполнения:
--   - Выполняется только один раз (проверяется флаг в db_metadata)
--   - Пути к базам данных подставляются через Python
--
-- Плейсхолдеры:
--   {AI_DB_PATH} - путь к ai_data.db
--   {APP_DB_PATH} - путь к app_data.db
-- ============================================================================

-- Проверяем, выполнена ли миграция
-- (этот запрос выполняется в Python коде перед запуском скрипта)

-- ============================================================================
-- МИГРАЦИЯ ИЗ ai_data.db -> bot_trades
-- ============================================================================

-- Подключаем ai_data.db
ATTACH DATABASE '{AI_DB_PATH}' AS ai_db;

-- Проверяем наличие таблицы bot_trades и мигрируем
-- (проверка выполняется в Python коде)

-- Миграция bot_trades
INSERT OR IGNORE INTO bot_trades_history (
    bot_id, symbol, direction, entry_price, exit_price,
    entry_time, exit_time, entry_timestamp, exit_timestamp,
    position_size_usdt, position_size_coins, pnl, roi,
    status, close_reason, decision_source, ai_decision_id,
    ai_confidence, entry_rsi, exit_rsi, entry_trend, exit_trend,
    entry_volatility, entry_volume_ratio, is_successful,
    is_simulated, source, order_id, extra_data_json,
    created_at, updated_at
)
SELECT 
    COALESCE(bot_id, 'unknown') as bot_id,
    symbol,
    COALESCE(direction, 'LONG') as direction,
    entry_price,
    exit_price,
    entry_time,
    exit_time,
    CASE 
        WHEN entry_time IS NOT NULL 
        THEN CAST((julianday(entry_time) - 2440587.5) * 86400.0 AS REAL)
        ELSE NULL
    END as entry_timestamp,
    CASE 
        WHEN exit_time IS NOT NULL 
        THEN CAST((julianday(exit_time) - 2440587.5) * 86400.0 AS REAL)
        ELSE NULL
    END as exit_timestamp,
    position_size as position_size_usdt,
    position_size_coins,
    pnl,
    roi,
    COALESCE(status, 'CLOSED') as status,
    close_reason,
    'AI_BOT_TRADE' as decision_source,
    ai_decision_id,
    ai_confidence,
    entry_rsi,
    exit_rsi,
    entry_trend,
    exit_trend,
    entry_volatility,
    entry_volume_ratio,
    CASE WHEN pnl > 0 THEN 1 ELSE 0 END as is_successful,
    0 as is_simulated,
    'ai_bot' as source,
    order_id,
    json_object(
        'rsi_params', rsi_params,
        'risk_params', risk_params,
        'config_params', config_params,
        'filters_params', filters_params,
        'entry_conditions', entry_conditions,
        'exit_conditions', exit_conditions,
        'restrictions', restrictions,
        'extra_config', extra_config_json
    ) as extra_data_json,
    COALESCE(created_at, datetime('now')) as created_at,
    COALESCE(updated_at, datetime('now')) as updated_at
FROM ai_db.bot_trades
WHERE status = 'CLOSED' AND pnl IS NOT NULL;

-- Миграция exchange_trades
INSERT OR IGNORE INTO bot_trades_history (
    bot_id, symbol, direction, entry_price, exit_price,
    entry_time, exit_time, entry_timestamp, exit_timestamp,
    position_size_usdt, position_size_coins, pnl, roi,
    status, close_reason, decision_source, ai_decision_id,
    ai_confidence, entry_rsi, exit_rsi, entry_trend, exit_trend,
    entry_volatility, entry_volume_ratio, is_successful,
    is_simulated, source, order_id, extra_data_json,
    created_at, updated_at
)
SELECT 
    COALESCE(bot_id, 'exchange') as bot_id,
    symbol,
    COALESCE(direction, 'LONG') as direction,
    entry_price,
    exit_price,
    entry_time,
    exit_time,
    CASE 
        WHEN entry_time IS NOT NULL 
        THEN CAST((julianday(entry_time) - 2440587.5) * 86400.0 AS REAL)
        ELSE NULL
    END as entry_timestamp,
    CASE 
        WHEN exit_time IS NOT NULL 
        THEN CAST((julianday(exit_time) - 2440587.5) * 86400.0 AS REAL)
        ELSE NULL
    END as exit_timestamp,
    position_size as position_size_usdt,
    position_size_coins,
    pnl,
    roi,
    COALESCE(status, 'CLOSED') as status,
    close_reason,
    'EXCHANGE' as decision_source,
    ai_decision_id,
    ai_confidence,
    entry_rsi,
    exit_rsi,
    entry_trend,
    exit_trend,
    entry_volatility,
    entry_volume_ratio,
    CASE WHEN pnl > 0 THEN 1 ELSE 0 END as is_successful,
    CASE WHEN is_real = 0 OR is_real IS NULL THEN 1 ELSE 0 END as is_simulated,
    'exchange' as source,
    order_id,
    json_object(
        'is_real', is_real,
        'exchange', exchange,
        'extra_data', extra_data_json
    ) as extra_data_json,
    COALESCE(created_at, datetime('now')) as created_at,
    COALESCE(updated_at, datetime('now')) as updated_at
FROM ai_db.exchange_trades
WHERE status = 'CLOSED' AND pnl IS NOT NULL
  AND (is_real = 1 OR is_real IS NULL);

-- Отключаем ai_db
DETACH DATABASE ai_db;

-- ============================================================================
-- МИГРАЦИЯ ИЗ app_data.db -> closed_pnl
-- ============================================================================

-- Подключаем app_data.db
ATTACH DATABASE '{APP_DB_PATH}' AS app_db;

-- Миграция closed_pnl
-- (проверка наличия таблицы и полей выполняется в Python коде)
INSERT OR IGNORE INTO bot_trades_history (
    bot_id, symbol, direction, entry_price, exit_price,
    entry_time, exit_time, entry_timestamp, exit_timestamp,
    position_size_usdt, position_size_coins, pnl, roi,
    status, close_reason, decision_source, ai_decision_id,
    ai_confidence, entry_rsi, exit_rsi, entry_trend, exit_trend,
    entry_volatility, entry_volume_ratio, is_successful,
    is_simulated, source, order_id, extra_data_json,
    created_at, updated_at
)
SELECT 
    COALESCE(bot_id, 'app') as bot_id,
    symbol,
    COALESCE(direction, 'LONG') as direction,
    entry_price,
    exit_price,
    entry_time,
    exit_time,
    CASE 
        WHEN entry_time IS NOT NULL 
        THEN CAST((julianday(entry_time) - 2440587.5) * 86400.0 AS REAL)
        ELSE NULL
    END as entry_timestamp,
    CASE 
        WHEN exit_time IS NOT NULL 
        THEN CAST((julianday(exit_time) - 2440587.5) * 86400.0 AS REAL)
        ELSE NULL
    END as exit_timestamp,
    position_size as position_size_usdt,
    position_size_coins,
    pnl,
    roi,
    'CLOSED' as status,
    close_reason,
    'APP_CLOSED_PNL' as decision_source,
    NULL as ai_decision_id,
    NULL as ai_confidence,
    NULL as entry_rsi,
    NULL as exit_rsi,
    NULL as entry_trend,
    NULL as exit_trend,
    NULL as entry_volatility,
    NULL as entry_volume_ratio,
    CASE WHEN pnl > 0 THEN 1 ELSE 0 END as is_successful,
    0 as is_simulated,
    'app_closed_pnl' as source,
    order_id,
    COALESCE(extra_data_json, '{}') as extra_data_json,
    COALESCE(created_at, datetime('now')) as created_at,
    COALESCE(updated_at, datetime('now')) as updated_at
FROM app_db.closed_pnl
WHERE pnl IS NOT NULL;

-- Отключаем app_db
DETACH DATABASE app_db;

-- ============================================================================
-- Установка флага миграции
-- ============================================================================
-- (выполняется в Python коде после успешной миграции)

