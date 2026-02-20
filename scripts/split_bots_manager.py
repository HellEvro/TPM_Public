#!/usr/bin/env python3
"""
Скрипт для разбиения bots_manager.js на модули.
Каждый модуль добавляет методы к BotsManager.prototype через Object.assign.
"""
import re
from pathlib import Path

# Путь к исходному файлу
SRC = Path(__file__).resolve().parent.parent / "static" / "js" / "managers" / "bots_manager.js"
OUT_DIR = Path(__file__).resolve().parent.parent / "static" / "js" / "managers" / "bots_manager"

# Группировка методов по модулям (порядок важен — core должен быть первым)
MODULES = {
    "00_core": [
        "constructor", "logDebug", "logInfo", "logError", "getTranslation", "init"
    ],
    "01_interface": [
        "initializeInterface", "applyReadabilityStyles", "initializeTabs", "switchTab"
    ],
    "02_search": [
        "initializeSearch", "updateClearButtonVisibility", "clearSearch"
    ],
    "03_filters": [
        "initializeManagementButtons", "initializeRsiFilters", "updateRsiFilterButtons",
        "initActiveBotsFilters", "getFilteredActiveBotsForDetails", "getVirtualPositionsAsBots",
        "updateActiveBotsFilterCounts", "updateTrendFilterLabels", "updateRsiThresholds",
        "refreshCoinsRsiClasses"
    ],
    "04_service": [
        "initializeBotControls", "checkBotsService", "updateServiceStatus",
        "showServiceUnavailable", "loadCoinsRsiData", "loadDelistedCoins",
        "loadMatureCoinsCount", "loadMatureCoinsAndMark", "updateCoinsCounter"
    ],
    "05_coins_display": [
        "renderCoinsList", "generateWarningIndicator", "generateEnhancedSignalInfo",
        "generateTimeFilterInfo", "generateExitScamFilterInfo", "generateAntiPumpFilterInfo",
        "getRsiZoneClass", "createTickerLink", "getExchangeLink", "updateManualPositionCounter",
        "getEffectiveSignal", "updateSignalCounters", "selectCoin", "showBotControlInterface",
        "updateCoinInfo", "updateActiveCoinIcons", "getRsiZone", "updateCoinStatusIcons",
        "updateStatusIcon", "updateFilterItem", "getStatusIcon", "forceShowAllFilters",
        "filterCoins", "applyRsiFilter", "restoreFilterState"
    ],
    "06_duplicates_config": [
        "collectDuplicateSettings", "loadIndividualSettings", "saveIndividualSettings",
        "deleteIndividualSettings", "copySettingsToAllCoins", "learnExitScamForCoin",
        "learnExitScamForAllCoins", "resetExitScamToConfigForAll", "resetAllCoinsToGlobalSettings",
        "getIndividualSettingsElementMap", "clearIndividualSettingDiffHighlights",
        "highlightIndividualSettingDiffs", "applyIndividualSettingsToUI",
        "updateIndividualSettingsStatus", "loadAndApplyIndividualSettings",
        "initializeIndividualSettingsButtons", "initializeQuickLaunchButtons"
    ],
    "07_bot_controls": [
        "createBot", "startBot", "stopBot", "pauseBot", "resumeBot", "deleteBot",
        "quickLaunchBot", "updateBotStatusInUI", "removeBotFromUI", "getBotStopButtonHtml",
        "getBotDeleteButtonHtml", "getBotControlButtonsHtml", "getBotDetailButtonsHtml",
        "updateBotStatus", "updateBotControlButtons", "updateCoinsListWithBotStatus",
        "updateActiveBotsTab", "renderActiveBotsDetails", "loadActiveBotsData",
        "updateActiveBotsDetailed"
    ],
    "08_filter_ui": [
        "loadFiltersData", "renderFilters", "renderWhitelist", "renderBlacklist",
        "initializeFilterControls", "addToWhitelist", "addToBlacklist",
        "removeFromWhitelist", "removeFromBlacklist",
        "clearWhitelist", "clearBlacklist", "exportFiltersToJson", "importFiltersFromJson",
        "updateFilters", "validateCoinSymbol", "translate", "showNotification",
        "showFilterControls", "updateFilterStatus", "addSelectedCoinToWhitelist",
        "addSelectedCoinToBlacklist", "removeSelectedCoinFromFilters",
        "updateSmartFilterControls", "getFoundCoins", "addFoundCoinsToWhitelist",
        "addFoundCoinsToBlacklist", "performFiltersSearch", "searchCoins",
        "renderSearchResults", "addCoinToWhitelistFromSearch", "addCoinToBlacklistFromSearch",
        "removeCoinFromFiltersFromSearch", "highlightStatus", "highlightFilterStatus"
    ],
    "09_periodic": [
        "updateBotsSummaryStats", "startPeriodicUpdate", "startBotMonitoring",
        "stopBotMonitoring", "updateBotsDetailedDisplay", "updateSingleBotDisplay",
        "calculateTimeLeft", "destroy"
    ],
    "10_configuration": [
        "initializeScopeButtons", "loadConfigurationData", "populateConfigurationForm",
        "showConfigurationLoading", "saveDefaultConfiguration", "camelToSnake",
        "mapElementIdToConfigKey", "collectConfigurationData", "collectFieldsFromElements",
        "saveBasicSettings", "_updateFullaiAdaptiveDependentFields", "loadFullaiAdaptiveConfig",
        "saveFullaiAdaptiveConfig", "saveSystemSettings", "saveTradingAndRsiExits",
        "saveRsiTimeFilter", "saveExitScamFilter", "saveEnhancedRsi", "saveProtectiveMechanisms",
        "saveMaturitySettings", "saveEmaParameters", "saveTrendParameters",
        "hasUnsavedConfigChanges", "createFloatingSaveButton", "saveAllConfiguration",
        "hideFloatingSaveButton", "updateFloatingSaveButtonVisibility", "filterChangedParams",
        "sendConfigUpdate", "saveConfiguration", "resetConfiguration", "exportConfig",
        "importConfig", "testConfiguration", "syncDuplicateSettings", "loadDuplicateSettings",
        "initializeGlobalAutoBotToggle", "initializeMobileAutoBotToggle", "loadAccountInfo",
        "updateAccountDisplay", "updateBulkControlsVisibility", "initializeBulkControls",
        "applyConfigViewMode", "_initConfigViewSwitcher", "initializeConfigurationButtons",
        "initializeAutoSave", "saveSingleToggleToBackend", "scheduleToggleAutoSave",
        "addAutoSaveHandlers", "addStepperButtons", "scheduleAutoSave", "reloadModules",
        "startAllBots", "stopAllBots", "deleteAllBots", "showConfigNotification",
        "detectConfigChanges"
    ],
    "11_trades_display": [
        "getCompactCardData", "getBotPositionInfo", "getBotTimeInfo", "renderTradesInfo",
        "getBotTrades", "renderTradeItem", "initializeManualPositionsControls",
        "initializeRSILoadingButtons"
    ],
    "12_history": [
        "initializeHistoryTab", "initializeAnalyticsTab", "loadFullaiAnalytics",
        "renderFullaiAnalytics", "runRsiAudit", "renderRsiAuditReport", "syncTradesFromExchange",
        "runAiReanalyze", "runTradingAnalytics", "renderAnalyticsReport",
        "initializeHistoryFilters", "initializeHistorySubTabs", "initializeHistoryActionButtons",
        "loadHistoryData", "getHistoryFilters", "loadAIHistory", "loadAIStats"
    ],
    "13_ai_training": [
        "initAIPeriodSelector", "loadAIDecisions", "loadAIOptimizerSummary",
        "displayAIOptimizerSummary", "loadAITrainingHistory", "displayAITrainingHistory",
        "getAITrainingStatusMeta", "getAITrainingEventLabel", "updateAITrainingSummary",
        "loadAIPerformanceMetrics", "displayAIPerformanceMetrics", "buildAIComparisonSummary",
        "displayAIDecisions", "loadBotActions", "loadBotTrades", "loadBotSignals",
        "loadHistoryStatistics"
    ],
    "14_history_display": [
        "displayBotActions", "displayBotTrades", "displayBotSignals",
        "displayHistoryStatistics", "updateHistoryBotFilterOptions", "clearHistoryFilters",
        "exportHistoryData", "createDemoHistoryData", "clearAllHistory",
        "getActionIcon", "formatTimestamp", "formatDuration",
        "saveCollapseState", "preserveCollapseState"
    ],
    "15_limit_orders": [
        "initializeLimitOrdersUI", "addLimitOrderRow", "saveLimitOrdersSettings",
        "resetLimitOrdersToDefault"
    ],
    "16_timeframe": [
        "loadTimeframe", "applyTimeframe", "updateTimeframeInUI", "getTextNodes",
        "initTimeframeControls"
    ],
}

# Методы которые нужно найти - возможно createBot, startBot, stopBot и др. не в списке выше
# Проверим - есть ли ещё методы в файле
ALL_KNOWN = set()
for methods in MODULES.values():
    ALL_KNOWN.update(methods)


# Ключевые слова и вызовы - не методы класса
RESERVED = {
    'if', 'for', 'while', 'switch', 'catch', 'with', 'return',
    'fetch', 'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval',
    'alert', 'applyStyles', 'applyValue', 'setValue', 'applyFullAiControl',
    'syncFullAiToggles', 'applyToGeneralSettings', 'setupAddButtonHandler',
    'updateUIState', 'scheduleFullaiAdaptiveSave', 'resetToGeneralSettings',
}


def find_method_boundaries(content: str):
    """Находит границы всех методов класса BotsManager."""
    # Ищем определения методов: "    methodName(...) {" или "    async methodName(...) {"
    pattern = re.compile(
        r'^ {4,}(?:async\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{',
        re.MULTILINE
    )
    matches = list(pattern.finditer(content))
    
    boundaries = []  # (method_name, start_pos, end_pos) - end исключительно
    for i, m in enumerate(matches):
        name = m.group(1)
        if name in RESERVED:
            continue
        start = m.start()
        if i + 1 < len(matches):
            end = matches[i + 1].start() - 1
        else:
            # Последний метод - до закрывающей скобки класса
            end = content.rfind("\n}", 0, len(content))
            if end == -1:
                end = len(content)
        boundaries.append((name, start, end))
    
    return boundaries


def extract_method_body(content: str, start: int, end: int) -> str:
    """Извлекает тело метода с отступом 4 пробела (для Object.assign)."""
    return content[start:end].rstrip()


def main():
    content = SRC.read_text(encoding="utf-8")
    boundaries = find_method_boundaries(content)
    
    # Словарь: имя метода -> (start, end)
    method_map = {name: (start, end) for name, start, end in boundaries}
    
    found = set(method_map.keys())
    missing = ALL_KNOWN - found
    extra = found - ALL_KNOWN
    
    if missing:
        print(f"[!] Методы в плане, но не найдены: {missing}")
    if extra:
        print(f"[i] Методы в файле, но не в плане: {extra}")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for module_name, method_list in MODULES.items():
        methods_code = []
        for method_name in method_list:
            if method_name not in method_map:
                print(f"[!] Пропуск {method_name} - не найден")
                continue
            start, end = method_map[method_name]
            body = extract_method_body(content, start, end)
            methods_code.append(body)
        
        if not methods_code:
            continue
            
        if module_name == "00_core":
            # Core: определяем класс с constructor и базовыми методами
            class_body = "\n\n".join(methods_code)
            out = f'''/**
 * BotsManager - ядро (constructor, init, логирование)
 */
class BotsManager {{
    {class_body}
}}
'''
        else:
            # Остальные модули - Object.assign
            joined = ",\n        ".join(methods_code)
            out = f'''/**
 * BotsManager - {module_name}
 */
(function() {{
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {{
        {joined}
    }});
}})();
'''
        
        out_path = OUT_DIR / f"{module_name}.js"
        out_path.write_text(out, encoding="utf-8")
        print(f"[OK] {out_path.name}: {len(method_list)} методов")
    
    # Index.js - экспорт и глобальные функции
    tail = content[content.find("// Экспортируем класс"):]
    index_content = f'''/**
 * BotsManager - экспорт и глобальные функции
 */
{tail}
'''
    (OUT_DIR / "index.js").write_text(index_content, encoding="utf-8")
    print("[OK] index.js: экспорт")
    
    print("\nГотово. Обновите index.html для загрузки модулей.")


if __name__ == "__main__":
    main()
