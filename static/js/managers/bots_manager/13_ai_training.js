/**
 * BotsManager - 13_ai_training
 */
(function() {
    if (typeof BotsManager === 'undefined') return;
    Object.assign(BotsManager.prototype, {
            initAIPeriodSelector() {
        const select = document.getElementById('aiPeriodSelect');
        if (!select || select._aiBound) return;
        select._aiBound = true;
        select.addEventListener('change', () => {
            this.loadAIHistory();
        });
    }
    
    /**
     * Загружает решения AI
     */,
            async loadAIDecisions() {
        try {
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/ai/decisions?limit=100`);
            const data = await response.json();,
            async loadAIOptimizerSummary() {
        const paramsContainer = document.getElementById('optimizerParamsList');,
            displayAIOptimizerSummary(data) {
        const paramsList = document.getElementById('optimizerParamsList');
        const topList = document.getElementById('optimizerTopSymbols');
        const patternsContainer = document.getElementById('optimizerPatternsSummary');
        const genomeVersionEl = document.getElementById('optimizerGenomeVersion');
        const updatedAtEl = document.getElementById('optimizerUpdatedAt');
        const maxTestsEl = document.getElementById('optimizerMaxTests');
        const symbolsCountEl = document.getElementById('optimizerSymbolsCount');

        const metadata = data?.metadata || {};,
            async loadAITrainingHistory() {
        const container = document.getElementById('aiTrainingHistoryList');,
            displayAITrainingHistory(history) {
        const container = document.getElementById('aiTrainingHistoryList');
        if (!container) return;,
            getAITrainingStatusMeta(status) {
        const normalized = (status || 'SUCCESS').toUpperCase();
        const meta = {
            'SUCCESS': { icon: '✅', className: 'success' },
            'FAILED': { icon: '❌', className: 'failed' },
            'SKIPPED': { icon: '⏸️', className: 'skipped' }
        };
        return meta[normalized] || meta.SUCCESS;
    },
            getAITrainingEventLabel(eventType) {,
            updateAITrainingSummary(record) {
        const timeEl = document.getElementById('aiLastTrainingTime');
        const durationEl = document.getElementById('aiLastTrainingDuration');
        const samplesEl = document.getElementById('aiLastTrainingSamples');,
            async loadAIPerformanceMetrics() {
        try {
            const periodSelect = document.getElementById('aiPeriodSelect');
            const rawPeriod = periodSelect ? (periodSelect.value || '7d') : '7d';
            const periodMap = { '24h': 'today', '7d': 'week', '30d': 'month', 'all': 'all' };
            const period = periodMap[rawPeriod] || 'all';
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/ai/performance?period=${encodeURIComponent(period)}`);
            const data = await response.json();,
            displayAIPerformanceMetrics(metrics) {
        const winRateEl = document.getElementById('aiOverallWinRate');
        const pnlEl = document.getElementById('aiOverallPnL');
        const decisionsEl = document.getElementById('aiOverallDecisions');
        const topSymbolsEl = document.getElementById('aiTopSymbols');

        let overall = metrics?.overall || {};
        
        // Если метрики пустые, используем данные из статистики как fallback
        if ((!overall.total_ai_decisions || overall.total_ai_decisions === 0) && this._lastAIStats) {
            const stats = this._lastAIStats;,
            buildAIComparisonSummary(aiStats = {}, scriptStats = {}, comparison = {}) {
        const aiTotal = aiStats.total || 0;
        const scriptTotal = scriptStats.total || 0;,
            displayAIDecisions(decisions) {
        const container = document.getElementById('aiDecisionsList');
        if (!container) return;,
            async loadBotActions(filters) {
        try {
            const params = new URLSearchParams();
            if (filters.symbol && filters.symbol !== 'all') params.append('symbol', filters.symbol);
            if (filters.action_type && filters.action_type !== 'all') params.append('action_type', filters.action_type);
            if (filters.period && filters.period !== 'all') params.append('period', filters.period);
            params.append('limit', filters.limit);
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/history?${params}`);
            const data = await response.json();,
            async loadBotTrades(filters) {
        try {
            const params = new URLSearchParams();
            if (filters.symbol && filters.symbol !== 'all') params.append('symbol', filters.symbol);
            if (filters.trade_type && filters.trade_type !== 'all') params.append('trade_type', filters.trade_type);
            if (filters.period && filters.period !== 'all') params.append('period', filters.period);
            params.append('limit', filters.limit);
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/trades?${params}`);
            const data = await response.json();,
            async loadBotSignals(filters) {
        try {
            const params = new URLSearchParams();
            if (filters.symbol && filters.symbol !== 'all') params.append('symbol', filters.symbol);
            params.append('action_type', 'SIGNAL');
            if (filters.period && filters.period !== 'all') params.append('period', filters.period);
            params.append('limit', filters.limit);
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/history?${params}`);
            const data = await response.json();,
            async loadHistoryStatistics(filters = {}) {
        try {
            const params = new URLSearchParams();
            const symbol = filters?.symbol;
            const period = filters?.period;

            if (symbol && symbol !== 'all') params.append('symbol', symbol);
            if (period && period !== 'all') params.append('period', period);
            
            const response = await fetch(`${this.BOTS_SERVICE_URL}/api/bots/statistics?${params}`);
            const data = await response.json();
    });
})();
