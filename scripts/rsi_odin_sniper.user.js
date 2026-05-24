// ==UserScript==
// @name         RSI Ship Sniper — Odin
// @namespace    https://robertsspaceindustries.com/
// @version      1.2.0
// @description  Быстро ловит Add to cart только на странице Odin; останавливается при уходе в корзину
// @author       InfoBot
// @match        *://robertsspaceindustries.com/*Standalone-Ships/Odin*
// @match        *://*.robertsspaceindustries.com/*Standalone-Ships/Odin*
// @run-at       document-end
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM_notification
// ==/UserScript==

(function () {
    'use strict';

    console.info('[RSI Sniper v1.2.0] скрипт загружен:', location.href);

    const CONFIG = {
        targetPathPart: '/Standalone-Ships/Odin',

        // === Страна и валюта (влияют на цену!) ===
        targetCountry: 'Belarus',
        targetCurrencyLabel: 'USD', // кнопка "USD / en" в шапке
        expectedPriceContains: '5,900.00', // не жмём Add to cart, пока цена другая

        reloadIntervalMs: 500,
        pollIntervalMs: 50,
        urlWatchIntervalMs: 100,
        countrySettleMs: 600,

        goToCartOnSuccess: true,

        stopAfterSuccess: true,
        playSoundOnSuccess: true,

        buttonTexts: [
            'add to cart',
            'add to basket',
            'в корзину',
        ],
        unavailableTexts: [
            'out of stock',
            'sold out',
            'unavailable',
            'нет в наличии',
            'распродано',
        ],
    };

    const STORAGE_SUCCESS = 'rsi_sniper_success_' + CONFIG.targetPathPart;
    const STORAGE_ACTIVE = 'rsi_sniper_active_' + CONFIG.targetPathPart;
    const SESSION_ATTEMPTS = 'rsi_sniper_attempts_' + CONFIG.targetPathPart;

    function isTargetPage(loc = location) {
        return loc.pathname.includes(CONFIG.targetPathPart);
    }

    function isCartOrCheckoutPage(loc = location) {
        const path = loc.pathname.toLowerCase();
        return path.includes('/cart') || path.includes('/checkout');
    }

    function getCartUrl() {
        const lang = location.pathname.match(/^\/([a-z]{2})\//);
        if (lang) {
            return `https://robertsspaceindustries.com/${lang[1]}/store/pledge/cart`;
        }
        return 'https://robertsspaceindustries.com/en/store/pledge/cart';
    }

    // Не целевая страница — ничего не делаем (на всякий случай при широком @match)
    if (!isTargetPage()) {
        console.info('[RSI Sniper] не страница Odin, выход');
        GM_setValue(STORAGE_ACTIVE, false);
        return;
    }

    if (CONFIG.stopAfterSuccess && GM_getValue(STORAGE_SUCCESS, false)) {
        console.info('[RSI Sniper] уже сработал ранее — rsiSniperReset() для сброса');
        const banner = document.createElement('div');
        banner.textContent = 'RSI Sniper: уже сработал. F12 → rsiSniperReset()';
        banner.style.cssText = 'position:fixed;top:12px;right:12px;z-index:2147483647;padding:8px 12px;background:#3ddc84;color:#000;font:600 12px sans-serif;border-radius:6px';
        document.documentElement.appendChild(banner);
        window.rsiSniperReset = () => {
            GM_setValue(STORAGE_SUCCESS, false);
            GM_setValue(STORAGE_ACTIVE, false);
            sessionStorage.removeItem(SESSION_ATTEMPTS);
            location.reload();
        };
        return;
    }

    let stopped = false;
    let snipeLocked = false;
    let attempts = parseInt(sessionStorage.getItem(SESSION_ATTEMPTS) || '0', 10);
    let lastReloadAt = 0;
    let pollTimer = null;
    let reloadTimer = null;
    let urlWatchTimer = null;
    let observer = null;
    let hudEl = null;
    let watchedUrl = location.pathname + location.search + location.hash;
    let countryChangeAt = 0;
    let countryDropdownOpen = false;

    GM_setValue(STORAGE_ACTIVE, true);

    function normalize(text) {
        return (text || '').replace(/\s+/g, ' ').trim().toLowerCase();
    }

    function isUnavailablePage() {
        const bodyText = normalize(document.body ? document.body.innerText : '');
        return CONFIG.unavailableTexts.some((t) => bodyText.includes(t));
    }

    function isVisible(el) {
        if (!el || !el.isConnected) return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
            return false;
        }
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    }

    function isDisabled(el) {
        if (!el) return true;
        if (el.disabled) return true;
        if (el.getAttribute('aria-disabled') === 'true') return true;
        if (el.classList.contains('disabled')) return true;
        if (el.hasAttribute('disabled')) return true;
        const style = window.getComputedStyle(el);
        if (style.pointerEvents === 'none') return true;
        if (parseFloat(style.opacity) < 0.5) return true;
        return false;
    }

    function matchesButtonText(el) {
        const text = normalize(el.innerText || el.textContent || el.getAttribute('aria-label') || '');
        return CONFIG.buttonTexts.some((needle) => text.includes(needle));
    }

    function findCountryButton() {
        const blocks = document.querySelectorAll('*');
        for (const el of blocks) {
            const text = el.childNodes.length === 1 && el.childNodes[0].nodeType === 3
                ? el.textContent
                : '';
            if (text && text.trim() === 'Current country selected:') {
                const root = el.closest('div') || el.parentElement;
                if (root) {
                    const btn = root.querySelector('button');
                    if (btn) return btn;
                }
            }
        }

        for (const btn of document.querySelectorAll('button')) {
            const label = (btn.textContent || '').trim();
            if (!label || label.length > 50) continue;
            const lower = normalize(label);
            if (lower.includes('usd') || lower.includes('add to cart') || lower.includes('prev')
                || lower.includes('next') || lower.includes('all products') || lower.includes('ships')
                || lower.includes('gear') || lower.includes('subscriber') || lower.includes('lifetime')
                || lower === 'odin' || lower.includes('download')) {
                continue;
            }
            const footer = btn.closest('footer');
            if (footer) return btn;
        }

        return null;
    }

    function findCurrencyButton() {
        for (const btn of document.querySelectorAll('button')) {
            const label = btn.textContent || '';
            if (label.includes('/') && normalize(label).includes('usd')) return btn;
            if (normalize(label).startsWith('usd')) return btn;
        }
        return null;
    }

    function isCountryCorrect() {
        if (!CONFIG.targetCountry) return true;
        const btn = findCountryButton();
        if (!btn) return true;
        const current = normalize(btn.textContent);
        const target = normalize(CONFIG.targetCountry);
        return current.includes(target) || target.includes(current);
    }

    function isCurrencyCorrect() {
        if (!CONFIG.targetCurrencyLabel) return true;
        const btn = findCurrencyButton();
        if (!btn) return true;
        return normalize(btn.textContent).includes(normalize(CONFIG.targetCurrencyLabel));
    }

    function isPriceCorrect() {
        if (!CONFIG.expectedPriceContains) return true;
        return (document.body?.innerText || '').includes(CONFIG.expectedPriceContains);
    }

    function pickCountryOption(countryName) {
        const target = normalize(countryName);
        const selectors = '[role="option"], [role="menuitem"], [role="menuitemradio"], li, div[class*="option"]';

        for (const opt of document.querySelectorAll(selectors)) {
            const text = normalize(opt.textContent);
            if (!text || text.length > 80) continue;
            if (text === target || text.includes(target)) {
                forceClick(opt);
                return true;
            }
        }

        for (const inp of document.querySelectorAll('input[type="search"], input[type="text"]')) {
            if (!isVisible(inp)) continue;
            inp.focus();
            inp.value = countryName;
            inp.dispatchEvent(new Event('input', { bubbles: true }));
            inp.dispatchEvent(new Event('change', { bubbles: true }));
            break;
        }

        for (const opt of document.querySelectorAll(selectors)) {
            const text = normalize(opt.textContent);
            if (text.includes(target)) {
                forceClick(opt);
                return true;
            }
        }

        return false;
    }

    /** Возвращает true, если страна/валюта/цена готовы к покупке */
    function ensurePricingReady() {
        if (countryChangeAt && Date.now() - countryChangeAt < CONFIG.countrySettleMs) {
            showHud(`Ждём обновление цены… (${CONFIG.targetCountry})`, 'warn');
            return false;
        }

        if (!isCurrencyCorrect()) {
            const curBtn = findCurrencyButton();
            if (curBtn) {
                forceClick(curBtn);
                showHud('Переключаем валюту на USD…', 'warn');
                countryChangeAt = Date.now();
                return false;
            }
        }

        if (!isCountryCorrect()) {
            const countryBtn = findCountryButton();
            if (!countryBtn) {
                showHud('Не найден выбор страны!', 'err');
                return false;
            }
            if (!countryDropdownOpen) {
                forceClick(countryBtn);
                countryDropdownOpen = true;
                countryChangeAt = Date.now();
                showHud(`Открываем список → ${CONFIG.targetCountry}…`, 'warn');
                console.info('[RSI Sniper] открываем выбор страны');
                return false;
            }
            if (pickCountryOption(CONFIG.targetCountry)) {
                countryDropdownOpen = false;
                countryChangeAt = Date.now();
                showHud(`Страна: ${CONFIG.targetCountry}`, 'info');
            }
            return false;
        }

        countryDropdownOpen = false;

        if (!isPriceCorrect()) {
            showHud(`Ждём цену $${CONFIG.expectedPriceContains}…`, 'warn');
            return false;
        }

        return true;
    }

    function findAddToCartButton() {
        const candidates = [];

        document.querySelectorAll('button, a[role="button"], [role="button"]').forEach((el) => {
            if (matchesButtonText(el)) candidates.push(el);
        });

        document.querySelectorAll('input[type="submit"], input[type="button"]').forEach((el) => {
            const val = normalize(el.value || '');
            if (CONFIG.buttonTexts.some((t) => val.includes(t))) candidates.push(el);
        });

        for (const el of candidates) {
            if (isVisible(el) && !isDisabled(el)) return el;
        }

        return null;
    }

    function forceClick(el) {
        el.scrollIntoView({ block: 'center', inline: 'center' });
        el.focus({ preventScroll: true });

        for (const type of ['pointerdown', 'mousedown', 'mouseup', 'click']) {
            el.dispatchEvent(new MouseEvent(type, {
                bubbles: true,
                cancelable: true,
                view: window,
            }));
        }

        if (typeof el.click === 'function') el.click();
    }

    function playSuccessSound() {
        if (!CONFIG.playSoundOnSuccess) return;
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'square';
            osc.frequency.value = 880;
            gain.gain.value = 0.08;
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            setTimeout(() => {
                osc.stop();
                ctx.close();
            }, 400);
        } catch (e) {
            // ignore
        }
    }

    function notifySuccess() {
        try {
            GM_notification({
                title: 'RSI Sniper',
                text: 'Odin в корзине. Оформляй checkout!',
                timeout: 10000,
            });
        } catch (e) {
            // ignore
        }
    }

    function removeHud() {
        if (hudEl && hudEl.parentNode) {
            hudEl.parentNode.removeChild(hudEl);
        }
        hudEl = null;
    }

    function showHud(message, level) {
        if (!hudEl) {
            hudEl = document.createElement('div');
            hudEl.id = 'rsi-sniper-hud';
            hudEl.style.cssText = [
                'position:fixed',
                'top:12px',
                'right:12px',
                'z-index:2147483647',
                'padding:10px 14px',
                'border-radius:8px',
                'font:600 13px/1.35 Segoe UI,system-ui,sans-serif',
                'color:#fff',
                'background:rgba(10,20,40,.92)',
                'border:1px solid rgba(80,160,255,.55)',
                'box-shadow:0 8px 24px rgba(0,0,0,.45)',
                'max-width:320px',
                'pointer-events:none',
            ].join(';');
            document.documentElement.appendChild(hudEl);
        }

        const colors = { info: '#58a6ff', warn: '#f0ad4e', ok: '#3ddc84', err: '#ff6b6b' };
        hudEl.style.borderColor = colors[level] || colors.info;
        hudEl.textContent = message;
    }

    function stopSniper(reason) {
        if (stopped) return;
        stopped = true;
        snipeLocked = true;

        GM_setValue(STORAGE_ACTIVE, false);

        if (pollTimer !== null) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
        if (reloadTimer !== null) {
            clearInterval(reloadTimer);
            reloadTimer = null;
        }
        if (urlWatchTimer !== null) {
            clearInterval(urlWatchTimer);
            urlWatchTimer = null;
        }
        if (observer) {
            observer.disconnect();
            observer = null;
        }

        if (reason) {
            console.info('[RSI Sniper] Остановлен:', reason);
        }
    }

    /** Ушли с Odin или открыли корзину — больше не мешаем */
    function stopIfLeftTargetPage() {
        if (stopped) return true;

        const currentUrl = location.pathname + location.search + location.hash;

        if (!isTargetPage()) {
            const wentToCart = isCartOrCheckoutPage();
            stopSniper(wentToCart ? 'переход в корзину/checkout' : 'уход со страницы Odin');
            removeHud();

            if (wentToCart && CONFIG.stopAfterSuccess) {
                GM_setValue(STORAGE_SUCCESS, true);
            }
            return true;
        }

        watchedUrl = currentUrl;
        return false;
    }

    function markSuccess() {
        stopSniper('add to cart');
        GM_setValue(STORAGE_SUCCESS, true);
        sessionStorage.removeItem(SESSION_ATTEMPTS);
        playSuccessSound();
        notifySuccess();
        showHud('В корзину! Скрипт остановлен.', 'ok');

        if (CONFIG.goToCartOnSuccess) {
            location.replace(getCartUrl());
        }
    }

    function trySnipe() {
        if (stopped || snipeLocked) return false;
        if (stopIfLeftTargetPage()) return false;
        if (!ensurePricingReady()) return false;

        attempts += 1;
        sessionStorage.setItem(SESSION_ATTEMPTS, String(attempts));
        const btn = findAddToCartButton();

        if (btn) {
            snipeLocked = true;
            forceClick(btn);
            markSuccess();
            return true;
        }

        const waiting = isUnavailablePage();
        showHud(
            waiting
                ? `Ожидание… #${attempts} (Out of Stock)`
                : `Поиск кнопки… #${attempts}`,
            waiting ? 'warn' : 'info'
        );
        return false;
    }

    function maybeReload() {
        if (stopped) return;
        if (stopIfLeftTargetPage()) return;

        const now = Date.now();
        if (now - lastReloadAt < CONFIG.reloadIntervalMs) return;
        lastReloadAt = now;
        location.reload();
    }

    pollTimer = setInterval(() => {
        if (!stopped) trySnipe();
    }, CONFIG.pollIntervalMs);

    urlWatchTimer = setInterval(() => {
        stopIfLeftTargetPage();
    }, CONFIG.urlWatchIntervalMs);

    observer = new MutationObserver(() => {
        if (!stopped) trySnipe();
    });

    if (document.body) {
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['disabled', 'aria-disabled', 'class'],
        });
    }

    reloadTimer = setInterval(() => {
        if (!stopped) maybeReload();
    }, Math.max(100, CONFIG.reloadIntervalMs));

    window.addEventListener('popstate', stopIfLeftTargetPage, true);
    window.addEventListener('hashchange', stopIfLeftTargetPage, true);
    window.addEventListener('beforeunload', () => stopSniper('закрытие вкладки'), true);

    document.addEventListener('click', (event) => {
        if (stopped) return;
        const link = event.target.closest('a[href]');
        if (!link) return;
        try {
            const href = link.href || '';
            if (href.includes('/cart') || href.includes('/checkout')) {
                stopSniper('клик по корзине');
                if (CONFIG.stopAfterSuccess) GM_setValue(STORAGE_SUCCESS, true);
                removeHud();
            }
        } catch (e) {
            // ignore
        }
    }, true);

    showHud(`Снайпер активен… попытка #${attempts}`, 'info');
    console.info('[RSI Sniper] ожидание Add to cart на Odin');
    trySnipe();

    window.rsiSniperStop = () => {
        stopSniper('вручную');
        removeHud();
        showHud('Остановлен вручную', 'info');
    };

    window.rsiSniperReset = () => {
        GM_setValue(STORAGE_SUCCESS, false);
        GM_setValue(STORAGE_ACTIVE, false);
        sessionStorage.removeItem(SESSION_ATTEMPTS);
        location.reload();
    };

    console.info('[RSI Sniper v1.2.0] rsiSniperStop() / rsiSniperReset()');
})();
