// ==UserScript==
// @name         RSI Ship Sniper
// @namespace    https://robertsspaceindustries.com/
// @version      1.10.0
// @description  GraphQL addMany + подтверждение корзины; Odin: Belarus + $5,900
// @author       InfoBot
// @match        *://robertsspaceindustries.com/*
// @match        *://*.robertsspaceindustries.com/*
// @run-at       document-idle
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM_notification
// @grant        GM_addStyle
// @grant        unsafeWindow
// ==/UserScript==

(function () {
  'use strict';

  const VERSION = '1.10.0';
  const ROOT = typeof unsafeWindow !== 'undefined' ? unsafeWindow : window;

  const cartNetworkState = {
    lastAddOk: false,
    lastAddError: null,
    lastAddAt: 0,
  };

  try {
    GM_addStyle(`
      #rsi-sniper-boot {
        position: fixed !important;
        top: 72px !important;
        right: 16px !important;
        z-index: 2147483647 !important;
        padding: 10px 14px !important;
        background: #3ddc84 !important;
        color: #000 !important;
        font: 700 13px/1.2 sans-serif !important;
        border-radius: 8px !important;
        box-shadow: 0 6px 20px rgba(0,0,0,.45) !important;
        pointer-events: none !important;
      }
    `);
  } catch (e) {
    // ignore
  }

  function showBootMarker() {
    if (document.getElementById('rsi-sniper-boot')) return;
    const el = document.createElement('div');
    el.id = 'rsi-sniper-boot';
    el.textContent = `RSI Sniper v${VERSION} OK`;
    (document.documentElement || document.body)?.appendChild(el);
    setTimeout(() => el.remove(), 12000);
  }

  console.error(`[RSI Sniper v${VERSION}] загружен (content-script):`, location.href);
  showBootMarker();

  const CONFIG = {
    shipPagePathPart: '/pledge/Standalone-Ships/',

    // Только для страницы Odin
    odinPathPart: '/Standalone-Ships/Odin',
    odinLabel: 'Odin',
    odinExpectedPrice: '5,900.00',

    targetCountry: 'Belarus',
    countryAliases: ['Belarus', 'Беларусь'],
    targetCurrencyLabel: 'USD',

    // Reload только если кнопка Add to cart найдена, но неактивна (не по таймеру «вслепую»)
    autoReloadEnabled: true,
    inactiveButtonReloadDelayMs: 2000,
    maxReloads: 3,
    pollIntervalMs: 300,
    pageReadyTimeoutMs: 45000,
    countrySettleMs: 2000,
    afterAddToCartDelayMs: 1500,
    cartAddedTimeoutMs: 20000,
    useGraphQLAddToCart: true,
    maxAddToCartRetries: 5,
    graphqlUrl: 'https://robertsspaceindustries.com/graphql',
    storeFront: 'pledge',
    stepDelayMs: 800,
    stepTimeoutMs: 12000,

    cartUrl: 'https://robertsspaceindustries.com/en/store/pledge/cart',
    couponCode: 'SRBQHQYZL8',
    goToCartOnSuccess: true,

    // Тренировка: дойти до Place order, но не нажимать
    trainingMode: true,
    clickPlaceOrder: false,

    // Оплата store credits: без disclaimer / Proceed to pay
    creditsOnlyCheckout: true,

    stopAfterSuccess: true,
    playSoundOnSuccess: true,

    buttonTexts: ['add to cart', 'add to basket', 'в корзину'],
    cartAddedTexts: [
      'successfully added to your cart',
      'item successfully added',
      'added to your cart',
    ],
    unavailableTexts: [
      'out of stock',
      'sold out',
      'unavailable',
      'нет в наличии',
      'распродано',
    ],
  };

  const SESSION_FLOW = 'rsi_sniper_flow_active';
  const SESSION_STEP = 'rsi_sniper_checkout_step';
  const SESSION_SHIP_KEY = 'rsi_sniper_ship_page_key';

  const STEPS = {
    CART: 'cart',
    PLACE_ORDER: 'place_order',
    ADDRESS: 'address',
    DISCLAIMER: 'disclaimer',
    AFTER_DISCLAIMER: 'after_disclaimer',
    DONE: 'done',
  };

  function normalize(text) {
    return (text || '').replace(/\s+/g, ' ').trim().toLowerCase();
  }

  function getShipPageKey(loc = location) {
    const path = typeof loc === 'string' ? loc : (loc.pathname || '');
    return path.toLowerCase().replace(/[^a-z0-9/-]+/g, '');
  }

  function storageKey(suffix, loc) {
    return `rsi_sniper_${suffix}_${getShipPageKey(loc || location)}`;
  }

  function isShipPledgePage(loc = location) {
    return loc.pathname.toLowerCase().includes(CONFIG.shipPagePathPart.toLowerCase());
  }

  function isOdinPage(loc = location) {
    return loc.pathname.toLowerCase().includes(CONFIG.odinPathPart.toLowerCase());
  }

  function getShipNameFromPath(loc = location) {
    const m = loc.pathname.match(/Standalone-Ships\/([^/?#]+)/i);
    if (!m) return 'Ship';
    return m[1].replace(/-/g, ' ');
  }

  function getShipProfile(loc = location) {
    const odin = isOdinPage(loc);
    return {
      isOdin: odin,
      label: odin ? CONFIG.odinLabel : getShipNameFromPath(loc),
      requireRegion: odin,
      requirePrice: odin,
      expectedPrice: odin ? CONFIG.odinExpectedPrice : null,
    };
  }

  function isCheckoutPage(loc = location) {
    return loc.pathname.toLowerCase().includes('/store/pledge/cart');
  }

  function getCartUrl() {
    return CONFIG.cartUrl;
  }

  function isFlowActive() {
    return sessionStorage.getItem(SESSION_FLOW) === '1';
  }

  function startFlow() {
    sessionStorage.setItem(SESSION_FLOW, '1');
    sessionStorage.setItem(SESSION_STEP, STEPS.CART);
    sessionStorage.setItem(SESSION_SHIP_KEY, getShipPageKey());
  }

  function getStep() {
    return sessionStorage.getItem(SESSION_STEP) || STEPS.CART;
  }

  function setStep(step) {
    sessionStorage.setItem(SESSION_STEP, step);
  }

  function clearFlow() {
    const shipKey = sessionStorage.getItem(SESSION_SHIP_KEY);
    sessionStorage.removeItem(SESSION_FLOW);
    sessionStorage.removeItem(SESSION_STEP);
    sessionStorage.removeItem(SESSION_SHIP_KEY);
    if (shipKey) sessionStorage.removeItem(`rsi_sniper_attempts_${shipKey}`);
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
    if (el.hasAttribute('disabled')) return true;
    if (el.getAttribute('aria-disabled') === 'true') return true;
    if (el.classList.contains('disabled') || el.classList.contains('is-disabled')) return true;
    if (el.closest('fieldset[disabled]')) return true;
    const style = window.getComputedStyle(el);
    if (style.pointerEvents === 'none') return true;
    if (style.cursor === 'not-allowed') return true;
    if (parseFloat(style.opacity) < 0.5) return true;
    return false;
  }

  function getCookie(name) {
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const m = document.cookie.match(new RegExp(`(?:^|; )${escaped}=([^;]*)`));
    return m ? decodeURIComponent(m[1]) : '';
  }

  function parseSkuSlugFromPage() {
    const comp = document.querySelector('[data-rsi-component="SkuDetailPage"]');
    if (comp) {
      const props = comp.getAttribute('data-rsi-component-props');
      if (props) {
        try {
          const parsed = JSON.parse(props);
          if (parsed.skuSlug) return parsed.skuSlug;
        } catch (e) {
          const m = props.match(/"skuSlug"\s*:\s*"([^"]+)"/);
          if (m) return m[1];
        }
      }
    }
    const html = document.documentElement?.innerHTML || '';
    const m2 = html.match(/"skuSlug"\s*:\s*"([^"]+)"/);
    return m2 ? m2[1] : null;
  }

  function getHeaderCartCount() {
    for (const el of document.querySelectorAll(
      'a[href*="/cart"], a[href*="Cart"], [class*="mini-cart"], [class*="MiniCart"], [aria-label*="cart"], [aria-label*="Cart"]',
    )) {
      const badge = el.querySelector('[class*="count"], [class*="badge"], span');
      const raw = (badge?.textContent || el.textContent || '').trim();
      const m = raw.match(/\b(\d{1,2})\b/);
      if (m) {
        const n = parseInt(m[1], 10);
        if (!Number.isNaN(n) && n >= 0 && n < 100) return n;
      }
    }
    return 0;
  }

  function isCartPageEmpty() {
    const text = normalize(document.body?.innerText || '');
    if (
      text.includes('your cart is empty')
      || text.includes('shopping cart is empty')
      || text.includes('корзина пуста')
    ) return true;
    if (text.includes('order summary') || text.includes('add a coupon')) return false;
    return getHeaderCartCount() === 0;
  }

  const ADD_CART_MULTI_MUTATION = `mutation AddCartMultiItemMutation($query: [CartAddInput!], $storeFront: String = "pledge") {
  store(name: $storeFront) {
    cart {
      mutations {
        addMany(query: $query) {
          count
        }
      }
    }
  }
}`;

  function markCartAddSuccess() {
    cartNetworkState.lastAddOk = true;
    cartNetworkState.lastAddAt = Date.now();
    cartNetworkState.lastAddError = null;
    try {
      ROOT.document?.dispatchEvent(new CustomEvent('rsi-store.updateminicart'));
    } catch (e) { /* ignore */ }
  }

  function inspectGraphQLCartResponse(text) {
    if (!text || !/addMany|AddCartMulti/i.test(text)) return;
    try {
      const json = JSON.parse(text);
      if (json.errors?.length) {
        cartNetworkState.lastAddOk = false;
        cartNetworkState.lastAddError = json.errors[0]?.message || 'GraphQL error';
        return;
      }
      const addMany = json.data?.store?.cart?.mutations?.addMany;
      if (addMany && (addMany.count > 0 || (addMany.resources && addMany.resources.length))) {
        markCartAddSuccess();
      }
    } catch (e) { /* ignore */ }
  }

  function installCartNetworkMonitor() {
    if (ROOT.__rsiSniperNetPatched) return;
    ROOT.__rsiSniperNetPatched = true;

    const origFetch = ROOT.fetch?.bind(ROOT);
    if (origFetch) {
      ROOT.fetch = async function rsiSniperFetch(...args) {
        const res = await origFetch(...args);
        try {
          const url = typeof args[0] === 'string' ? args[0] : args[0]?.url;
          if (url && String(url).includes('/graphql')) {
            inspectGraphQLCartResponse(await res.clone().text());
          }
        } catch (e) { /* ignore */ }
        return res;
      };
    }

    const XHR = ROOT.XMLHttpRequest;
    if (XHR?.prototype) {
      const origOpen = XHR.prototype.open;
      const origSend = XHR.prototype.send;
      XHR.prototype.open = function rsiSniperXhrOpen(method, url, ...rest) {
        this.__rsiSniperUrl = url;
        return origOpen.call(this, method, url, ...rest);
      };
      XHR.prototype.send = function rsiSniperXhrSend(...sendArgs) {
        if (String(this.__rsiSniperUrl || '').includes('/graphql')) {
          this.addEventListener('load', function rsiSniperXhrLoad() {
            inspectGraphQLCartResponse(this.responseText);
          });
        }
        return origSend.apply(this, sendArgs);
      };
    }
  }

  async function addToCartViaGraphQL(skuId, qty = 1) {
    if (!skuId) return { ok: false, reason: 'no-sku' };
    const headers = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    };
    const rsiToken = getCookie('Rsi-Token') || getCookie('X-Rsi-Token');
    if (rsiToken) headers['X-Rsi-Token'] = rsiToken;

    const body = {
      operationName: 'AddCartMultiItemMutation',
      query: ADD_CART_MULTI_MUTATION,
      variables: {
        storeFront: CONFIG.storeFront,
        query: [{ skuId, qty }],
      },
    };

    try {
      const res = await ROOT.fetch(CONFIG.graphqlUrl, {
        method: 'POST',
        headers,
        credentials: 'include',
        body: JSON.stringify(body),
      });
      const json = await res.json();
      if (json.errors?.length) {
        cartNetworkState.lastAddError = json.errors[0]?.message || 'GraphQL error';
        console.warn('[RSI Sniper] GraphQL addMany:', json.errors);
        return { ok: false, errors: json.errors };
      }
      const addMany = json.data?.store?.cart?.mutations?.addMany;
      const ok = !!(addMany && (addMany.count > 0 || (addMany.resources && addMany.resources.length)));
      if (ok) markCartAddSuccess();
      return { ok, count: addMany?.count };
    } catch (err) {
      console.warn('[RSI Sniper] GraphQL fetch:', err);
      return { ok: false, error: err };
    }
  }

  function bodyHasExpectedPrice(expectedPrice) {
    if (!expectedPrice) return true;
    const text = document.body?.innerText || '';
    if (text.includes(expectedPrice)) return true;
    const compact = expectedPrice.replace(/[,\s]/g, '');
    return compact.length > 0 && text.replace(/[,\s]/g, '').includes(compact);
  }

  function elementMatchesAddToCart(el) {
    const label = normalize([
      el.textContent,
      el.value,
      el.getAttribute('aria-label'),
      el.getAttribute('title'),
    ].filter(Boolean).join(' '));
    return CONFIG.buttonTexts.some((t) => label.includes(t));
  }

  function getEventView(el) {
    try {
      return el?.ownerDocument?.defaultView || ROOT;
    } catch (e) {
      return ROOT;
    }
  }

  function dispatchMouse(el, type, view) {
    const base = { bubbles: true, cancelable: true };
    try {
      el.dispatchEvent(new MouseEvent(type, { ...base, view }));
      return true;
    } catch (e) {
      try {
        el.dispatchEvent(new MouseEvent(type, base));
        return true;
      } catch (e2) {
        return false;
      }
    }
  }

  function reactPointerClick(el) {
    if (!el || !el.isConnected) return false;
    const rect = el.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;
    const view = getEventView(el);
    const base = {
      bubbles: true,
      cancelable: true,
      view,
      clientX: x,
      clientY: y,
      screenX: x,
      screenY: y,
      button: 0,
      buttons: 1,
      pointerId: 1,
      pointerType: 'mouse',
      isPrimary: true,
    };
    const types = [
      'pointerover', 'pointerenter', 'mouseover', 'mouseenter',
      'pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click',
    ];
    for (const type of types) {
      try {
        if (type.startsWith('pointer') && typeof PointerEvent === 'function') {
          el.dispatchEvent(new PointerEvent(type, base));
        } else {
          dispatchMouse(el, type, view);
        }
      } catch (e) { /* ignore */ }
    }
    return true;
  }

  function forceClick(el) {
    if (!el || !el.isConnected) return false;
    try {
      el.scrollIntoView({ block: 'center', inline: 'center' });
    } catch (e) { /* ignore */ }
    try {
      el.focus({ preventScroll: true });
    } catch (e) { /* ignore */ }

    if (typeof el.click === 'function') {
      try {
        el.click();
        return true;
      } catch (e) { /* fallback to synthetic events */ }
    }

    return reactPointerClick(el);
  }

  function setInputValue(input, value) {
    input.focus({ preventScroll: true });
    const proto = input.tagName === 'TEXTAREA'
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
    if (setter) setter.call(input, value);
    else input.value = value;
    input.dispatchEvent(new InputEvent('input', { bubbles: true, cancelable: true, data: value }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function findButtons() {
    const list = [];
    document.querySelectorAll('button, a[role="button"], [role="button"], input[type="submit"], input[type="button"]').forEach((el) => {
      if (!isVisible(el) || isDisabled(el)) return;
      const text = normalize(el.textContent || el.value || el.getAttribute('aria-label') || '');
      if (text) list.push({ el, text });
    });
    return list;
  }

  function findButton(matchFn) {
    for (const item of findButtons()) {
      if (matchFn(item.text, item.el)) return item.el;
    }
    return null;
  }

  function findBottomButton(matchFn) {
    const items = findButtons().filter((item) => matchFn(item.text, item.el));
    if (!items.length) return null;
    items.sort((a, b) => b.el.getBoundingClientRect().top - a.el.getBoundingClientRect().top);
    return items[0].el;
  }

  function findSectionRoot(labelNeedle) {
    const parts = normalize(labelNeedle).split('|').map((p) => p.trim()).filter(Boolean);
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const nodeText = normalize(walker.currentNode.textContent);
      if (!parts.some((part) => nodeText.includes(part))) continue;
      let root = walker.currentNode.parentElement;
      for (let depth = 0; depth < 12 && root; depth += 1) {
        if (root.querySelector('input, button')) return root;
        root = root.parentElement;
      }
    }
    return null;
  }

  let hudEl = null;

  function removeHud() {
    if (hudEl && hudEl.parentNode) hudEl.parentNode.removeChild(hudEl);
    hudEl = null;
  }

  function showHud(message, level) {
    if (!hudEl) {
      hudEl = document.createElement('div');
      hudEl.id = 'rsi-sniper-hud';
      hudEl.style.cssText = [
        'position:fixed', 'top:12px', 'right:12px', 'z-index:2147483647',
        'padding:10px 14px', 'border-radius:8px',
        'font:600 13px/1.35 Segoe UI,system-ui,sans-serif', 'color:#fff',
        'background:rgba(10,20,40,.92)', 'border:1px solid rgba(80,160,255,.55)',
        'box-shadow:0 8px 24px rgba(0,0,0,.45)', 'max-width:340px', 'pointer-events:none',
      ].join(';');
      document.documentElement.appendChild(hudEl);
    }
    const colors = { info: '#58a6ff', warn: '#f0ad4e', ok: '#3ddc84', err: '#ff6b6b' };
    hudEl.style.borderColor = colors[level] || colors.info;
    hudEl.textContent = message;
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
      setTimeout(() => { osc.stop(); ctx.close(); }, 400);
    } catch (e) { /* ignore */ }
  }

  function notifySuccess(text) {
    try {
      GM_notification({ title: 'RSI Sniper', text: text || 'Заказ оформлен!', timeout: 12000 });
    } catch (e) { /* ignore */ }
  }

  function markFlowComplete(reason) {
    const shipKey = sessionStorage.getItem(SESSION_SHIP_KEY) || getShipPageKey();
    GM_setValue(storageKey('success', { pathname: shipKey }), true);
    GM_setValue(storageKey('active', { pathname: shipKey }), false);
    clearFlow();
    setStep(STEPS.DONE);
    playSuccessSound();
    notifySuccess(reason);
  }

  ROOT.rsiSniperReset = () => {
    GM_setValue(storageKey('success'), false);
    GM_setValue(storageKey('active'), false);
    clearFlow();
    location.reload();
  };

  ROOT.rsiSniperStop = () => {
    GM_setValue(storageKey('active'), false);
    clearFlow();
    removeHud();
    showHud('Остановлен вручную', 'info');
  };

  if (CONFIG.stopAfterSuccess && isShipPledgePage() && GM_getValue(storageKey('success'), false)) {
    console.info('[RSI Sniper] уже сработал — rsiSniperReset()');
    const banner = document.createElement('div');
    banner.textContent = 'RSI Sniper: уже сработал. F12 → rsiSniperReset()';
    banner.style.cssText = 'position:fixed;top:12px;right:12px;z-index:2147483647;padding:8px 12px;background:#3ddc84;color:#000;font:600 12px sans-serif;border-radius:6px';
    document.documentElement.appendChild(banner);
    return;
  }

  function boot() {
    installCartNetworkMonitor();

    if (isCheckoutPage() && isFlowActive()) {
      if (isCartPageEmpty()) {
        clearFlow();
        showHud('Корзина пуста — снайпер остановлен', 'err');
        console.warn('[RSI Sniper] checkout отменён: корзина пуста');
        return;
      }
      const shipKey = sessionStorage.getItem(SESSION_SHIP_KEY) || '';
      CONFIG.applyStoreCredits = !shipKey.toLowerCase().includes('warbond');
      runCheckoutFlow();
      return;
    }

    if (isCheckoutPage()) {
      console.info('[RSI Sniper] checkout без активного flow — выход');
      return;
    }

    if (!isShipPledgePage()) {
      console.info('[RSI Sniper] не страница корабля:', location.pathname);
      return;
    }

    CONFIG.applyStoreCredits = !location.pathname.toLowerCase().includes('warbond');
    const profile = getShipProfile();
    console.info('[RSI Sniper] корабль:', profile.label, profile.isOdin ? '(Belarus + цена)' : '(без проверки цены)');
    runShipFlow(profile);
  }

  // ===========================================================================
  // CHECKOUT: корзина → купон → MAX credits → Continue → Address → Place order
  // ===========================================================================

  function findOrderSummaryRoot() {
    return findSectionRoot('order summary');
  }

  function findCouponInput() {
    const scope = findOrderSummaryRoot() || document.body;
    for (const inp of scope.querySelectorAll('input')) {
      if (!isVisible(inp)) continue;
      const ph = normalize(inp.placeholder || '');
      const label = normalize(inp.getAttribute('aria-label') || '');
      if (ph.includes('coupon') || label.includes('coupon')) return inp;
    }
    const block = findSectionRoot('add a coupon');
    return block?.querySelector('input') || null;
  }

  function findCouponAddButton(input) {
    if (!input) return null;
    let root = input.parentElement;
    for (let depth = 0; depth < 8 && root; depth += 1) {
      for (const btn of root.querySelectorAll('button')) {
        if (normalize(btn.textContent) === 'add' && isVisible(btn) && !isDisabled(btn)) return btn;
      }
      root = root.parentElement;
    }
    return null;
  }

  function findStoreCreditsMaxButton() {
    const block = findSectionRoot('add store credits');
    if (!block) return null;
    for (const btn of block.querySelectorAll('button')) {
      if (normalize(btn.textContent) === 'max' && isVisible(btn) && !isDisabled(btn)) return btn;
    }
    return null;
  }

  function isCouponApplied() {
    const code = normalize(CONFIG.couponCode);
    if (normalize(document.body?.innerText || '').includes(code)) return true;
    const summary = findOrderSummaryRoot();
    if (!summary) return false;
    const text = normalize(summary.innerText || '');
    return text.includes('coupon') && !text.includes('coupon code (optional)');
  }

  function isStoreCreditApplied() {
    const text = normalize(document.body?.innerText || '');
    return text.includes('store credit used') || text.includes('store credits used');
  }

  function isZeroTotal() {
    const text = document.body?.innerText || '';
    return /\$0\.00\s*USD/i.test(text) || /\btotal\s*\n?\s*0\b/i.test(text);
  }

  function shouldSkipPaymentSteps() {
    return CONFIG.creditsOnlyCheckout && CONFIG.applyStoreCredits
      && (isStoreCreditApplied() || isZeroTotal());
  }

  function findPlaceOrderButton() {
    return findBottomButton((text) => text === 'place order');
  }

  function isDisclaimerOpen() {
    const text = document.body?.innerText || '';
    return text.includes('Disclaimer') && text.includes('I agree to the');
  }

  function findDisclaimerRoot() {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      if (normalize(walker.currentNode.textContent) !== 'disclaimer') continue;
      let root = walker.currentNode.parentElement;
      for (let depth = 0; depth < 10 && root; depth += 1) {
        if (root.querySelector('input[type="checkbox"], [role="checkbox"]')) return root;
        root = root.parentElement;
      }
    }
    return document.body;
  }

  function findDisclaimerCheckbox() {
    const root = findDisclaimerRoot();
    for (const cb of root.querySelectorAll('input[type="checkbox"], [role="checkbox"]')) {
      if (!isVisible(cb)) continue;
      const label = normalize(cb.closest('label')?.textContent || cb.parentElement?.textContent || '');
      if (label.includes('i agree')) return cb;
    }
    return root.querySelector('input[type="checkbox"]');
  }

  function isCheckboxChecked(cb) {
    if (!cb) return false;
    if (cb.type === 'checkbox') return cb.checked;
    return cb.getAttribute('aria-checked') === 'true';
  }

  function checkDisclaimerCheckbox(cb) {
    if (!cb || isCheckboxChecked(cb)) return;
    forceClick(cb);
    if (cb.type === 'checkbox') {
      cb.checked = true;
      cb.dispatchEvent(new Event('change', { bubbles: true }));
      cb.dispatchEvent(new Event('input', { bubbles: true }));
    }
  }

  function isCartStepVisible() {
    const text = normalize(document.body?.innerText || '');
    return text.includes('order summary') && (findCouponInput() || text.includes('add a coupon'));
  }

  function isAddressStepVisible() {
    const text = normalize(document.body?.innerText || '');
    return text.includes('billing information') || text.includes('2. address');
  }

  function runCheckoutFlow() {
    let stopped = false;
    let pollTimer = null;
    let observer = null;
    let lastActionAt = 0;
    let stepStartedAt = Date.now();

    const state = {
      couponEntered: false,
      couponAddClicked: false,
      maxClicked: false,
      cartContinueClicked: false,
      proceedClicked: false,
      disclaimerDone: false,
      postContinueClicked: false,
      placeOrderClicked: false,
    };

    function stopCheckout(reason) {
      if (stopped) return;
      stopped = true;
      if (pollTimer) clearInterval(pollTimer);
      if (observer) observer.disconnect();
      if (reason) console.info('[RSI Sniper] checkout:', reason);
    }

    function canAct() {
      return Date.now() - lastActionAt >= CONFIG.stepDelayMs;
    }

    function act(fn) {
      lastActionAt = Date.now();
      fn();
    }

    function finishAtPlaceOrder() {
      const msg = CONFIG.trainingMode
        ? `${CONFIG.shipLabel}: тренировка — Place order на экране`
        : `${CONFIG.shipLabel} — заказ оформлен!`;
      markFlowComplete(msg);
      showHud(
        CONFIG.trainingMode ? 'Тренировка OK! Place order — вручную' : 'Заказ отправлен!',
        'ok',
      );
      stopCheckout('done');
    }

    function handlePlaceOrderStep() {
      const placeBtn = findPlaceOrderButton();
      if (!placeBtn) return false;

      setStep(STEPS.PLACE_ORDER);

      if (CONFIG.trainingMode || !CONFIG.clickPlaceOrder) {
        if (!canAct()) return true;
        act(() => {
          console.info('[RSI Sniper] Place order найден — стоп (тренировка)');
          state.placeOrderClicked = false;
          finishAtPlaceOrder();
        });
        return true;
      }

      if (!canAct()) return true;
      if (!state.placeOrderClicked) {
        act(() => {
          console.info('[RSI Sniper] PLACE ORDER');
          forceClick(placeBtn);
          state.placeOrderClicked = true;
          finishAtPlaceOrder();
        });
      }
      return true;
    }

    function tryCheckoutStep() {
      if (stopped || !isFlowActive()) return;

      if (!isCheckoutPage()) {
        stopCheckout('уход со checkout');
        removeHud();
        return;
      }

      const step = getStep();
      const bodyText = normalize(document.body?.innerText || '');

      // После credits: сразу Place order (без disclaimer / address)
      if (state.cartContinueClicked && shouldSkipPaymentSteps()) {
        if (handlePlaceOrderStep()) return;
        showHud('Credits OK — ищем Place order…', 'info');
        return;
      }

      if (handlePlaceOrderStep()) return;

      // Ниже — только если оплата не полностью credits (warbond / карта)
      if (!shouldSkipPaymentSteps() && (isDisclaimerOpen() || step === STEPS.DISCLAIMER)) {
        setStep(STEPS.DISCLAIMER);
        showHud('Disclaimer: галочка → I AGREE', 'info');

        if (!canAct()) return;

        const checkbox = findDisclaimerCheckbox();
        if (checkbox && !isCheckboxChecked(checkbox)) {
          act(() => {
            console.info('[RSI Sniper] ставим галочку I agree');
            checkDisclaimerCheckbox(checkbox);
          });
          return;
        }

        const agreeBtn = findButton((text) => text === 'i agree');
        if (agreeBtn) {
          act(() => {
            console.info('[RSI Sniper] жмём I AGREE');
            forceClick(agreeBtn);
            state.disclaimerDone = true;
            setStep(STEPS.AFTER_DISCLAIMER);
            stepStartedAt = Date.now();
          });
          return;
        }

        showHud('Ждём I AGREE…', 'warn');
        return;
      }

      if (!shouldSkipPaymentSteps() && (step === STEPS.AFTER_DISCLAIMER || (state.disclaimerDone && isAddressStepVisible()))) {
        setStep(STEPS.AFTER_DISCLAIMER);
        showHud('Continue после disclaimer…', 'info');

        if (!canAct()) return;

        const continueBtn = findBottomButton((text) => text === 'continue');
        if (continueBtn && !state.postContinueClicked) {
          act(() => {
            console.info('[RSI Sniper] Continue после disclaimer');
            forceClick(continueBtn);
            state.postContinueClicked = true;
            setStep(STEPS.PLACE_ORDER);
            stepStartedAt = Date.now();
          });
          return;
        }

        if (!state.postContinueClicked && Date.now() - stepStartedAt > CONFIG.stepTimeoutMs) {
          setStep(STEPS.PLACE_ORDER);
        } else if (!state.postContinueClicked) {
          showHud('Ищем Continue…', 'warn');
          return;
        }
      }

      if (!shouldSkipPaymentSteps() && isAddressStepVisible() && step !== STEPS.CART) {
        setStep(STEPS.ADDRESS);
        showHud('Address: Proceed to pay…', 'info');

        if (!canAct()) return;

        const proceedBtn = findBottomButton((text) => text.includes('proceed to pay'));
        if (proceedBtn && !state.proceedClicked) {
          act(() => {
            console.info('[RSI Sniper] PROCEED TO PAY');
            forceClick(proceedBtn);
            state.proceedClicked = true;
            setStep(STEPS.DISCLAIMER);
            stepStartedAt = Date.now();
          });
          return;
        }

        if (!state.proceedClicked) {
          showHud('Ищем Proceed to pay…', 'warn');
          return;
        }
      }

      if (isCartStepVisible() || step === STEPS.CART) {
        setStep(STEPS.CART);

        if (!bodyText.includes('order summary')) {
          showHud('Ждём корзину…', 'warn');
          return;
        }

        if (!canAct()) return;

        if (!isCouponApplied()) {
          const couponInput = findCouponInput();
          if (!couponInput) {
            showHud('Ищем Coupon Code…', 'warn');
            return;
          }

          const val = (couponInput.value || '').trim();
          if (val !== CONFIG.couponCode && !state.couponEntered) {
            act(() => {
              console.info('[RSI Sniper] купон:', CONFIG.couponCode);
              setInputValue(couponInput, CONFIG.couponCode);
              state.couponEntered = true;
              showHud('Купон введён — ADD…', 'info');
            });
            return;
          }
          state.couponEntered = true;

          const addBtn = findCouponAddButton(couponInput);
          if (addBtn && !state.couponAddClicked) {
            act(() => {
              console.info('[RSI Sniper] ADD купон');
              forceClick(addBtn);
              state.couponAddClicked = true;
              stepStartedAt = Date.now();
            });
            return;
          }

          showHud('Ждём применение купона…', 'warn');
          return;
        }

        if (CONFIG.applyStoreCredits && !isStoreCreditApplied()) {
          const maxBtn = findStoreCreditsMaxButton();
          if (maxBtn && !state.maxClicked) {
            act(() => {
              console.info('[RSI Sniper] MAX store credits');
              forceClick(maxBtn);
              state.maxClicked = true;
              stepStartedAt = Date.now();
              showHud('Store credits MAX…', 'info');
            });
            return;
          }
          if (!state.maxClicked && Date.now() - stepStartedAt > CONFIG.stepTimeoutMs) {
            state.maxClicked = true;
          } else if (!state.maxClicked) {
            showHud('Ищем MAX store credits…', 'warn');
            return;
          }
        }

        const continueBtn = findBottomButton((text) => text === 'continue');
        if (continueBtn && !state.cartContinueClicked) {
          act(() => {
            console.info('[RSI Sniper] Continue (корзина)');
            forceClick(continueBtn);
            state.cartContinueClicked = true;
            setStep(STEPS.PLACE_ORDER);
            stepStartedAt = Date.now();
            showHud('Continue → Place order…', 'info');
          });
          return;
        }

        showHud('Ищем Continue…', 'warn');
      }
    }

    console.info('[RSI Sniper] checkout: купон → MAX → Continue → Place order', {
      storeCredits: CONFIG.applyStoreCredits,
      training: CONFIG.trainingMode,
    });
    showHud('Checkout: купон → MAX → Continue', 'info');
    pollTimer = setInterval(tryCheckoutStep, CONFIG.pollIntervalMs);
    observer = new MutationObserver(tryCheckoutStep);
    if (document.body) {
      observer.observe(document.body, {
        childList: true, subtree: true, attributes: true,
        attributeFilter: ['disabled', 'aria-disabled', 'class', 'value', 'checked', 'aria-checked'],
      });
    }
    tryCheckoutStep();
  }

  // ===========================================================================
  // SHIP: Belarus → Add to cart → корзина
  // ===========================================================================

  function runShipFlow(shipProfile) {
    const pageKey = getShipPageKey();
    const attemptsKey = `rsi_sniper_attempts_${pageKey}`;
    let stopped = false;
    let snipeLocked = false;
    let attempts = parseInt(sessionStorage.getItem(attemptsKey) || '0', 10);
    let pollTimer = null;
    let observer = null;
    let countryChangeAt = 0;
    let countryDropdownOpen = false;
    let awaitingCartConfirm = false;
    let addToCartClickedAt = 0;
    let cartCountBeforeAdd = 0;
    let addToCartRetries = 0;
    let pageLoadAt = Date.now();
    let reloadCount = 0;
    let inactiveButtonSeenAt = 0;

    GM_setValue(storageKey('active'), true);

    function matchesCountry(text) {
      const names = CONFIG.countryAliases?.length ? CONFIG.countryAliases : [CONFIG.targetCountry];
      const normalized = normalize(text);
      return names.some((name) => {
        const target = normalize(name);
        return normalized.includes(target) || target.includes(normalized);
      });
    }

    function matchesButtonText(el) {
      const text = normalize(el.innerText || el.textContent || el.getAttribute('aria-label') || '');
      return CONFIG.buttonTexts.some((needle) => text.includes(needle));
    }

    function findCountryButton() {
      for (const el of document.querySelectorAll('*')) {
        const text = el.childNodes.length === 1 && el.childNodes[0].nodeType === 3 ? el.textContent : '';
        if (text && text.trim() === 'Current country selected:') {
          const root = el.closest('div') || el.parentElement;
          const btn = root?.querySelector('button');
          if (btn) return btn;
        }
      }
      for (const btn of document.querySelectorAll('button')) {
        const label = (btn.textContent || '').trim();
        if (!label || label.length > 50) continue;
        const lower = normalize(label);
        if (lower.includes('usd') || lower.includes('add to cart') || lower.includes('prev')
          || lower.includes('next') || lower.includes('all products') || lower.includes('ships')
          || lower.includes('gear') || lower.includes('download')) continue;
        if (btn.closest('footer')) return btn;
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
      const btn = findCountryButton();
      if (!btn) return true;
      return matchesCountry(btn.textContent);
    }

    function isCurrencyCorrect() {
      const btn = findCurrencyButton();
      if (!btn) return true;
      return normalize(btn.textContent).includes(normalize(CONFIG.targetCurrencyLabel));
    }

    function isPriceCorrect() {
      if (!shipProfile.requirePrice) return true;
      return bodyHasExpectedPrice(shipProfile.expectedPrice);
    }

    function isShipPageReady() {
      const text = document.body?.innerText || '';
      if (!text || text.length < 300) return false;
      const hasButton = !!findAddToCartButtonCandidate();
      const hasOos = CONFIG.unavailableTexts.some((t) => normalize(text).includes(t));
      if (shipProfile.requirePrice) {
        return hasButton || hasOos || bodyHasExpectedPrice(shipProfile.expectedPrice);
      }
      return hasButton || hasOos;
    }

    function pickCountryOption() {
      const selectors = '[role="option"], [role="menuitem"], [role="menuitemradio"], li, div[class*="option"]';
      for (const opt of document.querySelectorAll(selectors)) {
        if (matchesCountry(opt.textContent || '')) { forceClick(opt); return true; }
      }
      for (const inp of document.querySelectorAll('input[type="search"], input[type="text"]')) {
        if (!isVisible(inp)) continue;
        setInputValue(inp, CONFIG.targetCountry);
        break;
      }
      for (const opt of document.querySelectorAll(selectors)) {
        if (matchesCountry(opt.textContent || '')) { forceClick(opt); return true; }
      }
      return false;
    }

    function ensurePricingReady() {
      if (!shipProfile.requireRegion && !shipProfile.requirePrice) return true;

      if (countryChangeAt && Date.now() - countryChangeAt < CONFIG.countrySettleMs) {
        showHud('Ждём цену (Belarus)…', 'warn');
        return false;
      }
      if (shipProfile.requireRegion) {
        if (!isCurrencyCorrect()) {
          const btn = findCurrencyButton();
          if (btn) { forceClick(btn); countryChangeAt = Date.now(); showHud('USD…', 'warn'); return false; }
        }
        if (!isCountryCorrect()) {
          const btn = findCountryButton();
          if (!btn) { showHud('Страна не найдена', 'err'); return false; }
          if (!countryDropdownOpen) {
            forceClick(btn); countryDropdownOpen = true; countryChangeAt = Date.now();
            showHud('Belarus…', 'warn'); return false;
          }
          if (pickCountryOption()) { countryDropdownOpen = false; countryChangeAt = Date.now(); }
          return false;
        }
        countryDropdownOpen = false;
      }
      if (!isPriceCorrect()) {
        showHud(`Ждём $${shipProfile.expectedPrice}…`, 'warn');
        return false;
      }
      return true;
    }

    function isCartAddedModalVisible() {
      const bodyText = normalize(document.body?.innerText || '');
      return CONFIG.cartAddedTexts.some((t) => bodyText.includes(normalize(t)));
    }

    function isCartAddConfirmed() {
      if (isCartAddedModalVisible()) return true;
      if (cartNetworkState.lastAddOk && Date.now() - cartNetworkState.lastAddAt < 60000) return true;
      return getHeaderCartCount() > cartCountBeforeAdd;
    }

    function goToCart() {
      if (!isCartAddConfirmed()) {
        showHud('Корзина не подтверждена — ждём', 'warn');
        return;
      }
      awaitingCartConfirm = false;
      startFlow();
      showHud('→ корзина', 'info');
      if (CONFIG.goToCartOnSuccess) location.replace(getCartUrl());
    }

    async function attemptAddToCart(candidate) {
      cartCountBeforeAdd = getHeaderCartCount();
      cartNetworkState.lastAddOk = false;
      cartNetworkState.lastAddError = null;

      const skuId = parseSkuSlugFromPage();
      if (CONFIG.useGraphQLAddToCart && skuId) {
        showHud(`${shipProfile.label}: GraphQL addMany…`, 'info');
        const api = await addToCartViaGraphQL(skuId, 1);
        if (api.ok) {
          awaitingCartConfirm = true;
          addToCartClickedAt = Date.now();
          return;
        }
        console.warn('[RSI Sniper] GraphQL не удался, клик по кнопке', api);
      }

      try {
        forceClick(candidate);
      } catch (err) {
        snipeLocked = false;
        throw err;
      }
      awaitingCartConfirm = true;
      addToCartClickedAt = Date.now();
    }

    function findAddToCartButtonCandidate() {
      const selectors = 'button, a[role="button"], [role="button"], input[type="submit"], input[type="button"]';
      let fallback = null;
      for (const el of document.querySelectorAll(selectors)) {
        if (!elementMatchesAddToCart(el)) continue;
        if (!isVisible(el)) continue;
        if (el.closest('footer, nav, header')) continue;
        if (!fallback) fallback = el;
        if (isDisabled(el)) return el;
      }
      return fallback;
    }

    function isAddToCartActive(el) {
      return !!el && isVisible(el) && !isDisabled(el);
    }

    function findAddToCartButton() {
      const el = findAddToCartButtonCandidate();
      return isAddToCartActive(el) ? el : null;
    }

    function tryReloadForInactiveButton() {
      if (!CONFIG.autoReloadEnabled) {
        showHud('Add to cart неактивна (reload выкл.)', 'warn');
        return;
      }
      if (reloadCount >= CONFIG.maxReloads) {
        showHud(`Лимит reload (${CONFIG.maxReloads})`, 'err');
        return;
      }
      const now = Date.now();
      if (!inactiveButtonSeenAt) inactiveButtonSeenAt = now;
      const delay = CONFIG.inactiveButtonReloadDelayMs;
      const elapsed = now - inactiveButtonSeenAt;
      if (elapsed < delay) {
        const left = Math.max(1, Math.ceil((delay - elapsed) / 1000));
        const oos = CONFIG.unavailableTexts.some((t) => normalize(document.body?.innerText || '').includes(t));
        showHud(oos ? `Out of Stock — reload через ${left}с` : `Add to cart неактивна — reload через ${left}с`, 'warn');
        return;
      }
      inactiveButtonSeenAt = 0;
      reloadCount += 1;
      console.info('[RSI Sniper] reload (кнопка неактивна)', reloadCount, '/', CONFIG.maxReloads);
      showHud(`Reload ${reloadCount}/${CONFIG.maxReloads}…`, 'warn');
      location.reload();
    }

    function trySnipe() {
      if (stopped) return;
      if (awaitingCartConfirm) {
        const elapsed = Date.now() - addToCartClickedAt;
        const minWait = CONFIG.afterAddToCartDelayMs;
        if (elapsed < minWait) {
          showHud(`Пауза после Add to cart… ${Math.max(1, Math.ceil((minWait - elapsed) / 1000))}с`, 'info');
          return;
        }
        if (isCartAddConfirmed()) {
          goToCart();
          return;
        }
        if (elapsed > CONFIG.cartAddedTimeoutMs) {
          addToCartRetries += 1;
          awaitingCartConfirm = false;
          snipeLocked = false;
          cartNetworkState.lastAddOk = false;
          const errHint = cartNetworkState.lastAddError ? ` (${cartNetworkState.lastAddError})` : '';
          if (addToCartRetries >= CONFIG.maxAddToCartRetries) {
            showHud(`В корзину не добавлено${errHint}`, 'err');
            return;
          }
          showHud(`Повтор ${addToCartRetries}/${CONFIG.maxAddToCartRetries}${errHint}`, 'warn');
          return;
        }
        const cartNow = getHeaderCartCount();
        showHud(`Ждём корзину… ${cartNow} (было ${cartCountBeforeAdd}) · ${Math.ceil((CONFIG.cartAddedTimeoutMs - elapsed) / 1000)}с`, 'warn');
        return;
      }

      const pageAge = Date.now() - pageLoadAt;
      if (!isShipPageReady()) {
        if (pageAge > CONFIG.pageReadyTimeoutMs) {
          showHud('Страница не загрузилась (кнопка/цена)', 'err');
          return;
        }
        showHud('Ищем кнопку и цену…', 'warn');
        return;
      }

      if (snipeLocked || !ensurePricingReady()) return;

      const candidate = findAddToCartButtonCandidate();
      if (candidate) {
        if (isAddToCartActive(candidate)) {
          inactiveButtonSeenAt = 0;
          if (snipeLocked) return;
          snipeLocked = true;
          attempts += 1;
          sessionStorage.setItem(attemptsKey, String(attempts));
          attemptAddToCart(candidate).catch((err) => {
            console.warn('[RSI Sniper] attemptAddToCart:', err);
            snipeLocked = false;
            awaitingCartConfirm = false;
            showHud('Ошибка Add to cart', 'err');
          });
          return;
        }
        tryReloadForInactiveButton();
        return;
      }

      inactiveButtonSeenAt = 0;
      attempts += 1;
      sessionStorage.setItem(attemptsKey, String(attempts));
      const oos = CONFIG.unavailableTexts.some((t) => normalize(document.body?.innerText || '').includes(t));
      showHud(oos ? `Out of Stock (ищем кнопку) #${attempts}` : `Ищем Add to cart… #${attempts}`, oos ? 'warn' : 'info');
    }

    pollTimer = setInterval(trySnipe, CONFIG.pollIntervalMs);
    observer = new MutationObserver(trySnipe);
    if (document.body) {
      observer.observe(document.body, { childList: true, subtree: true, attributes: true,
        attributeFilter: ['disabled', 'aria-disabled', 'class'] });
    }

    const modeHint = shipProfile.isOdin ? 'Odin: Belarus + $5,900' : shipProfile.label;
    showHud(`Снайпер v${VERSION} · ${modeHint}`, 'info');
    trySnipe();
  }

  console.info(`[RSI Sniper v${VERSION}] GraphQL addMany + подтверждение корзины; Odin: Belarus + цена`);
  try {
    boot();
  } catch (err) {
    console.error('[RSI Sniper] FATAL boot:', err);
    showBootMarker();
  }
})();
