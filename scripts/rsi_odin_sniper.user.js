// ==UserScript==
// @name         RSI Ship Sniper — Avenger Titan (train)
// @namespace    https://robertsspaceindustries.com/
// @version      1.6.2
// @description  Тренировка: купон → MAX credits → Continue → Place order (без клика)
// @author       InfoBot
// @match        https://robertsspaceindustries.com/en/pledge/Standalone-Ships/Avenger-Titan-10-Year*
// @match        *://robertsspaceindustries.com/*/pledge/Standalone-Ships/Avenger-Titan-10-Year*
// @match        *://*.robertsspaceindustries.com/*/pledge/Standalone-Ships/Avenger-Titan-10-Year*
// @include      /^https?:\/\/(.*\.)?robertsspaceindustries\.com\/.*\/pledge\/Standalone-Ships\/Avenger-Titan-10-Year/
// @match        https://robertsspaceindustries.com/en/store/pledge/cart*
// @match        *://robertsspaceindustries.com/*/store/pledge/cart*
// @match        *://*.robertsspaceindustries.com/*/store/pledge/cart*
// @run-at       document-idle
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM_notification
// ==/UserScript==

(function () {
  'use strict';

  const VERSION = '1.6.2';
  console.info(`[RSI Sniper v${VERSION}] загружен:`, location.href);

  const CONFIG = {
    targetPathPart: '/Standalone-Ships/Avenger-Titan-10-Year',
    shipLabel: 'Avenger Titan 10 Year',

    targetCountry: 'Belarus',
    countryAliases: ['Belarus', 'Беларусь'],
    targetCurrencyLabel: 'USD',
    expectedPriceContains: '60.00',

    reloadIntervalMs: 500,
    pollIntervalMs: 50,
    countrySettleMs: 600,
    cartAddedTimeoutMs: 4000,
    stepDelayMs: 450,
    stepTimeoutMs: 8000,

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

  CONFIG.applyStoreCredits = !CONFIG.targetPathPart.toLowerCase().includes('warbond');

  const STORAGE_SUCCESS = 'rsi_sniper_success_' + CONFIG.targetPathPart;
  const STORAGE_ACTIVE = 'rsi_sniper_active_' + CONFIG.targetPathPart;
  const SESSION_ATTEMPTS = 'rsi_sniper_attempts_' + CONFIG.targetPathPart;
  const SESSION_FLOW = 'rsi_sniper_flow_active';
  const SESSION_STEP = 'rsi_sniper_checkout_step';

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

  function isTargetPage(loc = location) {
    return loc.pathname.toLowerCase().includes(CONFIG.targetPathPart.toLowerCase());
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
  }

  function getStep() {
    return sessionStorage.getItem(SESSION_STEP) || STEPS.CART;
  }

  function setStep(step) {
    sessionStorage.setItem(SESSION_STEP, step);
  }

  function clearFlow() {
    sessionStorage.removeItem(SESSION_FLOW);
    sessionStorage.removeItem(SESSION_STEP);
    sessionStorage.removeItem(SESSION_ATTEMPTS);
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

  function forceClick(el) {
    el.scrollIntoView({ block: 'center', inline: 'center' });
    el.focus({ preventScroll: true });
    for (const type of ['pointerdown', 'mousedown', 'mouseup', 'click']) {
      el.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, view: window }));
    }
    if (typeof el.click === 'function') el.click();
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
    GM_setValue(STORAGE_SUCCESS, true);
    GM_setValue(STORAGE_ACTIVE, false);
    clearFlow();
    setStep(STEPS.DONE);
    playSuccessSound();
    notifySuccess(reason);
  }

  window.rsiSniperReset = () => {
    GM_setValue(STORAGE_SUCCESS, false);
    GM_setValue(STORAGE_ACTIVE, false);
    clearFlow();
    location.reload();
  };

  window.rsiSniperStop = () => {
    GM_setValue(STORAGE_ACTIVE, false);
    clearFlow();
    removeHud();
    showHud('Остановлен вручную', 'info');
  };

  if (CONFIG.stopAfterSuccess && GM_getValue(STORAGE_SUCCESS, false)) {
    console.info('[RSI Sniper] уже сработал — rsiSniperReset()');
    const banner = document.createElement('div');
    banner.textContent = 'RSI Sniper: уже сработал. F12 → rsiSniperReset()';
    banner.style.cssText = 'position:fixed;top:12px;right:12px;z-index:2147483647;padding:8px 12px;background:#3ddc84;color:#000;font:600 12px sans-serif;border-radius:6px';
    document.documentElement.appendChild(banner);
    return;
  }

  function boot() {
    if (isCheckoutPage() && isFlowActive()) {
      runCheckoutFlow();
      return;
    }

    if (isCheckoutPage()) {
      console.info('[RSI Sniper] checkout без активного flow — выход');
      return;
    }

    if (!isTargetPage()) {
      console.info('[RSI Sniper] не целевая страница:', location.pathname);
      return;
    }

    runShipFlow();
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

  function runShipFlow() {
    let stopped = false;
    let snipeLocked = false;
    let attempts = parseInt(sessionStorage.getItem(SESSION_ATTEMPTS) || '0', 10);
    let lastReloadAt = 0;
    let pollTimer = null;
    let reloadTimer = null;
    let observer = null;
    let countryChangeAt = 0;
    let countryDropdownOpen = false;
    let awaitingCartConfirm = false;
    let addToCartClickedAt = 0;

    GM_setValue(STORAGE_ACTIVE, true);

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
          || lower.includes('gear') || lower.includes('download') || lower.includes('avenger')) continue;
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
      return (document.body?.innerText || '').includes(CONFIG.expectedPriceContains);
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
      if (countryChangeAt && Date.now() - countryChangeAt < CONFIG.countrySettleMs) {
        showHud('Ждём цену (Belarus)…', 'warn');
        return false;
      }
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
      if (!isPriceCorrect()) { showHud(`Ждём $${CONFIG.expectedPriceContains}…`, 'warn'); return false; }
      return true;
    }

    function isCartAddedModalVisible() {
      const bodyText = normalize(document.body?.innerText || '');
      return CONFIG.cartAddedTexts.some((t) => bodyText.includes(normalize(t)));
    }

    function goToCart() {
      awaitingCartConfirm = false;
      startFlow();
      showHud('→ корзина', 'info');
      if (CONFIG.goToCartOnSuccess) location.replace(getCartUrl());
    }

    function findAddToCartButton() {
      for (const el of document.querySelectorAll('button, a[role="button"], [role="button"], input[type="submit"], input[type="button"]')) {
        const text = normalize(el.textContent || el.value || '');
        if (!CONFIG.buttonTexts.some((t) => text.includes(t))) continue;
        if (isVisible(el) && !isDisabled(el)) return el;
      }
      return null;
    }

    function trySnipe() {
      if (stopped) return;
      if (awaitingCartConfirm) {
        const elapsed = Date.now() - addToCartClickedAt;
        if (isCartAddedModalVisible() || elapsed > CONFIG.cartAddedTimeoutMs) goToCart();
        else showHud(`Ждём added to cart… ${Math.ceil((CONFIG.cartAddedTimeoutMs - elapsed) / 1000)}с`, 'warn');
        return;
      }
      if (snipeLocked || !ensurePricingReady()) return;

      attempts += 1;
      sessionStorage.setItem(SESSION_ATTEMPTS, String(attempts));
      const btn = findAddToCartButton();
      if (btn) {
        forceClick(btn);
        snipeLocked = true;
        awaitingCartConfirm = true;
        addToCartClickedAt = Date.now();
        showHud('Add to cart…', 'info');
        return;
      }

      const oos = CONFIG.unavailableTexts.some((t) => normalize(document.body?.innerText || '').includes(t));
      showHud(oos ? `Out of Stock #${attempts}` : `Кнопка… #${attempts}`, oos ? 'warn' : 'info');
    }

    function maybeReload() {
      if (stopped || awaitingCartConfirm) return;
      const now = Date.now();
      if (now - lastReloadAt < CONFIG.reloadIntervalMs) return;
      lastReloadAt = now;
      location.reload();
    }

    pollTimer = setInterval(trySnipe, CONFIG.pollIntervalMs);
    reloadTimer = setInterval(maybeReload, CONFIG.reloadIntervalMs);
    observer = new MutationObserver(trySnipe);
    if (document.body) {
      observer.observe(document.body, { childList: true, subtree: true, attributes: true,
        attributeFilter: ['disabled', 'aria-disabled', 'class'] });
    }

    showHud(`Снайпер #${attempts}`, 'info');
    trySnipe();
  }

  console.info(`[RSI Sniper v${VERSION}] Avenger Titan train → Place order`);
  boot();
})();
