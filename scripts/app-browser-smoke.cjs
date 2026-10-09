/* Optional release check: npm install playwright in a separate tooling directory. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {chromium} = require('playwright');
const base = process.env.MARKETSIGNAL_BASE_URL || 'http://127.0.0.1:8050';
const output = process.env.MARKETSIGNAL_SCREENSHOT_DIR || '/tmp/marketsignal-app-check';

(async () => {
  fs.mkdirSync(output, {recursive: true});
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'marketsignal-browser-'));
  const context = await chromium.launchPersistentContext(profile, {
    viewport: {width: 1440, height: 1000},
    ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {}),
    args: ['--no-sandbox'],
  });
  try {
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base);
    await page.locator('#workspace-mode').waitFor();
    await page.evaluate(() => navigator.serviceWorker.ready);
    await page.waitForFunction(() => !!navigator.serviceWorker.controller);
    const manifest = await page.evaluate(async () => (await fetch('/manifest.webmanifest')).json());
    assert.equal(manifest.display, 'standalone');
    const cdp = await context.newCDPSession(page);
    await cdp.send('Page.enable');
    const install = await cdp.send('Page.getInstallabilityErrors');
    assert.deepEqual(install.installabilityErrors, [], 'Browser rejected app installability');
    // Accessible keyboard navigation starts with the skip link.
    await page.keyboard.press('Tab');
    assert.equal(await page.locator(':focus').textContent(), 'Skip to workspace');
    await page.keyboard.press('Enter');
    await page.getByText('Install app', {exact: true}).click();
    await page.getByText('iPhone / iPad', {exact: true}).waitFor();
    await page.getByText('Install app', {exact: true}).click();
    for (const width of [1440, 768, 390, 320]) {
      await page.setViewportSize({width, height: 900});
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `Overflow at ${width}px`);
      await page.screenshot({path: `${output}/research-${width}.png`, fullPage: true});
    }
    await page.locator('#workspace-mode label').filter({hasText: 'Studies'}).click();
    await page.locator('#study-workspace').waitFor({state: 'visible'});
    assert.equal(await page.locator('#research-workspace').isVisible(), false);
    await page.reload();
    await page.locator('#study-workspace').waitFor({state: 'visible'});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.screenshot({path: `${output}/studies-mobile.png`, fullPage: true});
    if (process.env.MARKETSIGNAL_LIVE_RESEARCH === '1') {
      await page.locator('#workspace-mode label').filter({hasText: 'Research'}).click();
      await page.locator('#run-button').click();
      await page.locator('#export-button').waitFor({timeout: 200000});
      await page.reload();
      await page.locator('#export-button').waitFor({timeout: 30000});
      for (const width of [390, 1440]) {
        await page.setViewportSize({width, height: 1000});
        await page.waitForTimeout(600); // Allow Plotly to finish its responsive resize.
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `Result overflow at ${width}px`);
        await page.screenshot({path: `${output}/completed-research-${width}.png`, fullPage: true});
      }
      const downloadEvent = page.waitForEvent('download');
      await page.locator('#export-button').click();
      const download = await downloadEvent;
      const saved = `${output}/research-report.json`;
      await download.saveAs(saved);
      const record = JSON.parse(fs.readFileSync(saved, 'utf8'));
      assert.equal(record.report.settings.ticker, 'AAPL');
      assert.ok(record.report_id);
      await page.locator('#workspace-mode label').filter({hasText: 'Saved'}).click();
      await page.locator('#library-list-status').filter({hasText: 'saved records available'}).waitFor();
      await page.locator('#library-record').click();
      await page.getByRole('option').filter({hasText: 'Research report'}).first().click();
      await page.waitForFunction(() => !document.querySelector('#library-export').disabled);
      await page.getByText('Archived results, not current market data.', {exact: false}).waitFor();
      await page.setViewportSize({width: 390, height: 900});
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      await page.locator('#library-view .js-plotly-plot .main-svg').first().waitFor();
      await page.screenshot({path: `${output}/saved-mobile.png`, fullPage: true});
      console.log('PASS: live research, reload recovery, mobile/desktop charts, retained report download, saved history.');
    }
    // Offline should show a clear state, then a standalone reconnect screen.
    await context.setOffline(true);
    await page.locator('#connection-banner').waitFor({state: 'visible'});
    await page.reload();
    await page.getByRole('heading', {name: 'Reconnect to your research.'}).waitFor();
    await page.screenshot({path: `${output}/offline-mobile.png`, fullPage: true});
    await context.setOffline(false);
    await page.getByRole('link', {name: 'Try again'}).click();
    await page.locator('#workspace-mode').waitFor();
    // The service worker must not retain research, callbacks, or application HTML.
    const cachedPaths = await page.evaluate(async () => {
      const keys = (await caches.keys()).filter(key => key.startsWith('marketsignal-offline-'));
      const requests = await Promise.all(keys.map(async key => (await caches.open(key)).keys()));
      return requests.flat().map(request => new URL(request.url).pathname).sort();
    });
    assert.deepEqual(cachedPaths, ['/assets/icons/app-192.png', '/assets/icons/app-512.png', '/offline']);
    assert.deepEqual(errors, []);
    console.log('PASS: installability, navigation/reload, 4 screen widths, keyboard access, offline/reconnect, cache isolation.');
    console.log(`Screenshots: ${output}`);
  } finally { await context.close(); fs.rmSync(profile, {recursive: true, force: true}); }
})().catch(error => { console.error(error); process.exitCode = 1; });
