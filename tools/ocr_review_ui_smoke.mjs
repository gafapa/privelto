// Run with an installed playwright-core module path; no documents or models are loaded.
import { createServer } from 'node:http';
import { readFile, mkdir } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
import { createRequire } from 'node:module';
import assert from 'node:assert/strict';

const require = createRequire(import.meta.url);
const { chromium } = require(process.argv[2] || 'playwright-core');
const root = resolve('app/ui');
const output = resolve('tmp/ocr-review-ui');
await mkdir(output, { recursive: true });
const server = createServer(async (request, response) => {
  const path = resolve(root, '.' + new URL(request.url, 'http://localhost').pathname);
  if (!path.startsWith(root + '/') && !path.startsWith(root + '\\') && path !== root) {
    response.writeHead(403).end(); return;
  }
  try {
    const file = path === root ? resolve(root, 'index.html') : path;
    response.setHeader('Content-Type', { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2' }[extname(file)] || 'application/octet-stream');
    response.end(await readFile(file));
  } catch { response.writeHead(404).end(); }
});
await new Promise((done) => server.listen(0, '127.0.0.1', done));
const browser = await chromium.launch({ headless: true });
try {
  for (const [name, width, height, language] of [['desktop', 1280, 900, 'es'], ['mobile', 390, 844, 'de']]) {
    const page = await browser.newPage({ viewport: { width, height } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.addInitScript(({ language }) => {
      const original = '<img src=x onerror="window.injected=true"> ' + 'Álvarez'.repeat(40) + ' العربية 日本語';
      const findings = [0, 1, 2].map((id) => ({ id: String(id), page: 0, type: 'email', label: 'Correo electrónico',
        value: `person${id}@example.test`, confidence: 0.999, token: `[EMAIL_${id}]`, enabled: true,
        source: 'ocr', view: [[0.1, 0.1 + id * 0.1, 0.8, 0.15 + id * 0.1]], ocr_review_ids: id < 2 ? [`r${id}`] : [] }));
      const details = { id: 'synthetic', name: 'Synthetic OCR review.pdf', revision: 1,
        findings, pages: [{ width: 595, height: 842 }], notes: [], nativeChars: 100, seconds: 1,
        history: {}, report: null, ocrReviews: [{ id: 'r0', original, alternative: 'person0@example.test', status: 'disagreed' },
          { id: 'r1', original: 'person1@example.test', alternative: '', status: 'illegible' }] };
      window.fixtureOriginal = original;
      window.pywebview = { api: {
        status: async () => ({ language, advancedInstalled: true, advancedLabel: 'Qwen 8B', services: { gpu: false },
          categories: [{ key: 'email', label: 'Correo electrónico', default: true }] }),
        documents: async () => [{ id: 'synthetic', name: details.name, status: 'ready', revision: details.revision, pages: 1, findings: 3 }],
        details: async () => structuredClone(details),
        page_image: async () => 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="595" height="842"><rect width="595" height="842" fill="white"/></svg>'),
        set_enabled: async (_, ids, enabled) => { findings.forEach((finding) => { if (ids.includes(finding.id)) finding.enabled = enabled; }); details.revision++; },
      } };
    }, { language });
    await page.goto(`http://127.0.0.1:${server.address().port}/index.html`);
    await page.evaluate(() => window.dispatchEvent(new Event('pywebviewready')));
    await page.waitForSelector('#rowsBody tr');
    assert.equal(await page.locator('#rowsBody tr').count(), 3);
    await page.selectOption('#typeFilter', '__review');
    assert.equal(await page.locator('#rowsBody tr').count(), 2);
    const summary = page.locator('.ocr-readings summary').first();
    await summary.focus();
    await page.keyboard.press('Space');
    assert.equal(await page.locator('.ocr-readings').first().getAttribute('open'), '');
    assert.equal(await page.locator('.ocr-readings dd').first().textContent(), await page.evaluate(() => window.fixtureOriginal));
    assert.equal(await page.locator('.ocr-readings img').count(), 0);
    assert.equal(await page.evaluate(() => Boolean(window.injected)), false);
    await page.keyboard.press('j');
    assert.equal(await page.locator('#rowsBody tr.current').count(), 0);
    await page.locator('#rowsBody input[type=checkbox]').first().uncheck();
    await page.waitForSelector('#rowsBody tr.off');
    assert.equal(await page.locator('.ocr-readings').first().getAttribute('open'), '');
    await page.locator('.ocr-readings summary').nth(1).click();
    assert.equal(await page.locator('.ocr-readings').nth(1).locator('dd').nth(1).textContent(), language === 'de' ? 'Keine Lesung' : 'Sin lectura');
    await page.locator('#rowsBody').scrollIntoViewIfNeeded();
    await page.screenshot({ path: resolve(output, `${name}.png`), fullPage: true });
    assert.deepEqual(errors, []);
    console.log(`${name}: review filter, keyboard, disclosure persistence, empty reading, and safe Unicode text passed`);
    await page.close();
  }
} finally {
  await browser.close();
  await new Promise((done) => server.close(done));
}
