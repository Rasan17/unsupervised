// ui_check.js: end-to-end browser test of the interface on both demo cohorts (Playwright + Chrome).
// Serves the folder on a local port, runs data -> approach -> explore -> clusters, and checks that the
// page raises no errors (including security-policy violations) and that the verdicts are the honest ones.
// Run: node tests/ui_check.js   (set PLAYWRIGHT to the playwright package path if it is not installed here)
'use strict';
const http = require('http'), fs = require('fs'), path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const ROOT = path.join(__dirname, '..');
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.csv': 'text/csv' };
const server = http.createServer((req, res) => {
    const f = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
    if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(f)] || 'application/octet-stream' }); fs.createReadStream(f).pipe(res);
});
let failed = 0;
const check = (ok, msg) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${msg}`); if (!ok) failed++; };
(async () => {
    await new Promise(r => server.listen(0, '127.0.0.1', r));
    const url = `http://127.0.0.1:${server.address().port}/index.html`;
    const browser = await chromium.launch(process.env.CI ? {} : { channel: 'chrome' });
    for (const [demo, name, expectReal] of [['#demo-phenotypes', 'three sub-types', true], ['#demo-none', 'one population', false]]) {
        const p = await browser.newPage({ viewport: { width: 1200, height: 900 } });
        const errs = [];
        p.on('pageerror', e => errs.push(String(e)));
        p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
        p.on('dialog', d => d.accept());
        await p.goto(url);
        const visibleSteps = () => p.$$eval('section.step', ss => ss.filter(s => s.offsetParent !== null).map(s => s.dataset.step));
        check(JSON.stringify(await visibleSteps()) === '["1"]', `${name}: only step 1 is shown at the start`);
        await p.click(demo); await p.click('#to-step3'); await p.click('#explore-btn');
        await p.waitForSelector('#describe-btn', { timeout: 300000 });
        const lead = await p.locator('#explore-out .lead').innerText();
        if (expectReal) check(/Suggested: \d clusters/.test(lead) && /Also stable: .*3/.test(lead), `${name}: suggestion offers a stable solution and lists 3 clusters (${lead.slice(0, 60)}...)`);
        else check(!/Suggested:/.test(lead), `${name}: no stable number of clusters is suggested`);
        if (expectReal) { await p.selectOption('#k-choice', '3'); }
        await p.click('#describe-btn');
        await p.waitForSelector('.verdict', { timeout: 300000 });
        const verdict = (await p.locator('.verdict').innerText()).replace(/\s+/g, ' ');
        check(expectReal ? !/Do not interpret/.test(verdict) : /Do not interpret/.test(verdict), `${name}: verdict (${verdict.slice(0, 90)}...)`);
        check((await p.locator('#clusters-out svg').count()) >= 2, `${name}: silhouette plot and map drawn`);
        await p.click('.tab[data-step="2"]');
        check(JSON.stringify(await visibleSteps()) === '["2"]', `${name}: clicking the "2 Variables" tab shows only that step`);
        await p.click('.tab[data-step="5"]');
        check(JSON.stringify(await visibleSteps()) === '["5"]', `${name}: clicking the "5 The clusters" tab shows only that step`);
        check(errs.length === 0, `${name}: no page errors${errs.length ? ': ' + errs.join(' | ') : ''}`);
        await p.close();
    }
    await browser.close(); server.close();
    console.log(failed ? `${failed} check(s) failed` : 'All UI checks passed');
    process.exit(failed ? 1 : 0);
})().catch(e => { console.error(e); server.close(); process.exit(1); });
