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
    const runs = [['#demo-phenotypes', 'three sub-types (k-means, MICE)', true, 'kmeans', 'mice'], ['#demo-phenotypes', 'three sub-types (latent classes)', true, 'lca', 'gower'], ['#demo-none', 'one population', false, null, null]];
    for (const [demo, name, expectReal, method, missing] of runs) {
        const p = await browser.newPage({ viewport: { width: 1200, height: 900 } });
        const errs = [];
        p.on('pageerror', e => errs.push(String(e)));
        p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
        p.on('dialog', d => d.accept());
        await p.goto(url);
        const visibleSteps = () => p.$$eval('section.step', ss => ss.filter(s => s.offsetParent !== null).map(s => s.dataset.step));
        check(JSON.stringify(await visibleSteps()) === '["1"]', `${name}: only step 1 is shown at the start`);
        await p.click(demo);
        await p.waitForSelector('#var-table table');
        const roles = await p.evaluate(() => ['Age', 'Sex'].map(h => window.UMEApp.state.vars[h].role));
        check(roles.every(r => r === 'describe'), `${name}: age and sex start as "describe only"`);
        await p.click('#to-step3');
        const advice = await p.locator('#method-advice').innerText();
        check(/Suggested for these data: k-medoids \(PAM\)/.test(advice) && (await p.inputValue('#method')) === 'pam-gower', `${name}: method advisor suggests and preselects PAM / Gower for these mixed data`);
        if (method) { await p.selectOption('#method', method); await p.selectOption('#missing', missing); }
        else check(!(await p.isDisabled('#gower-weights')) && (await p.inputValue('#gower-weights')) === 'balanced', `${name}: PAM / Gower uses balanced weighting by default`);
        await p.click('#explore-btn');
        await p.waitForSelector('#describe-btn', { timeout: 300000 });
        const lead = await p.locator('#explore-out .lead').innerText();
        if (missing === 'mice') check(/multiply imputed: 5 imputed datasets/.test(lead) && /Agreement across imputations/i.test(await p.locator('#explore-out table').innerText()), `${name}: multiple imputation (MICE) used and agreement across imputations shown`);
        if (method === 'kmeans') check(/Suggested: \d clusters/.test(lead) && /Also stable: .*3/.test(lead), `${name}: suggestion offers a stable solution and lists 3 clusters (${lead.slice(0, 60)}...)`);
        if (method === 'lca') check(/Suggested: 3 clusters, the stable solution with the lowest BIC/.test(lead) && /Entropy/i.test(await p.locator('#explore-out table').innerText()), `${name}: BIC chooses the true 3 classes; entropy shown`);
        if (!expectReal) check(!/Suggested:/.test(lead), `${name}: no stable number of clusters is suggested`);
        if (expectReal) { await p.selectOption('#k-choice', '3'); }
        await p.click('#describe-btn');
        await p.waitForSelector('.verdict', { timeout: 300000 });
        const verdict = (await p.locator('.verdict').innerText()).replace(/\s+/g, ' ');
        check(expectReal ? !/Do not interpret/.test(verdict) : /Do not interpret/.test(verdict), `${name}: verdict (${verdict.slice(0, 90)}...)`);
        check((await p.locator('#clusters-out svg').count()) >= 2, `${name}: silhouette plot and map drawn`);
        // Maps: t-SNE and UMAP made on request, faithful (trustworthiness), coloured by a variable
        if (method !== 'lca') {
            for (const m of ['tsne', 'umap']) { await p.click(`[data-map="${m}"]`); await p.waitForSelector(`[data-map="${m}"].primary`, { timeout: 300000 }); }
            const mapText = await p.locator('#map-panel').innerText(), tr = Number((mapText.match(/Trustworthiness ([0-9.]+)/) || [])[1]);
            check(/UMAP keeps close neighbours together/.test(mapText) && tr > 0.85, `${name}: t-SNE and UMAP maps made, with trustworthiness (${tr}) and a caution`);
            await p.selectOption('#map-colour', 'Callosal_Angle_Deg');
            check((await p.locator('#map-panel circle[fill]').count()) > 100 && /Callosal_Angle_Deg:/.test(await p.locator('#map-panel .legend').innerText()), `${name}: map coloured by a measured variable, with a colour scale`);
        } else check(/Classical scaling/.test(await p.locator('#map-panel').innerText()), `${name}: with values left missing, the map uses classical scaling of the dissimilarities`);
        // Step 6: classification (names, rules with cross-validated agreement, new-patient calculator)
        await p.click('#to-use');
        await p.waitForSelector('#np-go', { timeout: 300000 });
        const use = await p.locator('#use-out').innerText();
        check(/Bedside rules/.test(use) && /\d+% of patients not used to derive them/.test(use), `${name}: step 6 shows bedside rules with cross-validated agreement`);
        check(expectReal ? /reasonable to propose/.test(use) : /Caution: the clusters chosen in step 5 did not pass/.test(use), `${name}: step 6 ${expectReal ? 'proposes a classification' : 'warns that the classification would describe noise'}`);
        const ageIdx = await p.evaluate(() => window.UMEApp.state.explore.P.specs.findIndex(s => s.col === 'Callosal_Angle_Deg'));
        await p.fill(`#np-${ageIdx}`, '70'); await p.click('#np-go');
        const np = await p.locator('#np-out').innerText();
        check(/By the rules:/.test(np) && (method === 'lca' ? /By the latent class model: .*probability \d+%/ : /By the full clustering:/).test(np), `${name}: a new patient is classified by the rules and by the ${method === 'lca' ? 'latent class model (with probabilities)' : 'full clustering'}`);
        await p.fill('#name-0', 'Classic CSF-dynamic'); await p.locator('#name-0').dispatchEvent('change');
        check(/A · Classic CSF-dynamic/.test(await p.locator('#use-out').innerText()), `${name}: renaming a class updates the rules`);
        await p.click('.tab[data-step="2"]');
        check(JSON.stringify(await visibleSteps()) === '["2"]', `${name}: clicking the "2 Variables" tab shows only that step`);
        await p.click('.tab[data-step="5"]');
        check(JSON.stringify(await visibleSteps()) === '["5"]', `${name}: clicking the "5 The clusters" tab shows only that step`);
        const overflow = () => p.evaluate(() => {
            const de = document.documentElement, step = [...document.querySelectorAll('section.step')].find(x => x.offsetParent !== null);
            return de.scrollWidth > de.clientWidth + 1 || step.scrollWidth > step.clientWidth + 1;
        });
        for (const [w, label] of [[1200, 'desktop'], [375, 'phone']]) {
            await p.setViewportSize({ width: w, height: 900 });
            const bad = [];
            for (const st of ['1', '2', '3', '4', '5', '6']) { await p.click(`.tab[data-step="${st}"]`); if (await overflow()) bad.push(st); }
            check(bad.length === 0, `${name}: nothing spills past the page edge (${label} width)${bad.length ? ': steps ' + bad.join(', ') : ''}`);
        }
        check(errs.length === 0, `${name}: no page errors${errs.length ? ': ' + errs.join(' | ') : ''}`);
        await p.close();
    }
    await browser.close(); server.close();
    console.log(failed ? `${failed} check(s) failed` : 'All UI checks passed');
    process.exit(failed ? 1 : 0);
})().catch(e => { console.error(e); server.close(); process.exit(1); });
