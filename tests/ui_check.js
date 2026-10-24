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
const waitUntil = async (fn, ms = 120000) => { const t0 = Date.now(); while (Date.now() - t0 < ms) { if (await fn()) return true; await new Promise(r => setTimeout(r, 250)); } throw new Error('timed out'); };
const CE_PRNG = seed => { let t = seed >>> 0; return () => { t = (t + 0x6D2B79F5) >>> 0; let x = Math.imul(t ^ (t >>> 15), t | 1); x ^= x + Math.imul(x ^ (x >>> 7), x | 61); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; }; };
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
        if (!method || method === 'kmeans') {
            await p.click('.tab[data-step="2"]'); await p.click('#to-check'); await p.click('#check-btn'); await p.waitForSelector('#check-out h3', { timeout: 300000 });
            const ct = await p.locator('#check-out').innerText();
            const nUnusual = Number((ct.match(/Unusual patients \((\d+)/) || [])[1]);
            check(/Possible data errors \(0\)/.test(ct) && nUnusual <= 0.01 * 360, `${name}: the data checks find no data errors and at most 1% unusual patients in a clean cohort (${nUnusual})`);
            await p.click('#check-next');
        }
        const advice = await p.locator('#method-advice').innerText();
        check(await p.isDisabled('#method-neural') && /needs at least 1,000 records, so it is not offered for these data \(360\)/.test(advice), `${name}: the neural network is not offered for 360 patients, and the advisor states the minimum of 1,000 records`);
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
        {
            const ev = await p.locator('#clusters-out details.evidence').innerText();
            check(/Stability \(weakest cluster/.test(ev) && /Separation \(average silhouette\)/.test(ev) && /Smallest cluster/.test(ev) && /Not tested by the app \(clinical usefulness\)/.test(ev) && (method !== 'lca' || /random starts reached the best solution/.test(ev)), `${name}: an evidence summary separates statistical evidence from clinical usefulness`);
        }
        if (missing) check(/Missing values by cluster/.test(await p.locator('#clusters-out').innerText()), `${name}: missing values are compared between clusters`);
        if (missing === 'mice') {
            const ct = await p.locator('#clusters-out').innerText();
            await p.check('#map-imp');
            const want = await p.evaluate(() => { const P = window.UMEApp.state.explore.P, m = window.UMEMaps._state().maps[window.UMEMaps._state().kind]; return m.idx.filter(i => P.incomplete[i]).length; });
            const got = await p.$$eval('#map-panel circle.hl', c => c.length);
            check(/Patients with imputed values/.test(ct) && /multiply imputed/.test(ct) && want > 0 && got === want && /patient\(s\) on this map with imputed values/.test(await p.locator('#map-panel').innerText()), `${name}: each cluster's share of patients with imputed values is shown, and the map highlights those patients (${got})`);
            await p.uncheck('#map-imp');
        }
        // Maps: t-SNE and UMAP made on request, faithful (trustworthiness), coloured by a variable
        if (method !== 'lca') {
            for (const m of ['ae', 'tsne', 'umap']) {
                await p.click(`[data-map="${m}"]`); await p.waitForSelector(`[data-map="${m}"].primary`, { timeout: 300000 });
                if (m === 'ae') { const at = await p.locator('#map-panel').innerText(); check(/reconstructs \d+% of the variation/.test(at) && /Training curve/.test(at) && /overfit/.test(at), `${name}: autoencoder map with its reconstruction share, training curve and caution`); }
            }
            const mapText = await p.locator('#map-panel').innerText(), tr = Number((mapText.match(/Trustworthiness ([0-9.]+)/) || [])[1]);
            check(/UMAP keeps close neighbours together/.test(mapText) && tr > 0.85, `${name}: t-SNE and UMAP maps made, with trustworthiness (${tr}) and a caution`);
            await p.selectOption('#map-colour', 'Callosal_Angle_Deg');
            check((await p.locator('#map-panel circle[fill]').count()) > 100 && /Callosal_Angle_Deg:/.test(await p.locator('#map-panel .legend').innerText()), `${name}: map coloured by a measured variable, with a colour scale`);
        } else check(/FAMD/.test(await p.locator('#map-panel').innerText()), `${name}: for mixed data the base map is FAMD`);
        // The axes of the PCA / FAMD map: three main variables per axis, with their contribution (as in the dimensions table)
        await p.click(`[data-map="${(await p.locator('#map-panel [data-map="pca"]').count()) ? 'pca' : 'famd'}"]`).catch(() => {});
        const ax = await p.evaluate(() => { const t = document.querySelector('#map-panel table.axis-factors'); if (!t) return null; const M = window.UMEMaps._state(), F = M.famd, P = window.UMEApp.state.explore.P;
            const want = [0, 1].map(s2 => P.specs.map((sp, j) => [sp.col.replace(/_/g, ' '), F.contrib[j][s2]]).sort((a2, b2) => b2[1] - a2[1]).slice(0, 3).map(([n2, c2]) => `${n2} ${c2.toFixed(0)}%`));
            const got = [...t.querySelectorAll('tbody tr')].map(tr => [...tr.querySelectorAll('td')].slice(1).map(td => td.firstChild.textContent + td.childNodes[1].textContent.trim().replace(/^/, ' ').replace(/^ /, ' ')));
            return { rows: got.length, ok: want.every((w, r) => w.every((x, c) => (got[r][c] || '').replace(/\s+/g, ' ').trim() === x)) }; });
        check(ax && ax.rows === 2 && ax.ok, `${name}: the map lists the three main variables of each axis with their contribution, matching the dimensions table`);
        const dimsText = await p.locator('#dims-panel').innerText();
        check(/FAMD \(factor analysis of mixed data\)/.test(dimsText) && /Dimension 1 \(\d+% of the variation\): mainly/.test(dimsText) && /separates the clusters/.test(dimsText), `${name}: dimensions panel describes what drives dimension 1 and how it separates the clusters`);
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
        if (method !== 'lca') {
            const numIdx = await p.evaluate(() => window.UMEApp.state.explore.P.specs.map((sp, j) => (sp.type === 'continuous' ? j : -1)).filter(j => j >= 0));
            for (const j of numIdx) await p.fill(`#np-${j}`, '100000');
            await p.click('#np-go');
            check(/Poorly represented: treat as unclassified/.test(await p.locator('#np-out').innerText()), `${name}: a patient far from every class is reported as poorly represented (unclassified)`);
            await p.click('#np-clear');
        }
        await p.fill('#name-0', 'Classic CSF-dynamic'); await p.locator('#name-0').dispatchEvent('change');
        check(/A · Classic CSF-dynamic/.test(await p.locator('#use-out').innerText()), `${name}: renaming a class updates the rules`);
        // Step 8: decision tree diagram, consistent with the bedside rules of step 7
        await p.click('#to-tree');
        await p.waitForSelector('#tree-out svg.tree', { timeout: 60000 });
        const nRules = await p.evaluate(() => window.ClusterEngine.treeRules(window.UMEUse._state.tree).length);
        const leafBoxes = await p.$$eval('#tree-out svg.tree rect[stroke-width="2.5"]', r => r.length);
        check(leafBoxes === nRules && /yes<\/text>/.test(await p.locator('#tree-out .tree-wrap').innerHTML()) && /Classic CSF-dynamic/.test(await p.locator('#tree-out').innerText()), `${name}: decision tree drawn with one end point per bedside rule (${leafBoxes}), yes/no branches and the class names from step 7`);
        await p.selectOption('#tree-depth', '1');
        check(await p.$$eval('#tree-out svg.tree rect[stroke-width="2.5"]', r => r.length) === 2 && await p.evaluate(() => window.UMEUse._state.depth) === 1, `${name}: changing the depth redraws the tree (depth 1: 2 end points) and is shared with step 7`);
        const [dl] = await Promise.all([p.waitForEvent('download'), p.click('#tree-out .export-bar button[data-fmt="svg"]')]);
        const svgText = require('fs').readFileSync(await dl.path(), 'utf8');
        check(/^<\?xml/.test(svgText) && /<svg xmlns="http:\/\/www.w3.org\/2000\/svg"/.test(svgText) && /fill="#ffffff"/.test(svgText), `${name}: the diagram downloads as a stand-alone SVG on a white background`);
        // Every chart and diagram has an image download (SVG and PNG), with its legend, on white
        await p.click('.tab[data-step="6"]');
        const nCharts = await p.$$eval('svg.chart, svg.tree', s => s.length), nBars = await p.$$eval('.export-bar', b => b.length);
        check(nCharts > 0 && nCharts === nBars, `${name}: every chart and diagram has a download bar (${nBars} of ${nCharts})`);
        await p.click('.tab[data-step="5"]');
        const [d1] = await Promise.all([p.waitForEvent('download'), p.locator('#explore-out .export-bar button[data-fmt="svg"]').first().click()]);
        const t1 = require('fs').readFileSync(await d1.path(), 'utf8');
        check(/^<\?xml/.test(t1) && /fill="#ffffff"/.test(t1) && /Average silhouette/.test(t1) && !/class="/.test(t1), `${name}: a chart downloads as a stand-alone SVG with its legend (${d1.suggestedFilename()})`);
        await p.click('.tab[data-step="6"]');
        const [d2] = await Promise.all([p.waitForEvent('download'), p.locator('#map-panel .export-bar button[data-fmt="png"]').first().click()]);
        const b2 = require('fs').readFileSync(await d2.path());
        check(b2.slice(1, 4).toString() === 'PNG' && b2.length > 5000, `${name}: a map downloads as a PNG image (${d2.suggestedFilename()}, ${Math.round(b2.length / 1024)} kB)`);
        await p.click('.tab[data-step="2"]');
        check(JSON.stringify(await visibleSteps()) === '["2"]', `${name}: clicking the "2 Variables" tab shows only that step`);
        await p.click('.tab[data-step="6"]');
        check(JSON.stringify(await visibleSteps()) === '["6"]', `${name}: clicking the "6 The clusters" tab shows only that step`);
        const overflow = () => p.evaluate(() => {
            const de = document.documentElement, step = [...document.querySelectorAll('section.step')].find(x => x.offsetParent !== null);
            return de.scrollWidth > de.clientWidth + 1 || step.scrollWidth > step.clientWidth + 1;
        });
        for (const [w, label] of [[1200, 'desktop'], [375, 'phone']]) {
            await p.setViewportSize({ width: w, height: 900 });
            const bad = [];
            for (const st of ['1', '2', '3', '4', '5', '6', '7', '8']) { await p.click(`.tab[data-step="${st}"]`); if (await overflow()) bad.push(st); }
            check(bad.length === 0, `${name}: nothing spills past the page edge (${label} width)${bad.length ? ': steps ' + bad.join(', ') : ''}`);
        }
        check(errs.length === 0, `${name}: no page errors${errs.length ? ': ' + errs.join(' | ') : ''}`);
        await p.close();
    }
    // Audit features: warnings before the analysis, uncertainty of stability, q-values and effect sizes, sensitivity
    // analysis, the analysis record, a new cohort classified without retraining, and privacy (no data leave the page)
    {
        const name = 'audit features', p = await browser.newPage({ viewport: { width: 1200, height: 900 } });
        const errs = [], reqs = [];
        p.on('pageerror', e => errs.push(String(e))); p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); }); p.on('dialog', d => d.accept());
        p.on('request', r => reqs.push({ method: r.method(), url: r.url() }));
        await p.goto(url);
        // A file with problems that are not errors in single values
        const lines = ['ID,Visit_Date,Score,Score_Copy,Constant,Grade'];
        for (let i = 0; i < 60; i++) lines.push(`P${i},2024-0${1 + (i % 9)}-1${i % 9},${i % 7 === 0 ? -999 : 10 + (i % 13)},${i % 7 === 0 ? -999 : 20 + 2 * (i % 13)},same,${i % 3}`);
        await p.setInputFiles('#file', { name: 'problems.csv', mimeType: 'text/csv', buffer: Buffer.from(lines.join('\n')) });
        await p.waitForSelector('#var-table table');
        check(/Data: problems\.csv \(60 rows, your file\)/.test(await p.locator('#data-banner').innerText()) && /not a medical device/.test(await p.locator('section.step[data-step="1"]').innerText()), `${name}: a banner names the user's own file, and step 1 states that the tool is not a medical device`);
        await p.selectOption('#role-1', 'cluster');
        const warn = await p.locator('#var-warnings').innerText();
        check(/Visit_Date: looks like dates/.test(warn) && await p.evaluate(() => window.UMEApp.state.vars.Constant.role) === 'ignore' && /-999 \(\d+ patients\) may be a code for "not recorded"/.test(warn) && /Strongly related clustering variables: .*Score and Score_Copy/.test(warn), `${name}: warnings before the analysis for dates, a missing-value code and two variables measuring the same thing; a constant column starts as "Ignore"`);
        // The phenotype demo with a confirmatory purpose, PAM / Gower, 3 clusters
        await p.click('#to-step3');
        check(/warning\(s\) about the data were shown in step 2/.test(await p.locator('#warn-reminder').innerText()), `${name}: step 4 reminds the user of the warnings before the analysis`);
        await p.click('.tab[data-step="1"]'); await p.click('#demo-phenotypes'); await p.waitForSelector('#var-table table');
        check(/Data: demo_inph_phenotypes\.csv \(simulated demonstration, not real patients\)/.test(await p.locator('#data-banner').innerText()), `${name}: a banner marks simulated demonstration data on every step`);
        await p.click('#to-step3');
        await p.selectOption('#purpose', 'confirmatory'); await p.fill('#question', 'Are there sub-types of iNPH?');
        await p.click('#explore-btn'); await p.waitForSelector('#describe-btn', { timeout: 300000 });
        check(/\(0\.\d{3}–[01]\.\d{3}\)/.test(await p.locator('#explore-out table').first().innerText()), `${name}: step 5 shows the 95% interval of the stability of each candidate`);
        await p.selectOption('#k-choice', '3'); await p.click('#describe-btn'); await p.waitForSelector('#to-use', { timeout: 300000 });
        const co = await p.locator('#clusters-out').innerText();
        check(/Confirmatory purpose/.test(co) && /The 0.75 limit is a convention/.test(co) && /Patients: \d+% stay in their cluster in at least 90%/.test(co) && /\d+–\d+%/.test(co), `${name}: step 6 shows the purpose, interval and share range of each cluster's stability, and patient-level stability`);
        check(/Benjamini–Hochberg/.test(co) && /Cramér's V/.test(co), `${name}: the cluster profile gives q-values (false discovery rate) and effect sizes`);
        const fitE = Number((co.match(/reproduce the observed correlations to within ([\d.]+)/) || [])[1]);
        check(fitE >= 0 && fitE <= 0.05, `${name}: the copula test reports how closely its reference reproduces the observed correlations (${fitE})`);
        await p.click('#sens-btn'); await p.waitForSelector('#sens-out table', { timeout: 600000 });
        const so = await p.locator('#sens-out').innerText(), nAlt = await p.$$eval('#sens-out tbody tr', r => r.length);
        check(/Without Callosal_Angle_Deg/.test(so) && /Method:/.test(so) && /Patients: \d+% stay in the same cluster under every alternative/.test(so) && nAlt >= 6, `${name}: the sensitivity analysis compares ${nAlt} alternative specifications, at patient level`);
        const [dj] = await Promise.all([p.waitForEvent('download'), p.click('#record-json')]);
        const rec = JSON.parse(require('fs').readFileSync(await dj.path(), 'utf8'));
        check(rec.software && rec.clinicalQuestion === 'Are there sub-types of iNPH?' && /Confirmatory/.test(rec.purpose) && rec.chosen && rec.sensitivity && rec.sensitivity.length === nAlt && rec.settings && rec.data.simulatedDemo === true && /not a medical device/.test(rec.disclaimer), `${name}: the analysis record (.json) holds the software version, purpose, question, settings, chosen solution and sensitivity analysis`);
        const [dt] = await Promise.all([p.waitForEvent('download'), p.click('#record-txt')]);
        check(/Are there sub-types of iNPH\?/.test(require('fs').readFileSync(await dt.path(), 'utf8')), `${name}: the analysis report (.txt) downloads`);
        // R script: holds no patient data, and (when R is installed) repeats the PAM / Gower clustering exactly
        {
            const os = require('os'), cp = require('child_process'), dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ume-r-'));
            const [dr] = await Promise.all([p.waitForEvent('download'), p.click('#record-r')]); const rFile = path.join(dir, dr.suggestedFilename()); await dr.saveAs(rFile);
            const [dc] = await Promise.all([p.waitForEvent('download'), p.click('#export-btn')]); await dc.saveAs(path.join(dir, dc.suggestedFilename()));
            fs.copyFileSync(path.join(ROOT, 'demo_inph_phenotypes.csv'), path.join(dir, 'demo_inph_phenotypes.csv'));
            const rText = fs.readFileSync(rFile, 'utf8'), ids = await p.evaluate(() => window.UMEApp.state.data.rows.slice(0, 50).map(r => r.Patient_ID));
            check(/cluster::pam\(D, k, diss = TRUE\)/.test(rText) && /weights = w/.test(rText) && ids.every(id => !rText.includes(id)), `${name}: an R script to repeat the analysis downloads, without patient data`);
            let rOut = null; try { rOut = cp.execFileSync('Rscript', [path.basename(rFile)], { cwd: dir, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }); } catch (e) { rOut = e.code === 'ENOENT' ? null : `error: ${String(e.stderr || e.message).slice(-300)}`; }
            if (rOut === null) console.log('SKIP  R is not installed: the R script was not run');
            else check(/adjusted Rand index\): 1\b/.test(rOut), `${name}: run in R (cluster::pam on cluster::daisy), the script reproduces the app's clusters exactly${/Rand index\): ([\d.]+)/.test(rOut) ? ` (adjusted Rand ${rOut.match(/Rand index\): ([\d.]+)/)[1]})` : `: ${rOut.slice(-200)}`}`);
        }
        // Step 7: classify an independent cohort without retraining
        await p.click('#to-use'); await p.waitForSelector('#np-go', { timeout: 300000 });
        await p.setInputFiles('#val-file', path.join(ROOT, 'demo_inph_validation.csv'));
        await p.waitForSelector('#val-out table');
        const vo = await p.locator('#val-out').innerText();
        const agreeV = Number((vo.match(/same class for (\d+)%/) || [])[1]);
        check(/200 records classified without retraining/.test(vo) && agreeV >= 70 && /Outcomes by class in the new cohort/.test(vo) && /Gait_Improvement_12mo/.test(vo), `${name}: a new cohort is classified without retraining, with agreement (${agreeV}%) and outcomes by class`);
        // Privacy: only the page's own files were requested (all GET, same origin); nothing stored in the browser
        const origin = new URL(url).origin, foreign = reqs.filter(r => r.method !== 'GET' || !(r.url.startsWith(origin) || r.url.startsWith('blob:') || r.url.startsWith('data:')));
        const store = await p.evaluate(async () => ({ ls: localStorage.length, ss: sessionStorage.length, cookie: document.cookie, idb: indexedDB.databases ? (await indexedDB.databases()).length : 0, caches: 'caches' in window ? (await caches.keys()).length : 0 }));
        check(foreign.length === 0 && store.ls === 0 && store.ss === 0 && store.cookie === '' && store.idb === 0 && store.caches === 0, `${name}: privacy: ${reqs.length} requests, all GETs of the app's own files; no browser storage, cookies or caches used${foreign.length ? ': ' + JSON.stringify(foreign.slice(0, 3)) : ''}`);
        check(errs.length === 0, `${name}: no page errors${errs.length ? ': ' + errs.join(' | ') : ''}`);
        await p.close();
    }
    // Edge cases: very small data, rare categories, an all-missing column, duplicate identifiers, extreme numbers.
    // The page must either complete or explain why it cannot, without page errors.
    {
        const name = 'edge cases', p = await browser.newPage({ viewport: { width: 1200, height: 900 } });
        const errs = []; p.on('pageerror', e => errs.push(String(e))); p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); }); p.on('dialog', d => d.accept());
        const r = CE_PRNG(9), files = {
            'tiny.csv': ['ID,A,B', ...Array.from({ length: 12 }, (_, i) => `T${i},${(i % 2) * 5 + r()},${(i % 2) * 5 + r()}`)],
            'awkward.csv': ['ID,Score,Big,Empty,Group,Constant', ...Array.from({ length: 80 }, (_, i) => `${i < 3 ? 'D1' : 'P' + i},${(i % 2) * 4 + r()},${(1e9 * (1 + r())).toFixed(0)},,${i === 5 ? 'Rare' : i % 2 ? 'Left' : 'Right'},7`)],
            'one-variable.csv': ['ID,Only', ...Array.from({ length: 40 }, (_, i) => `O${i},${(i % 2) * 6 + r()}`)]
        };
        for (const [fname, lines] of Object.entries(files)) {
            await p.goto(url);
            await p.setInputFiles('#file', { name: fname, mimeType: 'text/csv', buffer: Buffer.from(lines.join('\n')) });
            await p.waitForSelector('#var-table table');
            let outcome;
            if (await p.isDisabled('#to-step3')) outcome = `stops at step 2: ${(await p.locator('#var-summary').innerText()).match(/Choose at least two variables[^.]*\./)?.[0] || 'next step not available'} (${JSON.stringify(await p.evaluate(() => Object.fromEntries(Object.entries(window.UMEApp.state.vars).map(([h, v]) => [h, v.role]))))})`;
            else {
                await p.click('#to-step3'); await p.click('#explore-btn');
                await waitUntil(async () => (await p.$('#describe-btn')) || /stopped|cannot|could not|needs|at least/i.test(await p.locator('#explore-status').innerText()));
                outcome = (await p.locator('#explore-status').innerText()).trim();
            }
            if (await p.$('#describe-btn')) {
                await p.click('#describe-btn');
                await waitUntil(async () => (await p.$('#to-use')) || /stopped|cannot|could not/i.test(await p.locator('#clusters-status').innerText()));
                outcome = (await p.$('#to-use')) ? 'completed' : (await p.locator('#clusters-status').innerText()).trim();
            }
            check(errs.length === 0 && outcome.length > 0, `${name}: ${fname} is handled without page errors (${outcome.slice(0, 90)})`);
        }
        await p.goto(url);
        await p.setInputFiles('#file', { name: 'awkward.csv', mimeType: 'text/csv', buffer: Buffer.from(files['awkward.csv'].join('\n')) });
        await p.waitForSelector('#var-table table');
        const w = await p.locator('#var-warnings').innerText();
        const roles = await p.evaluate(() => Object.fromEntries(Object.entries(window.UMEApp.state.vars).map(([h, v]) => [h, v.role])));
        check(/Rare categories.*"Rare" \(1\)/.test(w) && roles.Empty === 'ignore' && roles.Constant === 'ignore' && roles.Group === 'cluster', `${name}: the rare category is named; the empty and constant columns start as "Ignore", and the last column is not taken as an outcome`);
        await p.close();
    }
    // Step 3: data checks on the cohort with planted problems, and exclusion carried into the analysis
    {
        const name = 'planted data errors', p = await browser.newPage({ viewport: { width: 1200, height: 900 } });
        const errs = []; p.on('pageerror', e => errs.push(String(e))); p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
        await p.goto(url); await p.click('#demo-errors'); await p.waitForSelector('#var-table table');
        check(await p.evaluate(() => window.UMEApp.state.vars.Rout_mmHg_min_ml.type) === 'continuous', `${name}: a number column with one decimal comma stays numerical`);
        await p.click('#to-check'); await p.check('#check-ae'); await p.click('#check-btn'); await p.waitForSelector('#check-out h3', { timeout: 300000 });
        check(/of the 4 methods/.test(await p.locator('#check-out').innerText()) && /rebuilt worst by the autoencoder/.test(await p.locator('#check-out').innerText()), `${name}: the optional autoencoder adds a fourth method with the variables it rebuilds worst`);
        const planted = await p.evaluate(() => window.UMEApp.state.data.rows.map(r => r.Simulated_Planted_Problem ? r.Patient_ID : null).filter(Boolean));
        const found = await p.$$eval('#check-out table tbody tr td:nth-child(2)', tds => [...new Set(tds.map(td => td.textContent.trim()))]);
        const hit = planted.filter(id => found.includes(id) || (id === 'PH-115' && found.includes('PH-999')));
        check(hit.length >= planted.length - 0, `${name}: every planted problem is listed (${hit.length} of ${planted.length})`);
        check(/combination of .* is unusual/.test(await p.locator('#check-out').innerText()), `${name}: the patient with an unusual combination of plausible values is explained as such`);
        await p.click('#check-out details summary');
        const nErr = await p.$$eval('#check-out table', ts => ts[0].querySelectorAll('tbody tr').length);
        await p.locator('[data-genuine]', { hasText: 'negative values in Callosal_Angle_Deg' }).click();
        check(await p.$$eval('#check-out table', ts => ts[0].querySelectorAll('tbody tr').length) === nErr - 1 && /Marked as genuine: negative values in Callosal_Angle_Deg \(1\)/.test(await p.locator('#check-out').innerText()), `${name}: a flag can be marked as genuine for a whole variable and is no longer listed`);
        await p.click('#genuine-undo');
        check(await p.$$eval('#check-out table', ts => ts[0].querySelectorAll('tbody tr').length) === nErr, `${name}: marked flags can be listed again`);
        await p.click('[data-bulk="strong"][data-on="1"]');
        const nStrong = await p.$$eval('#check-out table', ts => ts[1].querySelectorAll('tbody tr').length);
        check(new RegExp(`${nStrong} patient\\(s\\) will be left out`).test(await p.locator('#check-summary').innerText()), `${name}: "Exclude all" excludes every unusual patient (${nStrong})`);
        await p.click('[data-bulk="strong"][data-on="0"]');
        check(/No patients excluded/.test(await p.locator('#check-summary').innerText()), `${name}: "Include all" includes them again`);
        await p.locator('#check-out input.excl').first().check();
        check(/1 patient\(s\) will be left out/.test(await p.locator('#check-summary').innerText()), `${name}: excluding a patient is summarised`);
        await p.click('#check-next'); await p.click('#explore-btn'); await p.waitForSelector('#describe-btn', { timeout: 300000 });
        check(/1 patient\(s\) were left out after the data checks/.test(await p.locator('#explore-out .lead').innerText()) && await p.evaluate(() => window.UMEApp.state.explore.P.rows.length) === 359, `${name}: the exclusion is applied to the analysis (359 patients)`);
        check(errs.length === 0, `${name}: no page errors${errs.length ? ': ' + errs.join(' | ') : ''}`);
        await p.close();
    }
    // Large administrative dataset: 8,000 referrals with the neural network (self-organising map)
    {
        const name = 'referral patterns (8,000 records)', p = await browser.newPage({ viewport: { width: 1200, height: 900 } });
        const errs = []; p.on('pageerror', e => errs.push(String(e))); p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
        await p.goto(url); await p.click('#demo-referrals'); await p.waitForSelector('#var-table table');
        await p.click('#to-check'); await p.click('#check-btn'); await p.waitForSelector('#check-out h3', { timeout: 600000 });
        const rc = await p.locator('#check-out').innerText(), nU = Number((rc.match(/Unusual patients \((\d+)/) || [])[1]);
        check(/Possible data errors \(0\)/.test(rc) && nU <= 80 && /computed record by record/.test(rc), `${name}: data checks on 8,000 records: no false data errors, ${nU} unusual records (at most 1%), local density computed record by record`);
        await p.click('#check-next');
        const adv = await p.locator('#method-advice').innerText();
        check((await p.inputValue('#method')) === 'neural' && /suitable at this size \(8,000 records; it needs at least 1,000/.test(adv) && await p.isDisabled('#method-pam') && await p.isDisabled('#method-ward'), `${name}: the advisor suggests the neural network, states the minimum, and PAM and Ward are not offered`);
        await p.selectOption('#missing', 'gower'); await p.selectOption('#kmax', '6');
        const t0 = Date.now(); await p.click('#explore-btn'); await p.waitForSelector('#describe-btn', { timeout: 600000 });
        const lead = await p.locator('#explore-out .lead').innerText();
        check(/Self-organising map of \d+ x \d+ units/.test(lead) && /random sample of 2,000 of the 8,000 records/.test(lead), `${name}: map trained and silhouettes on a sample (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
        const ari5 = await p.evaluate(() => { const S = window.UMEApp.state, t = S.explore.P.rows.map(r => r.Simulated_Referral_Pattern), lv = [...new Set(t)], r5 = S.explore.res.find(r => r.k === 5); return window.ClusterEngine.adjustedRand(r5.labels, t.map(x => lv.indexOf(x))); });
        check(ari5 >= 0.7, `${name}: 5 groups recover the simulated referral patterns (adjusted Rand ${ari5.toFixed(2)})`);
        await p.selectOption('#k-choice', '5'); await p.click('#describe-btn'); await p.waitForSelector('#to-use', { timeout: 600000 });
        check((await p.locator('#clusters-out svg[aria-label="Self-organising map of the records coloured by cluster"]').count()) === 1, `${name}: the self-organising map is drawn`);
        // Interactive map: drag a region, then the table shows only the records on the selected units
        await p.locator('#som-svg').scrollIntoViewIfNeeded();
        const bb = await p.locator('#som-svg').boundingBox();
        await p.mouse.move(bb.x + 4, bb.y + 4); await p.mouse.down(); await p.mouse.move(bb.x + bb.width * 0.4, bb.y + bb.height * 0.5, { steps: 6 }); await p.mouse.up();
        const sel = await p.evaluate(() => { const V = window.UMESom._state(), m = V.nn.map; return { units: V.sel.size, records: V.P.rows.filter((_, i) => V.sel.has(m.bmu[i])).length }; });
        const txt = await p.locator('#som-table p').first().innerText(), n1 = Number((txt.match(/^([\d,]+) record/) || [])[1]?.replace(/,/g, ''));
        check(sel.units > 0 && n1 === sel.records && (await p.$$eval('#som-table table.records tbody tr', r => r.length)) === Math.min(50, sel.records), `${name}: dragging over the map selects ${sel.units} units and the table lists exactly their ${sel.records} records`);
        const hlCount = await p.$$eval('#map-panel circle.hl', c => c.length), onMap = await p.evaluate(() => { const V = window.UMESom._state(), M = window.UMEMaps._state(), map = M.maps[M.kind]; return map.idx.filter(i => V.sel.has(V.nn.map.bmu[i])).length; });
        check(hlCount === onMap && hlCount > 0 && /Highlighted: the [\d,]+ record\(s\) selected on the self-organising map/.test(await p.locator('#map-panel').innerText()), `${name}: the selected records are highlighted on the map of the patients (${hlCount})`);
        await p.click('[data-selc="0"]');
        const c1 = await p.evaluate(() => { const V = window.UMESom._state(), m = V.nn.map; return V.P.rows.map((_, i) => i).filter(i => V.sel.has(m.bmu[i])).every(i => V.labels[i] === 0); });
        check(c1 && /cluster 1 100%/.test(await p.locator('#som-table p').first().innerText()), `${name}: the cluster button selects that cluster's units and records`);
        // Conversely: clicking a patient on the map of the patients selects its unit on the self-organising map
        await p.click('#som-clear');
        const dot = p.locator('#map-panel circle[data-row="123"]'); await dot.scrollIntoViewIfNeeded();
        const db = await dot.boundingBox(); await p.mouse.click(db.x + db.width / 2, db.y + db.height / 2);
        const pick = await p.evaluate(() => { const V = window.UMESom._state(), i = V.picked ? V.picked.i : -1; return { sel: [...V.sel], unit: i >= 0 ? V.nn.map.bmu[i] : -2, hits: i >= 0 ? V.nn.map.hits[V.nn.map.bmu[i]] : -1 }; });
        check(pick.sel.length === 1 && pick.sel[0] === pick.unit && (await p.locator(`#som-svg rect.picked[data-unit="${pick.unit}"]`).count()) === 1 && (await p.$$eval('#map-panel circle.hl', c => c.length)) === pick.hits && /Selected from the map of the patients/.test(await p.locator('#som-view').innerText()), `${name}: clicking a patient on the map selects and outlines its unit on the self-organising map (${pick.hits} records)`);
        // Region on the map of the patients: a rectangle, then a closed freehand outline (Option / Alt + drag)
        await p.click('#som-clear');
        const mp = p.locator('#map-panel svg.chart').first(); await mp.scrollIntoViewIfNeeded(); let mb = await mp.boundingBox();
        await p.mouse.move(mb.x + mb.width * 0.15, mb.y + mb.height * 0.55); await p.mouse.down(); await p.mouse.move(mb.x + mb.width * 0.4, mb.y + mb.height * 0.85, { steps: 6 }); await p.mouse.up();
        const recState = () => p.evaluate(() => { const V = window.UMESom._state(); return { rec: V.rec ? V.rec.size : 0, units: V.sel.size, unitsOk: V.rec ? [...V.sel].every(u => [...V.rec].some(i => V.nn.map.bmu[i] === u)) : false, hl: document.querySelectorAll('#map-panel circle.hl').length, table: Number(((document.querySelector('#som-table p') || {}).textContent || '').match(/^([\d,]+)/)?.[1].replace(/,/g, '')) }; });
        const r1 = await recState();
        check(r1.rec > 0 && r1.unitsOk && r1.hl === r1.rec && r1.table === r1.rec, `${name}: a rectangle on the map of the patients selects ${r1.rec} records, shown on ${r1.units} units of the self-organising map and listed exactly in the table`);
        await mp.scrollIntoViewIfNeeded(); mb = await mp.boundingBox();
        const loop = [[0.6, 0.3], [0.85, 0.35], [0.8, 0.7], [0.6, 0.75], [0.55, 0.5], [0.6, 0.3]];
        await p.keyboard.down('Alt'); await p.mouse.move(mb.x + mb.width * loop[0][0], mb.y + mb.height * loop[0][1]); await p.mouse.down();
        for (const [x, y] of loop.slice(1)) await p.mouse.move(mb.x + mb.width * x, mb.y + mb.height * y, { steps: 5 });
        await p.mouse.up(); await p.keyboard.up('Alt');
        const r2 = await recState();
        check(r2.rec > 0 && r2.rec !== r1.rec && r2.unitsOk && r2.table === r2.rec, `${name}: a closed freehand outline (Option + drag) selects ${r2.rec} records and replaces the previous selection`);
        // Removing and deselecting: Cmd/Ctrl-drag removes an area; Escape and a click on empty space clear
        const dragOn = async (from, to, mods = []) => { await mp.scrollIntoViewIfNeeded(); const bx = await mp.boundingBox(); for (const k of mods) await p.keyboard.down(k); await p.mouse.move(bx.x + bx.width * from[0], bx.y + bx.height * from[1]); await p.mouse.down(); await p.mouse.move(bx.x + bx.width * to[0], bx.y + bx.height * to[1], { steps: 6 }); await p.mouse.up(); for (const k of mods.slice().reverse()) await p.keyboard.up(k); };
        await dragOn([0.1, 0.1], [0.95, 0.9]); const rAll = await recState();
        await dragOn([0.1, 0.1], [0.5, 0.9], ['Control']); const rLess = await recState();
        check(rLess.rec > 0 && rLess.rec < rAll.rec && rLess.unitsOk && rLess.table === rLess.rec, `${name}: Cmd / Ctrl + drag removes an area from the selection (${rAll.rec} to ${rLess.rec} records)`);
        await p.keyboard.press('Escape');
        const rEsc = await recState();
        await dragOn([0.1, 0.1], [0.95, 0.9]); await mp.scrollIntoViewIfNeeded(); const bx2 = await mp.boundingBox(); await p.mouse.click(bx2.x + bx2.width * 0.02, bx2.y + bx2.height * 0.5);
        const rEmpty = await recState();
        check(rEsc.rec === 0 && rEsc.units === 0 && rEmpty.rec === 0 && rEmpty.units === 0, `${name}: Escape and a click on an empty area clear the selection`);
        await p.click('[data-selc="0"]'); await p.click('#som-clear');
        check(/No units selected/.test(await p.locator('#som-table').innerText()) && (await p.$$eval('#map-panel circle.hl, #map-panel circle.dim', c => c.length)) === 0, `${name}: clearing the selection empties the table and removes the highlight`);
        // Regression: the test of "no real sub-groups" with categories of several levels (missing values filled)
        await p.click('.tab[data-step="4"]'); await p.selectOption('#missing', 'fill'); await p.click('#explore-btn'); await p.waitForSelector('#describe-btn', { timeout: 600000 });
        await p.selectOption('#k-choice', '5'); await p.click('#describe-btn'); await p.waitForSelector('#to-use', { timeout: 600000 });
        const fitR = ((await p.locator('#clusters-out').innerText()).match(/reproduce the observed correlations to within ([\d.]+)/) || [])[1];
        check(/more separation than a single population/.test(await p.locator('#clusters-out .verdict').innerText()) && !/stopped/.test(await p.locator('#clusters-status').innerText()) && fitR !== undefined, `${name}: the test of no real sub-groups runs with categories of several levels and detects the patterns (correlations reproduced to within ${fitR})`);
        await p.click('#to-use'); await p.waitForSelector('#np-go');
        const j = await p.evaluate(() => window.UMEApp.state.explore.P.specs.findIndex(sp => sp.col === 'Urgency'));
        await p.selectOption(`#np-${j}`, 'Two-week wait'); await p.click('#np-go');
        check(/By the self-organising map:/.test(await p.locator('#np-out').innerText()), `${name}: a new referral is classified by the map`);
        // Decision tree: when more questions are allowed than the tree needs, it says why; unmerged it shows them
        await p.click('#to-tree'); await p.waitForSelector('#tree-out svg.tree'); await p.selectOption('#tree-depth', '4');
        const merged = await p.$$eval('#tree-out svg.tree rect[stroke-width="2.5"]', r => r.length), usedDepth = await p.evaluate(() => window.UMEUse._state.treeDepth);
        const noteOk = usedDepth >= 4 || /The tree uses only/.test(await p.locator('#tree-out').innerText());
        await p.uncheck('#tree-merge');
        const unmerged = await p.$$eval('#tree-out svg.tree rect[stroke-width="2.5"]', r => r.length);
        check(noteOk && unmerged >= merged, `${name}: decision tree explains when it needs fewer questions than allowed, and unmerged shows all end points (${merged} merged, ${unmerged} unmerged)`);
        check(errs.length === 0, `${name}: no page errors${errs.length ? ': ' + errs.join(' | ') : ''}`);
        await p.close();
    }
    await browser.close(); server.close();
    console.log(failed ? `${failed} check(s) failed` : 'All UI checks passed');
    process.exit(failed ? 1 : 0);
})().catch(e => { console.error(e); server.close(); process.exit(1); });
