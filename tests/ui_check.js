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
        check(/more separation than a single population/.test(await p.locator('#clusters-out .verdict').innerText()) && !/stopped/.test(await p.locator('#clusters-status').innerText()), `${name}: the test of no real sub-groups runs with categories of several levels and detects the patterns`);
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
