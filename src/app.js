// app.js: the interface of the Unsupervised Medical Explorer (plain script; uses DataEngine, ClusterEngine,
// UMECharts, UMEUse). Six steps: data, variables, approach, number of clusters, clusters, use of the clusters. All computation stays in the
// browser; every value from the data is escaped before display.
(function () {
    'use strict';
    const DE = window.DataEngine, CE = window.ClusterEngine, CH = window.UMECharts, esc = CH.esc;
    const $ = id => document.getElementById(id);
    const yieldUI = () => new Promise(r => setTimeout(r, 0));
    const S = { data: null, vars: {}, prep: null, explore: null, result: null };
    // Demographic columns start as "describe only": phenotypes are usually built from clinical features and then
    // described by age and sex; yes/no variables such as sex can otherwise dominate the clusters (especially with Gower)
    const DEMOGRAPHIC_RE = /^(age|sex|gender|ethnicity|ethnic[\s_-]?group|race)([\s_-]|$)/i;
    const OUTCOME_RE = /(response|responder|outcome|improve|mortality|death|died|surviv|revision|complication|readmission|recurrence|discharge)/i;
    const METHOD = {
        'pam-gower': { name: 'k-medoids (PAM) with Gower dissimilarity', short: 'PAM / Gower', note: 'Recommended for clinical data that mix numbers, grades and categories, and it tolerates missing values. Each cluster is represented by a real patient (its medoid). With balanced weighting (the default) every variable contributes equally on average; with equal weights (classic Gower), yes/no variables count for more than measurements and can split patients into artificial groups.' },
        'kmeans': { name: 'k-means on standardised variables', short: 'k-means', note: 'For numerical variables; categories are coded 0/1. Assumes roughly round clusters of similar spread.' },
        'ward': { name: 'Hierarchical clustering (Ward) on standardised variables', short: 'Hierarchical (Ward)', note: 'Builds a tree of nested groupings; cutting it gives the clusters. Similar to k-means; needs no random start.' },
        'lca': { name: 'Latent class analysis (model-based; numbers, grades and categories)', short: 'Latent class', note: 'A statistical model: each class has its own distribution of every variable (averages and spreads for numbers, proportions for grades and categories). Gives each patient a probability of belonging to each class and BIC to compare numbers of classes; uses the recorded values directly when some are missing. Widely used for clinical phenotypes. Assumes the variables are unrelated within a class, so strongly related variables can create extra classes.' },
        'gmm': { name: 'Gaussian mixture model (numerical variables only)', short: 'Gaussian mixture', note: 'Allows clusters of different size and spread, with BIC to compare numbers of clusters. Only meaningful for numerical variables (yes/no and graded variables make BIC unreliable).' }
    };

    const MICE_M = 5;   // number of imputed datasets

    // ------------------------------------------------------------------------------------- step navigation
    function goTo(step) {
        document.querySelectorAll('.step').forEach(s => { s.hidden = s.dataset.step !== String(step); });
        document.querySelectorAll('.tab').forEach(t => { const on = t.dataset.step === String(step); t.classList.toggle('active', on); if (on) t.setAttribute('aria-current', 'step'); else t.removeAttribute('aria-current'); });
        window.scrollTo(0, 0);
        if (String(step) === '6' && S.result && !$('use-out').innerHTML) renderUse();
    }
    function renderUse() {
        try { window.UMEUse.render({ S, P: S.explore.P, METHOD, profileTable }); } catch (e) { $('use-out').innerHTML = `<p class="status bad">The classification could not be built: ${esc(e.message)}</p>`; }
    }
    function enableTabs(upTo) { document.querySelectorAll('.tab').forEach(t => { t.disabled = Number(t.dataset.step) > upTo; }); }
    function status(id, text, kind = '') { const el = $(id); el.textContent = text; el.className = `status ${kind}`; }

    // ------------------------------------------------------------------------------------- step 1: data
    function loadText(text, name) {
        let parsed;
        try { parsed = DE.parseCSV(text); } catch (e) { status('data-status', `The file could not be read: ${e.message}`, 'bad'); return; }
        if (!parsed.rows.length) { status('data-status', 'The file has no data rows.', 'bad'); return; }
        S.data = { ...parsed, name, analysis: DE.analyzeColumns(parsed.headers, parsed.rows) };
        S.vars = {}; S.prep = null; S.explore = null; S.result = null; S.methodTouched = false;
        parsed.headers.forEach(h => {
            const a = S.data.analysis[h];
            const role = a.role === 'meta' ? 'ignore' : (a.role === 'target' || OUTCOME_RE.test(h) || /simulated|true.?sub.?type/i.test(h) || DEMOGRAPHIC_RE.test(h)) ? 'describe' : 'cluster';
            const type = a.type === 'continuous' && a.isNumeric ? 'continuous' : a.type === 'ordinal' ? 'ordinal' : 'nominal';
            S.vars[h] = { role, type };
        });
        const issues = DE.parseIssues ? DE.parseIssues(parsed) : [];
        const miss = parsed.headers.reduce((a, h) => a + (S.data.analysis[h].missingCount || 0), 0);
        $('data-summary').innerHTML = `<p><strong>${esc(name)}</strong>: ${parsed.rows.length} patients, ${parsed.headers.length} columns, ${miss} missing value(s).</p>
            ${issues.length ? `<p class="warn">${issues.map(esc).join(' ')}</p>` : ''}
            <div class="table-wrap"><table><thead><tr>${parsed.headers.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead>
            <tbody>${parsed.rows.slice(0, 6).map(r => `<tr>${parsed.headers.map(h => `<td>${esc(r[h] ?? '')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
        status('data-status', '');
        enableTabs(2); renderVariables(); goTo(2);
    }
    async function loadDemo(file) {
        try { const r = await fetch(file); if (!r.ok) throw new Error(String(r.status)); loadText(await r.text(), file); }
        catch (e) { status('data-status', `The demo file could not be loaded (${e.message}). If you opened the page from your disk, load the CSV file with "Choose a CSV file" instead.`, 'bad'); }
    }

    // ------------------------------------------------------------------------------------- step 2: variables
    function levelsOf(col) {
        const vals = [...new Set(S.data.rows.map(r => r[col]).filter(v => !DE.isMissing(v)).map(v => String(v).trim()))];
        return vals.every(v => Number.isFinite(Number(v))) ? vals.sort((a, b) => Number(a) - Number(b)) : vals.sort();
    }
    function renderVariables() {
        const rows = S.data.headers.map((h, i) => {
            const v = S.vars[h], a = S.data.analysis[h];
            return `<tr><td><strong>${esc(h)}</strong><div class="muted">${a.isNumeric ? `numbers, ${a.distinctCount} distinct` : `${a.distinctCount} categories`}${a.missingCount ? `, ${a.missingCount} missing` : ''}</div></td>
                <td><select id="role-${i}" data-col="${esc(h)}" class="role" aria-label="Use of ${esc(h)}">
                    <option value="cluster"${v.role === 'cluster' ? ' selected' : ''}>Use to find clusters</option>
                    <option value="describe"${v.role === 'describe' ? ' selected' : ''}>Describe clusters only</option>
                    <option value="ignore"${v.role === 'ignore' ? ' selected' : ''}>Ignore</option></select></td>
                <td><select id="type-${i}" data-col="${esc(h)}" class="vtype" aria-label="Type of ${esc(h)}">
                    <option value="continuous"${v.type === 'continuous' ? ' selected' : ''}${a.isNumeric ? '' : ' disabled'}>Number</option>
                    <option value="ordinal"${v.type === 'ordinal' ? ' selected' : ''}>Ordered grade</option>
                    <option value="nominal"${v.type === 'nominal' ? ' selected' : ''}>Category</option></select></td></tr>`;
        }).join('');
        $('var-table').innerHTML = `<table><thead><tr><th>Column</th><th>Use</th><th>Type</th></tr></thead><tbody>${rows}</tbody></table>`;
        document.querySelectorAll('.role').forEach(s => s.addEventListener('change', e => { S.vars[e.target.dataset.col].role = e.target.value; updateVarSummary(); }));
        document.querySelectorAll('.vtype').forEach(s => s.addEventListener('change', e => { S.vars[e.target.dataset.col].type = e.target.value; updateVarSummary(); }));
        const pc = $('patient-col');
        pc.innerHTML = '<option value="">Each row is a different patient</option>' + S.data.headers.map(h => `<option value="${esc(h)}">${esc(h)}</option>`).join('');
        updateVarSummary();
    }
    function updateVarSummary() {
        const used = Object.keys(S.vars).filter(h => S.vars[h].role === 'cluster'), desc = Object.keys(S.vars).filter(h => S.vars[h].role === 'describe');
        const types = { continuous: 0, ordinal: 0, nominal: 0 }; used.forEach(h => { types[S.vars[h].type]++; });
        $('var-summary').innerHTML = `${used.length} variable(s) used to find clusters (${types.continuous} numbers, ${types.ordinal} ordered grades, ${types.nominal} categories); ${desc.length} used only to describe them.`
            + (used.length < 2 ? ' <span class="warn">Choose at least two variables to find clusters.</span>' : '')
            + (Object.keys(S.vars).some(h => DEMOGRAPHIC_RE.test(h) && S.vars[h].role === 'describe') ? ' <span class="muted">Age and sex start as "describe only", so that the clusters reflect clinical features rather than demographics (yes/no variables such as sex can otherwise dominate them); change this if demographics are part of the phenotype you are looking for.</span>' : '')
            + (desc.length ? '' : ' <span class="muted">Tip: outcomes (for example shunt response) are best kept as "Describe clusters only": they then show whether the clusters differ in outcome, without having shaped them.</span>');
        $('to-step3').disabled = used.length < 2;
        renderAdvice(used);
        const onlyNumbers = used.every(h => S.vars[h].type === 'continuous');
        $('method-gmm').disabled = !onlyNumbers;
        if (!onlyNumbers && $('method').value === 'gmm') $('method').value = 'pam-gower';
        $('gower-weights').disabled = !['pam-gower', 'lca'].includes($('method').value);
        $('method-hint').textContent = METHOD[$('method').value].note + (onlyNumbers ? '' : ' (Gaussian mixtures need numerical variables only.)');
    }

    // Which method suits these data: from the mix of variable types, the share of missing values and the size
    function advise(used) {
        const n = S.data.rows.length, nCont = used.filter(h => S.vars[h].type === 'continuous').length, nCat = used.length - nCont;
        const miss = used.reduce((a, h) => a + S.data.rows.filter(r => DE.isMissing(r[h]) || (S.vars[h].type === 'continuous' && !Number.isFinite(Number(r[h])))).length, 0) / Math.max(1, n * used.length);
        const missText = miss > 0.05 ? ` About ${Math.round(100 * miss)}% of the values are missing.` : '';
        if (!used.length) return null;
        if (!nCat && miss <= 0.05) return { method: 'kmeans', why: `All ${used.length} clustering variables are numbers and few values are missing: k-means on standardised variables is simple and efficient. Latent class analysis (latent profiles) or a Gaussian mixture also allow clusters of different spread and give membership probabilities.` };
        if (!nCat) return { method: 'lca', why: `All clustering variables are numbers, but some values are missing.${missText} Latent class analysis (latent profiles) uses each patient's recorded values directly, without filling them in.` };
        if (nCat >= nCont) return { method: 'lca', why: `Most clustering variables are categories or grades (${nCat} of ${used.length}): latent class analysis is designed for these, gives each patient's probability of belonging to each class, and uses recorded values directly when some are missing.${missText}` };
        if (n > 3000) return { method: 'lca', why: `A mix of numbers and categories with ${n} patients: PAM needs every pair of patients' dissimilarity, which becomes slow above a few thousand patients; latent class analysis handles mixed data at this size.` };
        return { method: 'pam-gower', why: `A mix of numbers (${nCont}) and categories or grades (${nCat}): PAM with balanced Gower dissimilarity treats each type appropriately and tolerates missing values.${missText} Latent class analysis is a model-based alternative; k-means also works well when most variables are numbers.` };
    }
    function renderAdvice(used) {
        const a = advise(used), box = $('method-advice');
        if (!a) { box.innerHTML = ''; return; }
        if (!S.methodTouched && $('method').value !== a.method && !(a.method === 'gmm' && $('method-gmm').disabled)) { $('method').value = a.method; }
        box.innerHTML = `<p><strong>Suggested for these data: ${esc(METHOD[a.method].name)}.</strong> ${esc(a.why)}${$('method').value === a.method ? '' : ` <button id="use-advice" class="btn small" type="button">Use it</button>`}</p><p class="muted small">Tip: repeat the analysis with a second method. Sub-groups that appear with both are more convincing.</p>`;
        const b = $('use-advice'); if (b) b.addEventListener('click', () => { $('method').value = a.method; S.methodTouched = true; updateVarSummary(); });
    }

    // ------------------------------------------------------------------------------------- data preparation
    function prepare() {
        const method = $('method').value, missing = $('missing').value, gowerWeights = $('gower-weights').value, cols = S.data.headers.filter(h => S.vars[h].role === 'cluster');
        const describeCols = S.data.headers.filter(h => S.vars[h].role === 'describe');
        const specs = cols.map(h => ({ col: h, type: S.vars[h].type, levels: S.vars[h].type === 'continuous' ? null : levelsOf(h) }));
        let rows = S.data.rows.slice();
        const isMiss = (r, h) => DE.isMissing(r[h]) || (S.vars[h].type === 'continuous' && !Number.isFinite(Number(r[h])));
        const nAll = rows.length;
        if (missing === 'complete' || (missing === 'gower' && method !== 'pam-gower' && method !== 'lca')) rows = rows.filter(r => cols.every(h => !isMiss(r, h)));
        let filled = 0;
        const fillVal = {};
        if (missing === 'fill') specs.forEach(s => {
            const vals = S.data.rows.filter(r => !isMiss(r, s.col)).map(r => r[s.col]);
            if (s.type === 'continuous') { const x = vals.map(Number).sort((a, b) => a - b); fillVal[s.col] = x.length ? (x.length % 2 ? x[(x.length - 1) / 2] : (x[x.length / 2 - 1] + x[x.length / 2]) / 2) : 0; }
            else { const cnt = {}; vals.forEach(v => { const k = String(v).trim(); cnt[k] = (cnt[k] || 0) + 1; }); fillVal[s.col] = Object.keys(cnt).sort((a, b) => cnt[b] - cnt[a])[0]; }
        });
        if (missing === 'fill') rows = rows.map(r => { const c = { ...r }; cols.forEach(h => { if (isMiss(r, h)) { c[h] = fillVal[h]; filled++; } }); return c; });
        if (rows.length < 10) throw new Error(`Only ${rows.length} patients remain after handling missing values; at least 10 are needed.`);
        const seed = Number($('seed').value) || 2024, pcol = $('patient-col').value;
        const base = { method, missing, gowerWeights, cols, describe: describeCols, specs, rows, excluded: nAll - rows.length, filled, seed,
            groups: pcol ? rows.map(r => String(r[pcol] ?? '')) : null, patientCol: pcol, allContinuous: specs.every(s => s.type === 'continuous') };
        if (missing !== 'mice') return { ...base, ...encode(rows, specs, method, isMiss, gowerWeights) };
        // Multiple imputation by chained equations (predictive mean matching), from the clustering variables only:
        // outcomes kept for description must not shape the clusters. Each imputed dataset is clustered and the
        // results are combined into a consensus (Basagana et al. 2013, Am J Epidemiol).
        const clean = rows.map(r => { const c = { ...r }; cols.forEach(h => { if (isMiss(r, h)) c[h] = ''; }); return c; });
        const cfg = {}; cols.forEach(h => { cfg[h] = { role: 'feature', type: S.vars[h].type === 'nominal' ? 'nominal' : 'numeric' }; });
        const missingValues = clean.reduce((a, r) => a + cols.filter(h => DE.isMissing(r[h])).length, 0);
        const missingPatients = clean.filter(r => cols.some(h => DE.isMissing(r[h]))).length;
        const mi = DE.miceImpute(clean, cfg, { m: MICE_M, iterations: 10, seed });
        const imps = mi.datasets.map(d => encode(d, specs, method, isMiss, gowerWeights));
        const P = { ...base, ...imps[0], mice: { m: MICE_M, iterations: 10, missingValues, missingPatients } };
        P.impP = imps.map(e => ({ ...P, ...e }));
        // Map: principal components of the average of the imputed (standardised) datasets
        if (imps[0].Z) P.Zmap = imps[0].Z.map((r, i) => r.map((_, j) => imps.reduce((a, e) => a + e.Z[i][j], 0) / imps.length));
        return P;
    }
    // Raw values (for Gower and the mixed-data test) and a numerical coding (for k-means, Ward, mixtures, PCA)
    function encode(rows, specs, method, isMiss, gowerWeights) {
        const raw = rows.map(r => specs.map(s => (isMiss(r, s.col) ? null : (s.type === 'continuous' ? Number(r[s.col]) : String(r[s.col]).trim()))));
        const complete = raw.every(r => r.every(v => v !== null));
        let X = null, Z = null, zstat = null;
        const colNames = [];
        if (complete) {
            X = raw.map(r => { const out = []; specs.forEach((s, j) => { const v = r[j]; if (s.type === 'continuous') out.push(v); else if (s.type === 'ordinal') out.push(s.levels.indexOf(v)); else if (s.levels.length === 2) out.push(v === s.levels[1] ? 1 : 0); else s.levels.forEach(l => out.push(v === l ? 1 : 0)); }); return out; });
            specs.forEach(s => { if (s.type === 'nominal' && s.levels.length > 2) s.levels.forEach(l => colNames.push(`${s.col} = ${l}`)); else colNames.push(s.type === 'nominal' ? `${s.col} = ${s.levels[1]}` : s.col); });
            const st = CE.standardize(X); Z = st.Z; zstat = { mean: st.mean, sd: st.sd };
        }
        // Gower is also computed for latent classes: silhouettes and the new-patient calculator use it
        const gower = method === 'pam-gower' || method === 'lca' ? CE.gowerMatrix(raw, specs.map(s => ({ type: s.type, levels: s.levels })), { weights: gowerWeights === 'balanced' ? 'balanced' : null }) : null;
        const euclid = Z && method !== 'pam-gower' ? CE.euclideanMatrix(Z) : null;
        return { raw, X, Z, zstat, colNames, gower, euclid, D: gower || euclid };
    }
    const lcaVars = P => P.specs.map(s => ({ type: s.type, levels: s.levels }));
    const modelBased = P => P.method === 'gmm' || P.method === 'lca';
    // Clustering of a subset of the patients (indices into P.rows), for the main fit and for bootstrap resamples
    function clusterIdx(P, k, idx, seed) {
        const sub = M => { const m = idx.length, D = new Float64Array(m * m); for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) D[a * m + b] = M.D[idx[a] * M.n + idx[b]]; return { n: m, D }; };
        if (P.method === 'pam-gower') return CE.pam(sub(P.gower), k).labels;
        if (P.method === 'kmeans') return CE.kmeans(idx.map(i => P.Z[i]), k, { seed, nInit: 5 }).labels;
        if (P.method === 'ward') return CE.cutTree(CE.hierarchical(sub(P.euclid), 'ward'), idx.length, k);
        // Bootstrap refits: fewer starts and a looser convergence criterion (the classes, not the last decimals, matter)
        if (P.method === 'lca') return CE.lca(idx.map(i => P.raw[i]), lcaVars(P), k, { nInit: 3, seed, tol: 1e-6, maxIter: 300 }).labels;
        return CE.gmm(idx.map(i => P.Z[i]), k, { covType: 'diag', nInit: 3, seed }).labels;
    }
    function fitOne(P, k) {
        const all = P.rows.map((_, i) => i);
        if (P.method === 'gmm') { const g = CE.gmm(P.Z, k, { covType: 'diag', nInit: 5, seed: P.seed }); return { labels: g.labels, bic: g.bic }; }
        if (P.method === 'lca') { const m = CE.lca(P.raw, lcaVars(P), k, { nInit: 10, seed: P.seed, tol: 1e-8, maxIter: 1000 }); return { labels: m.labels, bic: m.bic, entropy: m.entropy, posterior: m.posterior, model: m.model }; }
        return { labels: clusterIdx(P, k, all, P.seed) };
    }
    // With multiple imputation: cluster every imputed dataset, then the consensus partition is k-medoids on the
    // proportion of imputed datasets in which two patients are NOT in the same cluster.
    function fullFit(P, k) {
        if (!P.impP) return fitOne(P, k);
        const fits = P.impP.map(Q => fitOne(Q, k)), n = P.rows.length, m = fits.length, D = new Float64Array(n * n);
        for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) { let apart = 0; fits.forEach(f => { if (f.labels[a] !== f.labels[b]) apart++; }); D[a * n + b] = D[b * n + a] = apart / m; }
        const labels = CE.pam({ n, D }, k).labels;
        return { labels, bic: modelBased(P) ? fits.reduce((a, f) => a + f.bic, 0) / m : undefined, entropy: P.method === 'lca' ? fits.reduce((a, f) => a + f.entropy, 0) / m : undefined,
            agreement: fits.reduce((a, f) => a + CE.adjustedRand(f.labels, labels), 0) / m };
    }
    // Statistic averaged over the imputed datasets (or computed once without imputation)
    const pooled = (P, fn) => { const Qs = P.impP || [P]; return Qs.reduce((a, Q) => a + fn(Q), 0) / Qs.length; };
    // Clustering of a bootstrap resample: with imputation, successive resamples cycle through the imputed datasets,
    // so the stability also reflects the uncertainty of the imputed values.
    function resampleFn(P, k, salt) { let b = 0; return idx => { const Q = P.impP ? P.impP[b++ % P.impP.length] : P; return clusterIdx(Q, k, idx, P.seed + idx.length * salt); }; }

    // ------------------------------------------------------------------------------------- step 4: how many clusters?
    async function explore() {
        let P;
        try { P = prepare(); } catch (e) { status('explore-status', e.message, 'bad'); return; }
        S.prep = P; S.result = null;
        const kMax = Math.min(Number($('kmax').value) || 8, P.rows.length - 1), ks = []; for (let k = 2; k <= kMax; k++) ks.push(k);
        $('explore-out').innerHTML = ''; $('explore-btn').disabled = true;
        const res = [];
        try {
            for (const k of ks) {
                status('explore-status', `Fitting ${k} clusters, with a quick stability check (${METHOD[P.method].short})…`); await yieldUI();
                const fit = fullFit(P, k);
                const sil = pooled(P, Q => CE.silhouette(Q.D, fit.labels).mean);
                const stab = CE.bootstrapStability(P.rows.length, fit.labels, resampleFn(P, k, 1), { B: 25, seed: P.seed, groups: P.groups });
                const sizes = new Array(k).fill(0); fit.labels.forEach(l => { sizes[l]++; });
                res.push({ k, labels: fit.labels, silhouette: sil, minJaccard: Math.min(...stab.perCluster.map(c => c.meanJaccard ?? 0)), meanJaccard: stab.perCluster.reduce((a, c) => a + (c.meanJaccard ?? 0), 0) / k,
                    ch: P.Z ? pooled(P, Q => CE.calinskiHarabasz(Q.Z, fit.labels)) : null, db: P.Z ? pooled(P, Q => CE.daviesBouldin(Q.Z, fit.labels)) : null, bic: fit.bic ?? null, entropy: fit.entropy ?? null, posterior: fit.posterior ?? null, model: fit.model ?? null, smallest: Math.min(...sizes), agreement: fit.agreement ?? null });
            }
            let gap = null;
            if (P.Z && (P.method === 'kmeans' || P.method === 'ward')) { status('explore-status', 'Gap statistic (comparison with reference data without clusters)…'); await yieldUI(); gap = CE.gapStatistic(P.Z, kMax, { B: 30, seed: P.seed, nInit: 3 }); }
            S.explore = { P, res, gap };
            renderExplore();
            status('explore-status', '');
        } catch (e) { status('explore-status', `The analysis stopped: ${e.message}`, 'bad'); }
        $('explore-btn').disabled = false;
    }
    // Suggested number of clusters: only stable solutions count (every cluster's mean bootstrap Jaccard >= 0.75);
    // among them the smallest k whose average silhouette is within 0.02 of the best (simpler is preferred when
    // the evidence is similar). The gap statistic (k-means, Ward) and BIC (mixtures) are shown alongside.
    function suggestion() {
        const { res, gap, P } = S.explore;
        if (gap && gap.kBest === 1) return { k: null, text: 'The gap statistic finds no more structure than in reference data without clusters (it chooses 1 cluster). Any clusters below may not be real.' };
        const stable = res.filter(r => r.minJaccard >= 0.75);
        if (!stable.length) return { k: null, text: 'No number of clusters gives stable clusters (every cluster reproducing in bootstrap resamples). The data may not contain distinct sub-groups; any clusters below should be treated as exploratory at best.' };
        // Model-based methods: the stable solution with the lowest BIC; otherwise the simplest stable solution with a
        // near-best silhouette
        const best = Math.max(...stable.map(r => r.silhouette));
        const pick = modelBased(P) ? stable.reduce((a, r) => (r.bic < a.bic ? r : a)) : stable.filter(r => r.silhouette >= best - 0.02).reduce((a, r) => (r.k < a.k ? r : a));
        const bicBest = modelBased(P) ? res.reduce((a, r) => (r.bic < a.bic ? r : a)) : null;
        const extra = modelBased(P) ? (bicBest.k !== pick.k ? ` Lowest BIC overall: ${bicBest.k} clusters, but that solution is not stable.` : '') + (P.method === 'lca' && pick.entropy < 0.6 ? ` Entropy is low (${pick.entropy.toFixed(2)}): the classes overlap heavily, and with related variables latent class analysis can create classes that are not real sub-groups.` : '') : gap ? ` Gap statistic: ${gap.kBest} clusters.` : '';
        const others = stable.filter(r => r.k !== pick.k).map(r => r.k);
        const also = others.length ? ` Also stable: ${others.join(', ')} clusters.${modelBased(P) ? '' : ' Average silhouette tends to favour 2 clusters,'} ${modelBased(P) ? 'Look' : 'so look'} at these too and choose the number that makes clinical sense.` : '';
        return { k: pick.k, text: `Suggested: ${pick.k} clusters, ${modelBased(P) ? 'the stable solution with the lowest BIC' : 'the simplest stable solution with a near-best average silhouette'}.${extra}${also}` };
    }
    function renderExplore() {
        const { res, gap, P } = S.explore, sug = suggestion(), f3 = v => (v === null || v === undefined || !Number.isFinite(v) ? '–' : v.toFixed(3));
        const stabWord = v => (v >= 0.75 ? 'stable' : v >= 0.6 ? 'some structure' : v >= 0.5 ? 'weak' : 'not reproducible');
        const rows = res.map(r => `<tr${r.k === sug.k ? ' class="pick"' : ''}><td>${r.k}${r.k === sug.k ? ' (suggested)' : ''}</td><td>${f3(r.silhouette)}</td><td>${f3(r.minJaccard)} <span class="muted">${stabWord(r.minJaccard)}</span></td>
            ${P.Z ? `<td>${f3(r.ch)}</td><td>${f3(r.db)}</td>` : ''}${gap ? `<td>${f3(gap.table[r.k - 1].gap)} ± ${f3(gap.table[r.k - 1].se)}</td>` : ''}${modelBased(P) ? `<td>${r.bic.toFixed(0)}</td>` : ''}${P.method === 'lca' ? `<td>${f3(r.entropy)}</td>` : ''}${P.mice ? `<td>${f3(r.agreement)}</td>` : ''}<td>${r.smallest}</td></tr>`).join('');
        const charts = CH.lineChart({ series: [{ name: 'Average silhouette', cls: 0, points: res.map(r => ({ x: r.k, y: r.silhouette })) }, { name: 'Stability (lowest cluster Jaccard)', cls: 1, points: res.map(r => ({ x: r.k, y: r.minJaccard })) }], xLabel: 'Number of clusters', yLabel: 'Value', aria: 'Silhouette and stability by number of clusters' })
            + CH.legend(['Average silhouette', 'Stability (lowest cluster Jaccard)']);
        $('explore-out').innerHTML = `
            <p class="lead">${esc(sug.text)} ${P.excluded ? `${P.excluded} patient(s) with missing values were left out.` : ''}${P.filled ? ` ${P.filled} missing value(s) were filled with the median or most frequent value.` : ''}${P.mice ? ` Missing values (${P.mice.missingValues}, in ${P.mice.missingPatients} patients) were multiply imputed: ${P.mice.m} imputed datasets by chained equations with predictive mean matching, from the clustering variables only; each dataset was clustered and the results combined into a consensus.${gap ? ' The gap statistic uses the first imputed dataset.' : ''}` : ''}</p>
            <div class="grid2"><div>${charts}</div><div class="muted small">
                <p><strong>Average silhouette</strong>: how clearly patients sit in their own cluster rather than the nearest other one (above 0.5 strong, 0.25–0.5 weak, below 0.25 little structure).</p>
                <p><strong>Stability</strong>: the clustering is repeated on 25 bootstrap resamples; the lowest mean Jaccard similarity of any cluster to its counterpart (0.75 or more: stable; below 0.5: that cluster does not reproduce).</p>
                ${modelBased(P) ? '<p><strong>BIC</strong>: fit of the model penalised for its complexity; lower is better, and differences under about 10 are not decisive.</p>' : ''}
                ${P.method === 'lca' ? '<p><strong>Entropy</strong>: how clearly patients are assigned to one class (1: every patient certainly in one class; above 0.8 good; below 0.6 classes overlap a lot).</p>' : ''}
                ${P.mice ? '<p><strong>Agreement across imputations</strong>: average adjusted Rand index between the clustering of each imputed dataset and the consensus (1: the imputed values make no difference; below about 0.65: the clusters depend on how the missing values are filled).</p>' : ''}
                ${gap ? '<p><strong>Gap</strong>: how much more compact the clusters are than in reference data without clusters; the suggestion follows Tibshirani\'s rule.</p>' : ''}
                <p>No single number decides: choose the smallest number of clusters that is stable and clinically interpretable.</p></div></div>
            <div class="table-wrap"><table><thead><tr><th>Clusters</th><th>Average silhouette</th><th>Stability</th>${P.Z ? '<th>Calinski–Harabasz ↑</th><th>Davies–Bouldin ↓</th>' : ''}${gap ? '<th>Gap ± SE</th>' : ''}${modelBased(P) ? '<th>BIC ↓</th>' : ''}${P.method === 'lca' ? '<th>Entropy ↑</th>' : ''}${P.mice ? '<th>Agreement across imputations</th>' : ''}<th>Smallest cluster</th></tr></thead><tbody>${rows}</tbody></table></div>
            <div class="row"><label for="k-choice">Describe</label><select id="k-choice">${res.map(r => `<option value="${r.k}"${r.k === (sug.k || 2) ? ' selected' : ''}>${r.k} clusters</option>`).join('')}</select><button id="describe-btn" class="btn primary">Describe these clusters</button></div>`;
        $('describe-btn').addEventListener('click', () => describe(Number($('k-choice').value)));
        enableTabs(4);
    }

    // ------------------------------------------------------------------------------------- step 5: the clusters
    async function describe(k) {
        const P = S.explore.P, ex = S.explore.res.find(r => r.k === k);
        goTo(5); enableTabs(5);
        $('use-out').innerHTML = '';
        $('clusters-out').innerHTML = '';
        try {
            status('clusters-status', `Checking stability on 100 bootstrap resamples…`); await yieldUI();
            const labels = ex.labels;
            const stab = CE.bootstrapStability(P.rows.length, labels, resampleFn(P, k, 7), { B: 100, seed: P.seed + 1, groups: P.groups });
            status('clusters-status', 'Testing whether the data have more structure than a single population…'); await yieldUI();
            let test = null;
            const testOne = Q => (P.allContinuous ? { ...CE.nullClusterTest(Q.Z, k, { B: 99, seed: P.seed }), kind: 'gaussian' } : { ...CE.copulaNullTest(Q.X, k, { B: 99, seed: P.seed }), kind: 'copula' });
            if (P.X && !P.impP) test = testOne(P);
            if (P.X && P.impP) {
                // One test per imputed dataset, combined by the median p-value (Eekhout et al. 2017)
                const tests = P.impP.map(testOne).sort((a, b) => a.pValue - b.pValue), mid = tests[Math.floor(tests.length / 2)];
                test = { ...mid, pooled: { m: tests.length, min: tests[0].pValue, max: tests[tests.length - 1].pValue } };
            }
            const sils = (P.impP || [P]).map(Q => CE.silhouette(Q.D, labels));
            const sil = { mean: sils.reduce((a, x) => a + x.mean, 0) / sils.length, values: labels.map((_, i) => sils.reduce((a, x) => a + x.values[i], 0) / sils.length) };
            const pca = P.Z ? CE.pca(P.Zmap || P.Z, 2) : null;
            S.result = { k, labels, stab, test, sil, pca, posterior: ex.posterior, model: ex.model, entropy: ex.entropy };
            renderClusters();
            status('clusters-status', '');
            enableTabs(6);
        } catch (e) { status('clusters-status', `The analysis stopped: ${e.message}`, 'bad'); }
    }
    // opts.only = 'describe': outcome variables only; opts.names(c): column heading for cluster c
    function profileTable(opts = {}) {
        const { P } = S.explore, { labels, k } = S.result, n = labels.length, head = opts.names || (c => `Cluster ${c + 1}`);
        const groups = Array.from({ length: k }, (_, c) => labels.map((l, i) => (l === c ? i : -1)).filter(i => i >= 0));
        const fmtNum = xs => { if (!xs.length) return '–'; const m = xs.reduce((a, b) => a + b, 0) / xs.length, sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, xs.length - 1)); return `${m.toFixed(Math.abs(m) < 10 ? 2 : 1)} (${sd.toFixed(Math.abs(m) < 10 ? 2 : 1)})`; };
        const q = (xs, f) => { const s = xs.slice().sort((a, b) => a - b); const h = (s.length - 1) * f, lo = Math.floor(h); return s[lo] + (s[Math.ceil(h)] - s[lo]) * (h - lo); };
        const fmtOrd = xs => (xs.length ? `${q(xs, 0.5)} [${q(xs, 0.25)}–${q(xs, 0.75)}]` : '–');
        const pf = p => (p === null || p === undefined ? '–' : p < 0.001 ? '< 0.001' : p.toFixed(3));
        const rowsOut = [];
        [...(opts.only === 'describe' ? [] : P.cols.map(c => [c, true])), ...P.describe.map(c => [c, false])].forEach(([col, used]) => {
            const t = S.vars[col].type, vals = P.rows.map(r => r[col]), ok = vals.map(v => !DE.isMissing(v));
            const missingN = ok.filter(x => !x).length;
            if (t === 'nominal' || (t === 'ordinal' && !vals.filter((v, i) => ok[i]).every(v => Number.isFinite(Number(v))))) {
                const levels = levelsOf(col), sv = vals.map(v => (DE.isMissing(v) ? null : String(v).trim()));
                const test = used ? null : CE.chiSquareTest(sv.filter(v => v !== null), labels.filter((_, i) => sv[i] !== null));
                levels.forEach((lv, li) => {
                    const cell = idx => { const okIdx = idx.filter(i => sv[i] !== null), c = okIdx.filter(i => sv[i] === lv).length; return okIdx.length ? `${c} (${(100 * c / okIdx.length).toFixed(0)}%)` : '–'; };
                    rowsOut.push(`<tr${used ? '' : ' class="describe"'}><td>${li === 0 ? `<strong>${esc(col)}</strong>${missingN ? ` <span class="muted">(${missingN} missing)</span>` : ''}` : ''}<div class="lvl">${esc(lv)}, n (%)</div></td>${groups.map(g => `<td>${cell(g)}</td>`).join('')}<td>${cell(labels.map((_, i) => i))}</td><td>${li === 0 ? (used ? '<span class="muted">used to form clusters</span>' : `${pf(test.p)}${test.minExpected < 5 ? ' <span class="muted">(small counts)</span>' : ''}`) : ''}</td></tr>`);
                });
            } else {
                const xs = idx => idx.filter(i => ok[i]).map(i => Number(vals[i]));
                const fmt = t === 'ordinal' ? fmtOrd : fmtNum;
                const okI = labels.map((_, i) => i).filter(i => ok[i]);
                const test = used ? null : CE.kruskalWallis(okI.map(i => Number(vals[i])), okI.map(i => labels[i]));
                rowsOut.push(`<tr${used ? '' : ' class="describe"'}><td><strong>${esc(col)}</strong>${missingN ? ` <span class="muted">(${missingN} missing)</span>` : ''}<div class="lvl">${t === 'ordinal' ? 'median [IQR]' : 'mean (SD)'}</div></td>${groups.map(g => `<td>${fmt(xs(g))}</td>`).join('')}<td>${fmt(xs(labels.map((_, i) => i)))}</td><td>${used ? '<span class="muted">used to form clusters</span>' : pf(test.p)}</td></tr>`);
            }
        });
        return `<div class="table-wrap"><table class="profile"><thead><tr><th>Variable</th>${groups.map((g, c) => `<th><i class="sw s${c}"></i>${esc(head(c))}<div class="muted">n = ${g.length}</div></th>`).join('')}<th>All<div class="muted">n = ${n}</div></th><th>p (describe-only variables)</th></tr></thead><tbody>${rowsOut.join('')}</tbody></table></div>
            <p class="muted small">${opts.only === 'describe' ? 'P-values: Kruskal–Wallis test (numbers, grades) or chi-square test (categories); with several variables, some small p-values occur by chance. Differences in outcome support, but do not prove, that the classes are clinically meaningful.</p>' : ''}${opts.only === 'describe' ? '<p hidden>' : '<p class="muted small">'}Variables used to form the clusters differ between clusters by construction, so no p-values are shown for them. For variables used only to describe the clusters (shaded), p-values come from the Kruskal–Wallis test (numbers, grades) or the chi-square test (categories); with several variables, some small p-values occur by chance.</p>`;
    }
    function renderClusters() {
        const { P } = S.explore, { k, labels, stab, test, sil, posterior } = S.result;
        const certainty = posterior ? Array.from({ length: k }, (_, c) => { const idx = labels.map((l, i) => (l === c ? i : -1)).filter(i => i >= 0); return idx.reduce((a, i) => a + posterior[i][c], 0) / Math.max(1, idx.length); }) : null;
        const sizes = new Array(k).fill(0); labels.forEach(l => { sizes[l]++; });
        const stabWord = v => (v >= 0.75 ? 'stable' : v >= 0.6 ? 'indicates a pattern' : v >= 0.5 ? 'weak' : 'not reproducible');
        const f3 = v => (Number.isFinite(v) ? v.toFixed(3) : '–'), pf = p => (p <= 0.01 ? '≤ 0.01' : p.toFixed(2));
        const weakest = Math.min(...stab.perCluster.map(c => c.meanJaccard ?? 0));
        const evidence = [];
        if (test) evidence.push(test.pValue <= 0.05 ? `more separation than ${test.kind === 'gaussian' ? 'a single Gaussian population with the same covariance' : 'a single population with the same variable distributions and correlations'} gives by chance (p ${pf(test.pValue)})` : `no more separation than ${test.kind === 'gaussian' ? 'a single Gaussian population' : 'a single population with the same distributions and correlations'} gives by chance (p = ${pf(test.pValue)})`);
        evidence.push(weakest >= 0.75 ? 'every cluster reproduces in bootstrap resamples' : weakest >= 0.5 ? 'at least one cluster reproduces only weakly in bootstrap resamples' : 'at least one cluster does not reproduce in bootstrap resamples');
        const verdict = (!test || test.pValue <= 0.05) && weakest >= 0.75 ? 'good' : (test && test.pValue > 0.05) || weakest < 0.5 ? 'bad' : 'mixed';
        $('clusters-out').innerHTML = `
            <div class="verdict ${verdict}"><strong>${k} clusters (${esc(METHOD[P.method].short)}).</strong> Evidence: ${esc(evidence.join('; '))}.
                ${verdict === 'good' ? 'These look like real sub-groups in these data; they still need replication in another cohort before clinical interpretation.' : verdict === 'bad' ? 'These clusters may be artefacts of the method: clustering algorithms always return clusters, even in data without sub-groups. Do not interpret them as patient sub-types.' : 'The evidence is mixed: treat the clusters as exploratory.'}</div>
            <div class="grid2">
                <div><h3>Clusters</h3><div class="table-wrap"><table><thead><tr><th>Cluster</th><th>Patients</th><th>Stability (mean Jaccard)</th><th>Dissolved in</th>${certainty ? '<th>Average membership probability</th>' : ''}</tr></thead><tbody>
                    ${stab.perCluster.map(c => `<tr><td><i class="sw s${c.cluster}"></i>Cluster ${c.cluster + 1}</td><td>${c.size} (${(100 * c.size / labels.length).toFixed(0)}%)</td><td>${f3(c.meanJaccard)} <span class="muted">${stabWord(c.meanJaccard)}</span></td><td>${(100 * c.dissolved).toFixed(0)}% of resamples</td>${certainty ? `<td>${f3(certainty[c.cluster])}</td>` : ''}</tr>`).join('')}</tbody></table></div>
                    ${certainty ? `<p class="muted small">Average membership probability: how certain the model is, on average, that the patients of a class belong to it (above 0.8: well separated). Entropy ${f3(S.result.entropy)}.</p>` : ''}
                    <p class="muted small">Stability: ${stab.resamples} bootstrap resamples${P.groups ? ` (whole patients by "${esc(P.patientCol)}")` : ''}; mean Jaccard 0.75 or more: stable; 0.6–0.75: indicates a pattern; below 0.5: not reproducible (Hennig 2007). Average agreement with the full-data clustering: adjusted Rand ${f3(stab.meanAdjustedRand)}.${P.mice ? ` The resamples cycle through the ${P.mice.m} imputed datasets. Agreement of each imputed dataset's clustering with the consensus: adjusted Rand ${f3(S.explore.res.find(r => r.k === k).agreement)}.` : ''}</p>
                    ${test ? `<p class="muted small">Test of "no real sub-groups": the separation of the clustering (R² = ${f3(test.observed)}) against ${test.B} reference datasets from ${test.kind === 'gaussian' ? 'a single Gaussian population with exactly the data\'s covariance (exact for Gaussian data)' : 'a single population with each variable\'s own distribution and the observed correlations (a Gaussian copula; conservative: a small p is strong evidence, a large p does not prove there are no sub-groups)'}: reference mean ${f3(test.nullMean)}, 95th centile ${f3(test.nullQuantile95)}.${test.pooled ? ` With multiple imputation the test was run on each of the ${test.pooled.m} imputed datasets and the median p-value is reported (range ${pf(test.pooled.min)} to ${pf(test.pooled.max)}); the figures above are from the median dataset.` : ''}</p>` : '<p class="muted small">The test of "no real sub-groups" needs complete data (choose to fill or leave out missing values).</p>'}
                </div>
                <div><h3>Map of the patients</h3><div id="map-panel"></div></div>
            </div>
            <h3>Dimensions of the data: what drives the differences between patients</h3>
            <div id="dims-panel"></div>
            <h3>How well each patient fits (silhouette)</h3>
            <div class="sil">${CH.silhouettePlot({ values: sil.values, labels, aria: 'Silhouette width of each patient by cluster' })}</div>
            <p class="muted small">Average silhouette ${f3(sil.mean)}${P.mice ? ' (averaged over the imputed datasets)' : ''}. Patients with negative widths sit closer to another cluster than to their own.</p>
            <h3>Clinical profile of the clusters</h3>
            ${profileTable()}
            <div class="row"><button id="to-use" class="btn primary" type="button">Next: use the clusters</button><button id="export-btn" class="btn" type="button">Download patients with their cluster (.CSV)</button></div>`;
        $('export-btn').addEventListener('click', exportCsv);
        $('to-use').addEventListener('click', () => goTo(6));
        window.UMEMaps.render({ S, P });
    }
    function exportCsv() {
        if (!confirm('The file will contain your patients\' data (every column of the loaded file) with the cluster of each patient. Save it only where your data-governance rules allow. Continue?')) return;
        const { P } = S.explore, labels = S.result.labels;
        const extra = window.UMEUse.exportColumns();
        const head = [...S.data.headers, 'Cluster', ...(extra ? ['Class', 'Class by bedside rules'] : [])];
        const cell = v => { let s = String(v ?? ''); if (/^[=+\-@\t\r]/.test(s) && !Number.isFinite(Number(s))) s = `'${s}`; return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
        const lines = [head.map(cell).join(','), ...P.rows.map((r, i) => [...S.data.headers.map(h => r[h]), labels[i] + 1, ...(extra ? [extra.names[i], extra.rules[i]] : [])].map(cell).join(','))];
        const url = URL.createObjectURL(new Blob([lines.join('\r\n')], { type: 'text/csv' }));
        const a = document.createElement('a'); a.href = url; a.download = `clusters_${String(S.data.name).replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_')}.csv`;
        document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    }

    // ------------------------------------------------------------------------------------- wiring
    document.addEventListener('DOMContentLoaded', () => {
        document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => goTo(t.dataset.step)));
        $('file').addEventListener('change', e => { const f = e.target.files[0]; if (!f) return; const r = new FileReader(); r.onload = () => loadText(String(r.result), f.name); r.readAsText(f); });
        $('demo-phenotypes').addEventListener('click', () => loadDemo('demo_inph_phenotypes.csv'));
        $('demo-none').addEventListener('click', () => loadDemo('demo_no_subtypes.csv'));
        $('to-step3').addEventListener('click', () => { enableTabs(3); goTo(3); });
        $('method').addEventListener('change', () => { S.methodTouched = true; updateVarSummary(); });
        $('explore-btn').addEventListener('click', () => { goTo(4); explore(); });
        $('theme').addEventListener('click', () => { const d = document.documentElement, dark = d.getAttribute('data-theme') !== 'dark' && !(d.getAttribute('data-theme') !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches); d.setAttribute('data-theme', dark ? 'dark' : 'light'); });
        enableTabs(1); goTo(1);
    });
    window.UMEApp = { state: S, prepare, loadText };
})();
