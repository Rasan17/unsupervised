// checks.js: step 3, "Check the data". Likely data-entry errors (column by column) and unusual patients (three
// methods on the clustering variables: robust Mahalanobis distance, isolation forest, local outlier factor), each
// with a plain reason, a map with the flagged patients, and the option to exclude patients from the analysis.
// Plain script (window.UMEChecks); every value from the data is escaped.
(function (root) {
    'use strict';
    const NN = root.NNEngine, AE = root.AnomalyEngine, CE = root.ClusterEngine, EE = root.EmbedEngine, DE = root.DataEngine, CH = root.UMECharts, esc = CH.esc;
    const $ = id => document.getElementById(id);
    const LIMITS = { mcd: 0.999, iso: 0.6, lof: 1.6 };
    // Local outlier factor has no natural scale: its limit is 1.6, raised where needed to the robust upper fence of
    // the scores (upper quartile + 3 interquartile ranges, on the log scale), as in large administrative datasets
    function lofLimit(lf) {
        const l = lf.filter(v => v > 0 && Number.isFinite(v)).map(Math.log).sort((a, b) => a - b), q = f => l[Math.floor(f * (l.length - 1))];
        return l.length >= 20 ? Math.max(LIMITS.lof, Math.exp(q(0.75) + 3 * (q(0.75) - q(0.25)))) : LIMITS.lof;
    }
    const ID_RE = /(^|[\s_-])(id|mrn|nhs|hospital[\s_-]?number|record|patient)([\s_-]?(id|no|number))?$/i;
    let C = null;   // { S, onChange, results }

    const isNum = v => v !== null && v !== undefined && String(v).trim() !== '' && Number.isFinite(Number(String(v).trim()));
    function idColumn(S) { return S.data.headers.find(h => S.vars[h].role === 'ignore' && ID_RE.test(h)) || S.data.headers.find(h => ID_RE.test(h)) || null; }
    function label(S, i) { const id = idColumn(S); const v = id ? String(S.data.rows[i][id] ?? '').trim() : ''; return v || `Row ${i + 1}`; }
    const pretty = s => String(s).replace(/_/g, ' ');
    const fmtV = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : Number(v.toPrecision(3)).toString());

    async function run(S) {
        const rows = S.data.rows, n = rows.length, id = idColumn(S);
        const used = S.data.headers.filter(h => S.vars[h].role !== 'ignore');
        const columns = [...(id ? [{ name: id, kind: 'id' }] : []), ...used.map(h => ({ name: h, kind: S.vars[h].type === 'nominal' || !rows.some(r => isNum(r[h])) ? 'category' : 'number' }))];
        const issues = AE.dataChecks(rows, columns);
        // Unusual patients, on the clustering variables
        const cl = S.data.headers.filter(h => S.vars[h].role === 'cluster');
        const numCols = cl.filter(h => S.vars[h].type !== 'nominal'), catCols = cl.filter(h => S.vars[h].type === 'nominal');
        const num = h => rows.map(r => (isNum(r[h]) ? Number(String(r[h]).trim()) : NaN));
        const numVals = Object.fromEntries(numCols.map(h => [h, num(h)]));
        // The multivariate methods use log(1 + x) for clearly right-skewed non-negative measurements (waiting times,
        // distances, counts), so that a long natural tail does not make records look unusual; reasons quote raw values
        const skewedCols = new Set(numCols.filter(h => S.vars[h].type === 'continuous' && AE.skewedNonNegative(numVals[h])));
        const mv = Object.fromEntries(numCols.map(h => [h, skewedCols.has(h) ? numVals[h].map(v => (Number.isFinite(v) ? Math.log1p(v) : NaN)) : numVals[h]]));
        const mvRaw = r => cl.map(h => { const v = r[h]; if (DE.isMissing(v)) return null; if (S.vars[h].type === 'continuous') { if (!isNum(v)) return null; const x = Number(String(v).trim()); return skewedCols.has(h) && x >= 0 ? Math.log1p(x) : x; } return String(v).trim(); });
        const rz = Object.fromEntries(numCols.map(h => [h, AE.robustZ(mv[h])]));
        const flags = Array.from({ length: n }, () => ({ mcd: null, iso: null, lof: null, ae: null }));
        const methods = [];
        // 1. Robust Mahalanobis distance (numbers and grades; patients with all of them recorded)
        let robust = null;
        const complete = rows.map((_, i) => i).filter(i => numCols.every(h => Number.isFinite(mv[h][i])));
        if (numCols.length >= 2 && complete.length >= Math.max(20, 5 * numCols.length)) {
            const X = complete.map(i => numCols.map(h => mv[h][i]));
            try {
                robust = AE.mcd(X, { seed: 2024 }); robust.cols = numCols; robust.rows = complete;
                const cut = AE.chi2Ppf(LIMITS.mcd, numCols.length);
                complete.forEach((i, k) => { flags[i].mcd = { value: robust.dist[k], flagged: robust.dist[k] > cut }; });
                methods.push({ key: 'mcd', name: 'Robust distance', note: `robust Mahalanobis distance (minimum covariance determinant) on the ${numCols.length} numbers and grades, for the ${complete.length} patients with all of them recorded; flagged above the 99.9% chi-square limit (${cut.toFixed(1)})` });
            } catch (e) { robust = null; }
        }
        // 2. Isolation forest (all clustering variables; missing numbers -> median, missing category -> most frequent)
        if (cl.length >= 2 && n >= 20) {
            const X = rows.map(() => []);
            numCols.forEach(h => { const med = rz[h].med; rows.forEach((_, i) => X[i].push(Number.isFinite(mv[h][i]) ? mv[h][i] : med)); });
            catCols.forEach(h => {
                const vals = rows.map(r => (DE.isMissing(r[h]) ? null : String(r[h]).trim())), cnt = new Map(); vals.forEach(v => { if (v !== null) cnt.set(v, (cnt.get(v) || 0) + 1); });
                const mode = [...cnt.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
                [...cnt.keys()].sort().forEach(l => rows.forEach((_, i) => X[i].push((vals[i] ?? mode) === l ? 1 : 0)));
            });
            const sc = AE.isolationForest(X, { seed: 2024 });
            sc.forEach((v, i) => { flags[i].iso = { value: v, flagged: v > LIMITS.iso }; });
            methods.push({ key: 'iso', name: 'Isolation', note: `isolation forest on all ${cl.length} clustering variables (100 trees); flagged when the anomaly score is above ${LIMITS.iso} (isolated much faster than the others)` });
        }
        // 3. Local outlier factor on balanced Gower dissimilarities (mixed data, missing values)
        if (cl.length >= 2 && n >= 25 && n > 3000) {
            // Large datasets: the same local outlier factor, with dissimilarities computed one record at a time
            const specs = cl.map(h => ({ type: S.vars[h].type, levels: S.vars[h].type === 'continuous' ? null : C.levelsOf(h) }));
            const raw = rows.map(mvRaw);
            const lf = await AE.lofStreamAsync(n, CE.gowerPairFn(raw, specs, CE.gowerBalancedWeights(raw, specs)), 20, (d, N) => status(`Checking the data: local density, record ${d.toLocaleString('en-GB')} of ${N.toLocaleString('en-GB')}…`)), lim = lofLimit(lf);
            lf.forEach((v, i) => { flags[i].lof = { value: v, flagged: v > lim }; });
            methods.push({ key: 'lof', name: 'Local density', note: `local outlier factor (20 neighbours) on balanced Gower dissimilarities of the clustering variables, computed record by record for the ${n.toLocaleString('en-GB')} records; flagged above ${lim.toFixed(2)} (in a much sparser region than its neighbours; the limit is 1.6, raised to the robust upper fence of these scores where that is higher)` });
            C.famdRaw = { raw, specs: cl.map((h, j) => ({ name: h, type: specs[j].type, levels: specs[j].levels })) };
        }
        if (cl.length >= 2 && n >= 25 && n <= 3000) {
            const specs = cl.map(h => ({ type: S.vars[h].type, levels: S.vars[h].type === 'continuous' ? null : C.levelsOf(h) }));
            const raw = rows.map(mvRaw);
            try {
                const G = CE.gowerMatrix(raw, specs, { weights: 'balanced' }), lf = AE.lof(G, 20), lim = lofLimit(lf);
                lf.forEach((v, i) => { flags[i].lof = { value: v, flagged: v > lim }; });
                methods.push({ key: 'lof', name: 'Local density', note: `local outlier factor (20 neighbours) on balanced Gower dissimilarities of the clustering variables; flagged above ${lim.toFixed(2)} (in a much sparser region than its neighbours${lim > LIMITS.lof ? '; the usual limit of 1.6 raised to the robust upper fence of these scores' : ''})` });
                C.famdRaw = { raw, specs: cl.map((h, j) => ({ name: h, type: specs[j].type, levels: specs[j].levels })) };
            } catch (e) { /* patients sharing no recorded variable: no local outlier factor */ }
        }
        // 4. Optional: autoencoder reconstruction error (deep learning); flagged above the upper fence Q3 + 3 IQR
        let aeWorst = null;
        if (C.useAE && cl.length >= 2 && n >= 50) {
            const vars = cl.map(h => ({ name: h, type: S.vars[h].type, levels: S.vars[h].type === 'continuous' ? null : C.levelsOf(h) }));
            const raw = rows.map(mvRaw);
            const coded = NN.codeData(raw, vars), T = NN.train(coded.X, coded.M, { hidden: [8], latent: 2, seed: 2024 }), E = NN.errors(T.net, coded.X, coded.M);
            const e = E.map(x => x.mean), srt = e.filter(Number.isFinite).sort((a, b) => a - b), q = f => srt[Math.min(srt.length - 1, Math.floor(f * (srt.length - 1)))], fence = q(0.75) + 3 * (q(0.75) - q(0.25));
            e.forEach((v, i) => { flags[i].ae = { value: v, flagged: v > fence }; });
            aeWorst = E.map(x => { const per = new Array(cl.length).fill(0); x.perVar.forEach((v, c) => { if (v !== null) per[coded.columns[c].variable] += v; }); return per.map((v, j) => [v, j]).sort((a, b) => b[0] - a[0]).slice(0, 2).map(([, j]) => pretty(cl[j])); });
            methods.push({ key: 'ae', name: 'Autoencoder', note: `autoencoder (a small neural network, ${coded.X[0].length}-8-2-8-${coded.X[0].length}) trained to rebuild the clustering variables; flagged when a patient is rebuilt much worse than the others (error above the upper quartile plus 3 interquartile ranges)` });
        }
        // Reasons: extreme single values, rare categories, otherwise the most unusual pair of numbers
        // Category shares and robust-distance positions, counted once (not per flagged patient)
        const catCount = Object.fromEntries(catCols.map(h => { const m = new Map(); rows.forEach(r => { const v = String(r[h] ?? '').trim(); m.set(v, (m.get(v) || 0) + 1); }); return [h, m]; }));
        const robustPos = robust ? new Map(robust.rows.map((r, k) => [r, k])) : null;
        const reason = i => {
            const parts = [];
            numCols.forEach(h => { const z = rz[h].z[i]; if (Math.abs(z) >= 3) parts.push(`${pretty(h)} ${numVals[h][i]} (typical ${fmtV(skewedCols.has(h) ? Math.expm1(rz[h].med) : rz[h].med)}; ${z > 0 ? 'very high' : 'very low'})`); });
            catCols.forEach(h => { const v = rows[i][h]; if (DE.isMissing(v)) return; const share = (catCount[h].get(String(v).trim()) || 0) / n; if (share < 0.03) parts.push(`${pretty(h)} "${String(v).trim()}" (only ${Math.round(share * n)} patient${Math.round(share * n) === 1 ? '' : 's'})`); });
            if (parts.length) return parts.join('; ');
            if (robust && numCols.length >= 2) {
                const k = robustPos.has(i) ? robustPos.get(i) : -1;
                if (k >= 0) {
                    let best = null;
                    for (let a = 0; a < numCols.length; a++) for (let b = a + 1; b < numCols.length; b++) {
                        const S2 = [[robust.covariance[a][a], robust.covariance[a][b]], [robust.covariance[b][a], robust.covariance[b][b]]], det = S2[0][0] * S2[1][1] - S2[0][1] * S2[1][0];
                        if (!(det > 0)) continue;
                        const x = [mv[numCols[a]][i] - robust.location[a], mv[numCols[b]][i] - robust.location[b]];
                        const d2 = (S2[1][1] * x[0] * x[0] - 2 * S2[0][1] * x[0] * x[1] + S2[0][0] * x[1] * x[1]) / det;
                        if (!best || d2 > best.d2) best = { a, b, d2 };
                    }
                    if (best) {
                        const ha = numCols[best.a], hb = numCols[best.b], pair = `${pretty(ha)} ${numVals[ha][i]} with ${pretty(hb)} ${numVals[hb][i]}`;
                        return flags[i].mcd && flags[i].mcd.flagged ? `no single value is extreme, but the combination of ${pair} is unusual in these data` : `no single value is extreme; the least typical pair of values is ${pair}`;
                    }
                }
            }
            return 'no single value is extreme; the overall pattern of values is unusual';
        };
        const unusual = flags.map((f, i) => ({ i, count: ['mcd', 'iso', 'lof', 'ae'].filter(k => f[k] && f[k].flagged).length, f })).filter(u => u.count > 0)
            .sort((a, b) => b.count - a.count || (b.f.lof?.value ?? 0) - (a.f.lof?.value ?? 0));
        unusual.forEach(u => { u.why = reason(u.i) + (aeWorst && u.f.ae && u.f.ae.flagged ? ` (rebuilt worst by the autoencoder: ${aeWorst[u.i].join(', ')})` : ''); });
        const active = methods.filter(m => !m.skipped);
        const allScores = Object.fromEntries(['mcd', 'iso', 'lof', 'ae'].map(k => [k, flags.map(f => (f[k] ? f[k].value : null)).filter(v => v !== null)]));
        return { issues, unusual, methods: active, notes: methods.filter(m => m.skipped), nMethods: active.length, allScores };
    }

    const MARKABLE = { negative: 'negative values', extreme: 'extreme values', decimal: 'possible decimal-point errors', rare: 'one-off categories', variant: 'possible spelling variants', code: 'possible "not recorded" codes' };
    // Flags marked as genuine (for the analysis record)
    function markedGenuine() {
        if (!C || !C.results || !C.S.markedGenuine) return [];
        return [...C.S.markedGenuine].map(k => { const [column, kind] = k.split('\u0001'); return { column, flag: MARKABLE[kind] || kind, values: C.results.issues.filter(x => x.column === column && x.kind === kind).length }; });
    }
    function render(ctx) {
        C = { ...ctx, results: C && C.S === ctx.S ? C.results : null };
        draw();
    }
    function draw() {
        const el = $('check-out'), { S } = C;
        if (!C.results) { el.innerHTML = ''; summary(); return; }
        const { unusual, methods, nMethods, notes = [] } = C.results;
        const ex = S.excluded;
        // Flags the user has marked as genuine for a whole variable (for example negative values of a change score)
        S.markedGenuine = S.markedGenuine || new Set();
        const gKey = x => `${x.column}\u0001${x.kind}`, issues = C.results.issues.filter(x => !S.markedGenuine.has(gKey(x)));
        const flagGroups = keys => [...keys.reduce((m, x) => m.set(gKey(x), [x, (m.get(gKey(x)) || [x, 0])[1] + 1]), new Map()).values()];
        const genuineBtns = flagGroups(issues.filter(x => MARKABLE[x.kind])).map(([x, c]) => `<button type="button" class="btn small" data-genuine="${esc(gKey(x))}">Genuine: ${esc(MARKABLE[x.kind])} in ${esc(x.column)} (${c})</button>`).join('');
        const marked = flagGroups(C.results.issues.filter(x => S.markedGenuine.has(gKey(x))));
        const box = i => `<input type="checkbox" class="excl" data-row="${i}" aria-label="Exclude ${esc(label(S, i))}"${ex.has(i) ? ' checked' : ''}>`;
        // "Exclude all" / "Include all" for the patients of one table
        const groups = { errors: [...new Set(issues.map(x => x.row))] };
        const bulk = (key, what) => `<div class="row bulk" role="group" aria-label="Exclude or include all ${what}"><button type="button" class="btn small" data-bulk="${key}" data-on="1">Exclude all ${what}</button><button type="button" class="btn small" data-bulk="${key}" data-on="0">Include all ${what}</button></div>`;
        const errRows = issues.map(x => `<tr><td>${box(x.row)}</td><td>${esc(label(S, x.row))}</td><td>${esc(x.column)}</td><td>${esc(x.value)}</td><td>${esc(x.message)}</td></tr>`).join('');
        const strong = unusual.filter(u => u.count >= Math.min(2, nMethods)), weak = unusual.filter(u => u.count < Math.min(2, nMethods));
        groups.strong = strong.map(u => u.i); groups.weak = weak.map(u => u.i);
        const tick = (u, k) => (u.f[k] ? (u.f[k].flagged ? '<strong>yes</strong>' : 'no') : '–');
        const uRow = u => `<tr><td>${box(u.i)}</td><td>${esc(label(S, u.i))}</td>${methods.map(m => `<td>${tick(u, m.key)}</td>`).join('')}<td>${esc(u.why)}</td></tr>`;
        const uHead = `<tr><th>Exclude</th><th>Patient</th>${methods.map(m => `<th>${esc(m.name)}</th>`).join('')}<th>Why</th></tr>`;
        // Map: FAMD of the clustering variables, flagged patients highlighted
        let map = '';
        if (C.famdRaw) {
            try {
                const F = EE.famd(C.famdRaw.raw, C.famdRaw.specs, { nDim: 2 });
                const flagged = new Set(strong.map(u => u.i)), errs = new Set(issues.map(x => x.row));
                // Large datasets: every flagged patient plus a random sample of the others (at most 3,000 points drawn)
                const nAll = F.scores.length, keep = new Set([...flagged, ...errs]);
                if (nAll > 3000) { const rng = CE.createPRNG(2024), rest = []; for (let i = 0; i < nAll; i++) if (!keep.has(i)) rest.push(i); for (let i = rest.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [rest[i], rest[j]] = [rest[j], rest[i]]; } rest.slice(0, Math.max(0, 3000 - keep.size)).forEach(i => keep.add(i)); }
                const shown = nAll > 3000 ? [...keep].sort((a2, b2) => a2 - b2) : F.scores.map((_, i) => i);
                map = CH.scatter({ points: shown.map(i => { const s = F.scores[i]; return { x: s[0], y: s[1], group: flagged.has(i) ? 1 : errs.has(i) ? 3 : null, tip: `${label(S, i)}${flagged.has(i) ? ': unusual' : ''}${errs.has(i) ? ': possible data error' : ''}` }; }), xLabel: `Dimension 1 (${(100 * F.ratio[0]).toFixed(0)}%)`, yLabel: `Dimension 2 (${(100 * F.ratio[1]).toFixed(0)}%)`, aria: 'Map of the patients with unusual patients highlighted' })
                    + '<div class="legend"><span><i class="sw s1"></i>unusual patient</span><span><i class="sw s3"></i>possible data error</span><span><i class="sw na"></i>other patients</span></div>'
                    + (nAll > 3000 ? `<p class="muted small">Every flagged patient is shown, with a random sample of the others (${shown.length.toLocaleString('en-GB')} of ${nAll.toLocaleString('en-GB')} points).</p>` : '');
            } catch (e) { map = ''; }
        }
        el.innerHTML = `
            <h3>Possible data errors (${issues.length})</h3>
            ${issues.length ? `${bulk('errors', `${groups.errors.length} patient(s) with possible errors`)}<div class="table-wrap"><table><thead><tr><th>Exclude</th><th>Patient</th><th>Variable</th><th>Value</th><th>Possible problem</th></tr></thead><tbody>${errRows}</tbody></table></div>
                <p class="muted small">Best corrected in the source file and loaded again; excluding the patient is the alternative. Values that cannot be read as numbers are treated as missing in the analysis.</p>
                ${genuineBtns ? `<details class="small"><summary>Values known to be genuine? Mark them for a whole variable</summary><p class="muted small">For example a change score that can be negative. Marked flags are no longer listed, and the analysis record notes them.</p><div class="row bulk" role="group" aria-label="Mark flags as genuine values">${genuineBtns}</div></details>` : ''}` : '<p>None found.</p>'}
            ${marked.length ? `<p class="small">Marked as genuine: ${marked.map(([x, c]) => `${esc(MARKABLE[x.kind] || x.kind)} in ${esc(x.column)} (${c})`).join('; ')}. <button type="button" class="btn small" id="genuine-undo">List them again</button></p>` : ''}
            <h3>Unusual patients (${strong.length}${nMethods >= 2 ? ' flagged by at least 2 of the ' + nMethods + ' methods' : ''})</h3>
            ${strong.length ? `${bulk('strong', `${strong.length} unusual patient(s)`)}<div class="table-wrap"><table><thead>${uHead}</thead><tbody>${strong.map(uRow).join('')}</tbody></table></div>` : '<p>None.</p>'}
            ${weak.length ? `<details><summary>${weak.length} patient(s) flagged by one method only</summary>${bulk('weak', `${weak.length} patient(s) flagged by one method`)}<div class="table-wrap"><table><thead>${uHead}</thead><tbody>${weak.map(uRow).join('')}</tbody></table></div></details>` : ''}
            <div class="grid2"><div>${map}</div><div class="muted small">${[...methods, ...notes].map(m => `<p><strong>${esc(m.name)}</strong>: ${esc(m.note)}.</p>`).join('')}
                <p>Unusual is not the same as wrong: check flagged patients against the records. Exclude only clear errors or patients outside the population of interest; rare but genuine patients may matter clinically. Report any exclusions.</p></div></div>`;
        el.querySelectorAll('[data-genuine]').forEach(b => b.addEventListener('click', () => { S.markedGenuine.add(b.dataset.genuine); draw(); }));
        const undo = $('genuine-undo'); if (undo) undo.addEventListener('click', () => { S.markedGenuine.clear(); draw(); });
        el.querySelectorAll('[data-bulk]').forEach(b => b.addEventListener('click', () => {
            const on = b.dataset.on === '1';
            groups[b.dataset.bulk].forEach(i => { if (on) S.excluded.add(i); else S.excluded.delete(i); el.querySelectorAll(`input.excl[data-row="${i}"]`).forEach(o => { o.checked = on; }); });
            summary(); C.onChange();
        }));
        el.querySelectorAll('input.excl').forEach(b => b.addEventListener('change', () => { const i = Number(b.dataset.row); if (b.checked) S.excluded.add(i); else S.excluded.delete(i); el.querySelectorAll(`input.excl[data-row="${i}"]`).forEach(o => { o.checked = b.checked; }); summary(); C.onChange(); }));
        summary();
    }
    function summary() {
        const { S } = C, ex = [...S.excluded].sort((a, b) => a - b), el = $('check-summary');
        el.innerHTML = ex.length ? `<p><strong>${ex.length} patient(s) will be left out of the analysis:</strong> ${esc(ex.map(i => label(S, i)).join(', '))}. <button id="excl-clear" class="btn small" type="button">Include them again</button></p>` : '<p class="muted">No patients excluded.</p>';
        const b = $('excl-clear'); if (b) b.addEventListener('click', () => { S.excluded.clear(); draw(); C.onChange(); });
    }
    async function start() {
        C.useAE = !!($('check-ae') && $('check-ae').checked);
        status(C.useAE ? 'Checking the data (training the autoencoder)…' : 'Checking the data…');
        await new Promise(r => setTimeout(r, 0));
        try { C.results = await run(C.S); status(''); } catch (e) { status(`The checks stopped: ${e.message}`); }
        draw();
    }
    function status(t) { const el = $('check-status'); if (el) el.textContent = t; }
    // Why a patient was listed (for the analysis record): data-error messages and the reason for being unusual
    function reasonsFor(i) {
        if (!C || !C.results) return [];
        const out = C.results.issues.filter(x => x.row === i && !(C.S.markedGenuine && C.S.markedGenuine.has(`${x.column}\u0001${x.kind}`))).map(x => `${x.column}: ${x.message}`);
        const u = C.results.unusual.find(x => x.i === i); if (u) out.push(`unusual (${u.count} method(s)): ${u.why}`);
        return out;
    }
    root.UMEChecks = { reasonsFor, markedGenuine, render, start, run: S => { C = C || { S, onChange: () => {} }; C.S = S; return run(S); }, label };
})(typeof window !== 'undefined' ? window : globalThis);
