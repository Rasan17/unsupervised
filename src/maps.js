// maps.js: the "Map of the patients" panel of step 5. PCA (or classical scaling of the dissimilarities when PCA
// is not possible), t-SNE and UMAP, each from the same dissimilarities as the clustering; patients coloured by
// cluster or by any variable; trustworthiness of each map and what it can and cannot show. Plain script
// (window.UMEMaps); every value from the data is escaped.
(function (root) {
    'use strict';
    const EE = root.EmbedEngine, CE = root.ClusterEngine, NN = root.NNEngine, CH = root.UMECharts, esc = CH.esc;
    const $ = id => document.getElementById(id);
    const yieldUI = () => new Promise(r => setTimeout(r, 0));
    const LIMIT = { tsne: 2000, umap: 5000, ae: 20000 };
    // Colour scale (viridis), readable on light and dark backgrounds
    const STOPS = ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'];
    function colourAt(t) {
        const x = Math.min(1, Math.max(0, t)) * (STOPS.length - 1), i = Math.min(STOPS.length - 2, Math.floor(x)), f = x - i;
        const h = s => [1, 3, 5].map(k => parseInt(s.slice(k, k + 2), 16)), a = h(STOPS[i]), b = h(STOPS[i + 1]);
        return '#' + a.map((v, k) => Math.round(v + (b[k] - v) * f).toString(16).padStart(2, '0')).join('');
    }
    const NAMES = { pca: 'PCA', famd: 'FAMD', mds: 'Classical scaling', tsne: 't-SNE', umap: 'UMAP', ae: 'Autoencoder' };
    const NOTES = {
        pca: 'Principal components of the standardised variables: a faithful projection, but only the two directions with the most variation are shown, so clusters can overlap here and still be distinct in other directions.',
        famd: 'Factor analysis of mixed data (FAMD, the counterpart of PCA for numbers and categories together): a faithful projection of the two main dimensions, so clusters can overlap here and still be distinct. The panel below shows which variables make up each dimension.',
        mds: 'Classical scaling of the dissimilarities used for clustering (the counterpart of PCA for mixed data): a faithful projection of the two main directions, so clusters can overlap here and still be distinct.',
        tsne: 't-SNE keeps each patient\'s close neighbours together but distorts everything else: the sizes of groups and the distances between them mean nothing, and it can show apparent groups even in data without sub-groups. Use it to see whether the clusters found are coherent, not as evidence that they exist.',
        ae: 'An autoencoder is a small neural network (deep learning) trained to compress each patient into two numbers and to rebuild all the variables from them: a non-linear generalisation of PCA. With a few hundred patients it can overfit, so training stops when the error on held-back patients stops improving (see the curve). Distances on this map are not to scale, and another random seed gives a somewhat different map.',
        umap: 'UMAP keeps close neighbours together and more of the overall layout than t-SNE, but still distorts sizes and distances, and apparent groups are not evidence of sub-groups. Use it to see whether the clusters found are coherent.'
    };
    let HL = null;  // records highlighted from the self-organising map (original row numbers), or null
    let M = null;   // { key, kind, colourBy, maps: { kind: { Y, idx, trust, xLabel, yLabel } }, busy }

    // The clustering dissimilarities; for large datasets they cover a random sample, whose original row numbers are
    // in M.Didx (identity otherwise)
    function dissim(P) {
        if (M.D) return M.D;
        M.Didx = P.sampleIdx || P.rows.map((_, i) => i);
        if (!P.impP) { M.D = P.D; return M.D; }
        // With multiple imputation: the dissimilarities averaged over the imputed datasets
        const n = P.D.n, D = new Float64Array(n * n);
        P.impP.forEach(Q => { for (let i = 0; i < n * n; i++) D[i] += Q.D.D[i] / P.impP.length; });
        M.D = { n, D };
        return M.D;
    }
    function subset(D, idx) {
        if (idx.length === D.n) return D;
        const m = idx.length, S = new Float64Array(m * m);
        for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) S[a * m + b] = D.D[idx[a] * D.n + idx[b]];
        return { n: m, D: S };
    }
    function sampleIdx(n, max, seed) {
        const all = Array.from({ length: n }, (_, i) => i);
        if (n <= max) return all;
        const rng = CE.createPRNG(seed);
        for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [all[i], all[j]] = [all[j], all[i]]; }
        return all.slice(0, max).sort((a, b) => a - b);
    }
    const kTrust = n => Math.max(1, Math.min(10, Math.floor(n / 2) - 1));

    function render(ctx) {
        const { S, P } = ctx;
        if (!M || M.key !== S.result) {
            M = { key: S.result, ctx, kind: null, colourBy: '__cluster', maps: {}, D: null, busy: false };
            const n = P.rows.length, all = Array.from({ length: n }, (_, i) => i);
            M.famd = famdOf(P);
            if (S.result.pca && P.allContinuous) {
                const pca = S.result.pca;
                M.maps.pca = { Y: pca.scores.map(s => [s[0], s[1]]), idx: all, xLabel: `Principal component 1 (${(100 * pca.varianceRatio[0]).toFixed(0)}% of variance)`, yLabel: `Component 2 (${(100 * pca.varianceRatio[1]).toFixed(0)}%)` };
                M.kind = 'pca';
            } else if (M.famd && M.famd.nDim >= 2) {
                const F = M.famd;
                M.maps.famd = { Y: F.scores.map(s => [s[0], s[1]]), idx: all, xLabel: `Dimension 1 (${(100 * F.ratio[0]).toFixed(0)}% of variance)`, yLabel: `Dimension 2 (${(100 * F.ratio[1]).toFixed(0)}%)` };
                M.kind = 'famd';
            } else {
                const D = dissim(P);
                if (D) { M.maps.mds = { Y: EE.classicalMDS(D, 2), idx: M.Didx.slice(), xLabel: 'Dimension 1', yLabel: 'Dimension 2' }; M.kind = 'mds'; }
            }
        }
        M.ctx = ctx;
        draw();
        drawDimensions();
    }
    // Trustworthiness on the patients that have dissimilarities (all, or the random sample), at most 2000
    function trustOf(map) {
        if (map.trust === undefined) {
            const D0 = dissim(M.ctx.P), posOf = new Map(M.Didx.map((o, q) => [o, q])), rowsIn = [];
            map.idx.forEach((o, r) => { if (posOf.has(o) && rowsIn.length < 2000) rowsIn.push(r); });
            const D = subset(D0, rowsIn.map(r => posOf.get(map.idx[r])));
            map.trust = EE.trustworthiness(D, rowsIn.map(r => map.Y[r]), kTrust(D.n)); map.k = kTrust(D.n);
        }
        return map.trust;
    }
    async function compute(kind) {
        if (M.busy) return;
        const P = M.ctx.P, D0 = dissim(P);
        if (!D0) return;
        M.busy = true;
        // t-SNE and UMAP work from the dissimilarities (positions within them); the autoencoder from the records
        const pos = kind === 'ae' ? null : sampleIdx(D0.n, LIMIT[kind], P.seed), D = pos ? subset(D0, pos) : null;
        const idx = pos ? pos.map(q => M.Didx[q]) : sampleIdx(P.rows.length, LIMIT.ae, P.seed);
        const status = $('map-status');
        try {
            if (kind === 'tsne') {
                const t = EE.createTSNE(EE.squared(D), { perplexity: 30 });
                while (!t.done) { t.step(25); status.textContent = `t-SNE: iteration ${t.iter} of ${t.maxIter}…`; await yieldUI(); }
                M.maps.tsne = { Y: t.Y, idx, xLabel: 't-SNE 1', yLabel: 't-SNE 2', extra: `perplexity ${Math.round(t.perplexity)}, divergence ${t.kl.toFixed(3)}` };
            } else if (kind === 'ae') {
                const vars = P.specs.map(s => ({ name: s.col, type: s.type, levels: s.levels })), coded = NN.codeData(idx.map(i => P.raw[i]), vars);
                const t = NN.createTrainer(coded.X, coded.M, { hidden: [16], latent: 2, seed: P.seed });
                while (!t.done) { t.step(10); status.textContent = `Autoencoder: training, epoch ${t.epoch} (stops when the error on held-back patients no longer improves)…`; await yieldUI(); }
                const net = t.net, Y = coded.X.map(x => NN.encode(net, x));
                let sse = 0, sst = 0; coded.X.forEach((x, i) => { const y = NN.reconstruct(net, x); x.forEach((v, j) => { if (coded.M[i][j]) { sse += (y[j] - v) ** 2; sst += v * v; } }); });
                const r2 = 1 - sse / sst, F = M.famd, ref = F && F.ratio.length >= 2 ? F.ratio[0] + F.ratio[1] : null;
                M.maps.ae = { Y, idx, xLabel: 'Autoencoder 1', yLabel: 'Autoencoder 2', history: t.history, bestEpoch: t.bestEpoch,
                    extra: `network ${coded.X[0].length}-16-2-16-${coded.X[0].length}, best after ${t.bestEpoch} epochs; it reconstructs ${(100 * r2).toFixed(0)}% of the variation${ref ? `, against ${(100 * ref).toFixed(0)}% for the first two ${P.allContinuous ? 'principal components' : 'FAMD dimensions'}` : ''}` };
            } else {
                const u = EE.createUMAP(D, { nNeighbors: Math.min(15, D.n - 1), minDist: 0.1, seed: P.seed });
                while (!u.done) { u.step(20); status.textContent = `UMAP: epoch ${u.epoch} of ${u.nEpochs}…`; await yieldUI(); }
                M.maps.umap = { Y: u.Y, idx, xLabel: 'UMAP 1', yLabel: 'UMAP 2', extra: `${Math.min(15, D.n - 1)} neighbours, minimum distance 0.1` };
            }
            status.textContent = 'Measuring how faithful the map is…'; await yieldUI();
            trustOf(M.maps[kind]);
            M.kind = kind; status.textContent = '';
        } catch (e) { status.textContent = `The map could not be made: ${e.message}`; }
        M.busy = false;
        draw();
    }
    // Colouring: cluster, or a variable (numbers and numeric grades on a colour scale; categories by colour)
    function colouring(map) {
        const { S, P } = M.ctx, labels = S.result.labels;
        if (M.colourBy === '__cluster') {
            const k = S.result.k;
            return { pts: i => ({ group: labels[i], tip: `Cluster ${labels[i] + 1}` }), legend: CH.legend(Array.from({ length: k }, (_, c) => `Cluster ${c + 1}`)) };
        }
        const col = M.colourBy, vals = P.rows.map(r => r[col]), miss = v => v === null || v === undefined || String(v).trim() === '';
        const recorded = vals.filter(v => !miss(v)), numeric = recorded.length && recorded.every(v => Number.isFinite(Number(v))) && S.vars[col].type !== 'nominal';
        if (numeric) {
            const xs = map.idx.map(i => vals[i]).filter(v => !miss(v)).map(Number), lo = Math.min(...xs), hi = Math.max(...xs);
            return { pts: i => (miss(vals[i]) ? { group: null, tip: `${col}: not recorded` } : { color: colourAt(hi > lo ? (Number(vals[i]) - lo) / (hi - lo) : 0.5), tip: `${col}: ${vals[i]} (cluster ${labels[i] + 1})` }),
                legend: CH.scaleLegend(STOPS, String(lo), String(hi), col) };
        }
        const levels = [...new Set(recorded.map(v => String(v).trim()))].sort();
        const shown = levels.slice(0, 7), other = levels.length > 7;
        const g = v => { const l = shown.indexOf(String(v).trim()); return l >= 0 ? l : 7; };
        return { pts: i => (miss(vals[i]) ? { group: null, tip: `${col}: not recorded` } : { group: g(vals[i]), tip: `${col}: ${String(vals[i]).trim()} (cluster ${labels[i] + 1})` }),
            legend: CH.legend([...shown, ...(other ? ['other'] : [])]) };
    }
    function draw() {
        const el = $('map-panel'); if (!el) return;
        const { P } = M.ctx, n = P.rows.length;
        const kinds = [M.maps.pca ? 'pca' : M.maps.famd ? 'famd' : 'mds', 'tsne', 'umap', 'ae'];
        const vars = [...P.cols, ...P.describe];
        const map = M.maps[M.kind];
        let body = '<p class="muted">No map is available for these data.</p>';
        if (map) {
            const c = colouring(map);
            const svg = CH.scatter({ points: map.idx.map((i, r) => ({ x: map.Y[r][0], y: map.Y[r][1], ...c.pts(i), row: i, ...(HL ? { mark: HL.has(i) ? 'hl' : 'dim' } : {}) })), xLabel: map.xLabel, yLabel: map.yLabel, noAxes: M.kind === 'tsne' || M.kind === 'umap' || M.kind === 'ae', aria: `${NAMES[M.kind]} map of the patients` });
            const tr = trustOf(map);
            const hlShown = HL ? map.idx.filter(i => HL.has(i)).length : 0;
            body = `${svg}${c.legend}
                ${M.ctx.S.result.neural && M.ctx.S.result.neural.unitLabels ? '<p class="muted small">Select patients here: click a point, drag a rectangle, or hold Option (Alt) and drag (with either mouse button) for a freehand outline. Hold Shift to add, Cmd (Ctrl on Windows) to remove. Deselect: click an empty area, press Escape, or "Clear selection". The self-organising map below shows where they are.</p>' : ''}
                ${HL ? `<p class="small note">Highlighted: the ${HL.size.toLocaleString('en-GB')} record(s) selected ${root.UMESom && root.UMESom._state() && root.UMESom._state().rec ? 'here' : 'on the self-organising map below'}${hlShown < HL.size ? ` (${hlShown.toLocaleString('en-GB')} of them are on this map, which shows a sample)` : ''}. <button type="button" class="btn small" id="map-clear">Clear selection</button></p>` : ''}
                ${map.history ? `<details><summary class="small">Training curve</summary>${CH.lineChart({ series: [{ name: 'Training patients', cls: 0, points: map.history.map(h => ({ x: h.epoch, y: h.train })) }, { name: 'Held-back patients', cls: 1, points: map.history.map(h => ({ x: h.epoch, y: h.val })) }], xLabel: 'Epoch', yLabel: 'Reconstruction error', aria: 'Training curve of the autoencoder', H: 220 })}${CH.legend(['Training patients', 'Held-back patients'])}<p class="muted small">If the error on held-back patients rises while the training error keeps falling, the network is learning the noise of these patients (overfitting); the weights from the lowest held-back error are used.</p></details>` : ''}
                ${M.kind === 'pca' || M.kind === 'famd' ? axisFactors() : '<p class="muted small">The axes of this map have no fixed meaning (it keeps neighbours, not directions); for what drives the differences between patients, see the PCA / FAMD map or "Dimensions of the data".</p>'}
                <p class="muted small"><strong>${NAMES[M.kind]}</strong>${map.extra ? ` (${esc(map.extra)})` : ''}. ${esc(NOTES[M.kind])}</p>
                <p class="muted small">Trustworthiness ${tr.toFixed(3)}: how well the map keeps each patient's ${map.k} nearest neighbours (1 = perfectly; above 0.9 good).${map.idx.length < n ? ` Map of a random sample of ${map.idx.length} of the ${n} patients.` : ''}</p>`;
        }
        el.innerHTML = `
            <div class="row map-controls" role="group" aria-label="Type of map">${kinds.map(k => `<button type="button" class="btn small${M.kind === k ? ' primary' : ''}" data-map="${k}" aria-pressed="${M.kind === k}"${M.busy ? ' disabled' : ''}>${NAMES[k]}${(k === 'tsne' || k === 'umap' || k === 'ae') && !M.maps[k] ? ' (make)' : ''}</button>`).join('')}
                <label for="map-colour" class="small">Colour by</label><select id="map-colour"><option value="__cluster">Cluster</option>${vars.map(v => `<option value="${esc(v)}"${M.colourBy === v ? ' selected' : ''}>${esc(v)}</option>`).join('')}</select></div>
            <p id="map-status" class="status small" role="status"></p>
            ${body}`;
        // With a self-organising map: clicking a patient selects its unit there (Shift / Cmd / Ctrl adds)
        const somOn = root.UMESom && root.UMESom.selectRecord && M.ctx.S.result.neural && M.ctx.S.result.neural.unitLabels;
        const svgEl = el.querySelector('svg.chart');
        if (somOn && svgEl) { svgEl.classList.add('pickable'); wireRegion(svgEl); }
        const mc = el.querySelector('#map-clear'); if (mc) mc.addEventListener('click', () => root.UMESom.clear());
        el.querySelectorAll('[data-map]').forEach(b => b.addEventListener('click', () => { const k = b.dataset.map; if (M.maps[k]) { M.kind = k; draw(); } else compute(k); }));
        $('map-colour').addEventListener('change', e => { M.colourBy = e.target.value; draw(); });
    }
    // ---------------------------------------------------------------------------------- dimensions of the data
    function famdOf(P) {
        try { return EE.famd(P.raw, P.specs.map(s => ({ name: s.col, type: s.type, levels: s.levels })), { nDim: 10 }); } catch (e) { return null; }
    }
    const pretty = s => String(s).replace(/_/g, ' ');
    function drawDimensions() {
        const el = $('dims-panel'); if (!el) return;
        const { S, P } = M.ctx, F = M.famd;
        if (!F || F.nDim < 1) { el.innerHTML = '<p class="muted">Dimensions need at least two clustering variables.</p>'; return; }
        const labels = S.result.labels, k = S.result.k, n = labels.length;
        const nCat = P.specs.filter(s => s.type === 'nominal').length, method = !nCat ? 'PCA (principal component analysis)' : nCat === P.specs.length ? 'MCA (multiple correspondence analysis, computed as FAMD)' : 'FAMD (factor analysis of mixed data)';
        const pct = x => `${(100 * x).toFixed(0)}%`;
        const shown = Math.min(F.nDim, 10), dims = Math.min(3, F.nDim), avgContrib = 100 / P.specs.length;
        // How strongly each dimension separates the clusters: correlation ratio of the cluster labels
        const eta = s => { const F_ = F.scores.map(r => r[s]), m = F_.reduce((a, b) => a + b, 0) / n, tot = F_.reduce((a, b) => a + (b - m) ** 2, 0); let between = 0; for (let c = 0; c < k; c++) { const idx = labels.map((l, i) => (l === c ? i : -1)).filter(i => i >= 0); if (!idx.length) continue; const mc = idx.reduce((a, i) => a + F_[i], 0) / idx.length; between += idx.length * (mc - m) ** 2; } return tot > 0 ? between / tot : 0; };
        const word = e => (e >= 0.5 ? 'strongly' : e >= 0.2 ? 'moderately' : e >= 0.05 ? 'slightly' : 'hardly');
        const describe = s => {
            const top = P.specs.map((sp, j) => ({ sp, j, c: F.contrib[j][s] })).filter(x => x.c >= avgContrib).sort((a, b) => b.c - a.c).slice(0, 4);
            const pos = [], neg = [];
            top.forEach(({ sp, j }) => {
                if (F.corr[j]) (F.corr[j][s] >= 0 ? pos : neg).push(`higher ${pretty(sp.col)}`);
                else { const cm = F.catMeans[j].slice().sort((a, b) => b.coord[s] - a.coord[s]); pos.push(`${pretty(sp.col)} "${cm[0].level}"`); neg.push(`${pretty(sp.col)} "${cm[cm.length - 1].level}"`); }
            });
            const e = eta(s);
            return `<li><strong>Dimension ${s + 1}</strong> (${pct(F.ratio[s])} of the variation): mainly ${esc(top.map(x => pretty(x.sp.col)).join(', ') || 'no single variable')}. ${pos.length ? `Towards one end: ${esc(pos.join(', '))}` : ''}${pos.length && neg.length ? '; towards the other: ' : ''}${neg.length ? esc(neg.join(', ')) : ''}. It separates the clusters ${word(e)} (η² ${e.toFixed(2)}).</li>`;
        };
        const scree = CH.lineChart({ series: [{ name: 'Share of the variation', cls: 0, points: F.ratio.slice(0, shown).map((r, i) => ({ x: i + 1, y: 100 * r })) }, { name: 'Equal share (no structure)', cls: 1, points: F.ratio.slice(0, shown).map((_, i) => ({ x: i + 1, y: 100 / F.eig.length })) }], xLabel: 'Dimension', yLabel: '% of the variation', aria: 'Share of the variation by dimension' }) + CH.legend(['Share of the variation', 'Equal share (no structure)']);
        const cum2 = F.ratio[0] + (F.ratio[1] || 0);
        const table = `<div class="table-wrap"><table><thead><tr><th>Variable</th>${Array.from({ length: dims }, (_, s) => `<th>Dimension ${s + 1}</th>`).join('')}</tr></thead><tbody>${P.specs.map((sp, j) => `<tr><td>${esc(sp.col)}</td>${Array.from({ length: dims }, (_, s) => { const c = F.contrib[j][s]; return `<td${c >= avgContrib ? ' class="diag"' : ''}>${c.toFixed(0)}%</td>`; }).join('')}</tr>`).join('')}</tbody></table></div>`;
        el.innerHTML = `<p class="small">${esc(method)} of the clustering variables. The first two dimensions explain <strong>${pct(cum2)}</strong> of the variation between patients${cum2 < 0.3 ? ': the variation is spread over many directions, with no dominant pattern' : ''}.</p>
            <div class="grid2"><div>${scree}</div><div><ol class="small dims">${Array.from({ length: dims }, (_, s) => describe(s)).join('')}</ol></div></div>
            <h4>Contribution of each variable to the dimensions</h4>${table}
            <p class="muted small">Contributions add up to 100% for each dimension; bold: above the average share (${avgContrib.toFixed(0)}%). Numbers are standardised; categories are coded as in MCA; grades are treated as numbers.${P.mice ? ' With multiple imputation the first imputed dataset is used.' : P.raw.some(r => r.some(v => v === null)) ? ' Missing values are replaced by the mean (numbers) or by the category shares.' : ''} If the clusters lie next to each other along a dimension without gaps (see the map), patients may differ by degree along a continuum rather than form distinct types.</p>`;
    }
    // Selecting patients on the map: click a point (its unit on the self-organising map); drag for a rectangle; Option
    // (Alt) + drag, or Option + right-button drag, for a freehand outline. Shift / Cmd / Ctrl adds to the selection.
    function wireRegion(svg) {
        const NS = 'http://www.w3.org/2000/svg';
        const pt = e => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; const q = p.matrixTransform(svg.getScreenCTM().inverse()); return [q.x, q.y]; };
        let drag = null;
        svg.addEventListener('contextmenu', e => { if (e.altKey || drag) e.preventDefault(); });
        svg.addEventListener('pointerdown', e => {
            if (e.button !== 0 && !(e.button === 2 && e.altKey)) return;
            const shape = document.createElementNS(NS, e.altKey ? 'path' : 'rect');
            shape.setAttribute('class', 'band');
            svg.appendChild(shape);
            drag = { start: pt(e), pts: [pt(e)], lasso: e.altKey, mode: e.metaKey || e.ctrlKey ? 'remove' : e.shiftKey ? 'add' : 'replace', shape, target: e.target };
            if (drag.mode === 'remove') shape.classList.add('remove');
            svg.setPointerCapture(e.pointerId); e.preventDefault();
        });
        svg.addEventListener('pointermove', e => {
            if (!drag) return;
            const p = pt(e); drag.pts.push(p);
            if (drag.lasso) drag.shape.setAttribute('d', 'M' + drag.pts.map(q => q.map(v => v.toFixed(1)).join(',')).join(' L') + ' Z');
            else { const [x0, y0] = drag.start; drag.shape.setAttribute('x', Math.min(x0, p[0])); drag.shape.setAttribute('y', Math.min(y0, p[1])); drag.shape.setAttribute('width', Math.abs(p[0] - x0)); drag.shape.setAttribute('height', Math.abs(p[1] - y0)); }
        });
        const finish = e => {
            if (!drag) return;
            const d = drag; drag = null; d.shape.remove();
            // A drag, not a click, if the pointer went more than 3 units from where it started at any time (a closed
            // freehand outline ends where it began)
            const end = pt(e), moved = d.pts.concat([end]).some(q => Math.hypot(q[0] - d.start[0], q[1] - d.start[1]) > 3);
            if (!moved) {
                const t = d.target;
                // A click on a point selects (or with Shift / Cmd / Ctrl toggles) its unit; a click on empty space clears
                if (t && t.dataset && t.dataset.row !== undefined) root.UMESom.selectRecord(Number(t.dataset.row), d.mode !== 'replace'); else root.UMESom.clear();
                return;
            }
            const inside = d.lasso ? (x, y) => { let c = false; for (let i = 0, j = d.pts.length - 1; i < d.pts.length; j = i++) { const [xi, yi] = d.pts[i], [xj, yj] = d.pts[j]; if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c; } return c; }
                : (x, y) => x >= Math.min(d.start[0], end[0]) && x <= Math.max(d.start[0], end[0]) && y >= Math.min(d.start[1], end[1]) && y <= Math.max(d.start[1], end[1]);
            const rows = [...svg.querySelectorAll('circle[data-row]')].filter(c => inside(Number(c.getAttribute('cx')), Number(c.getAttribute('cy')))).map(c => Number(c.dataset.row));
            root.UMESom.selectRecords(rows, d.mode);
        };
        svg.addEventListener('pointerup', finish);
        svg.addEventListener('pointercancel', () => { if (drag) { drag.shape.remove(); drag = null; } });
    }
    // The three variables that contribute most to each axis of the PCA / FAMD map, with their contribution and direction
    function axisFactors() {
        const F = M.famd, P = M.ctx.P;
        if (!F || F.nDim < 2) return '';
        const rows = [0, 1].map(s => {
            const top = P.specs.map((sp, j) => ({ sp, j, c: F.contrib[j][s] })).sort((a, b) => b.c - a.c).slice(0, 3);
            const dirWord = s === 0 ? ['→ right', '← left'] : ['↑ up', '↓ down'];
            const cells = top.map(({ sp, j, c }) => {
                let dir;
                if (F.corr[j]) dir = `higher ${F.corr[j][s] >= 0 ? dirWord[0] : dirWord[1]}`;
                else { const cm = F.catMeans[j].slice().sort((a, b) => b.coord[s] - a.coord[s]); dir = `"${cm[0].level}" ${dirWord[0]}`; }
                return `<td><strong>${esc(pretty(sp.col))}</strong> ${c.toFixed(0)}%<div class="muted">${esc(dir)}</div></td>`;
            }).join('');
            return `<tr><td>Dimension ${s + 1} <span class="muted">(${s === 0 ? 'horizontal' : 'vertical'})</span><div class="muted">${(100 * F.ratio[s]).toFixed(0)}% of the variation</div></td>${cells}</tr>`;
        }).join('');
        return `<h4>What the axes mean</h4><div class="table-wrap"><table class="axis-factors"><thead><tr><th>Axis</th><th>1st variable</th><th>2nd</th><th>3rd</th></tr></thead><tbody>${rows}</tbody></table></div>
            <p class="muted small">Percentages: each variable's contribution to that dimension (contributions of all variables add up to 100%). Direction: where patients with higher values of a number, or in that category, lie on the map. The full table is under "Dimensions of the data".</p>`;
    }
    function highlight(set) { HL = set && set.size ? set : null; if (M) draw(); }
    root.UMEMaps = { highlight, render, renderDimensions: () => drawDimensions(), _state: () => M };
})(typeof window !== 'undefined' ? window : globalThis);
