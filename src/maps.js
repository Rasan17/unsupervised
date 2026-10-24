// maps.js: the "Map of the patients" panel of step 5. PCA (or classical scaling of the dissimilarities when PCA
// is not possible), t-SNE and UMAP, each from the same dissimilarities as the clustering; patients coloured by
// cluster or by any variable; trustworthiness of each map and what it can and cannot show. Plain script
// (window.UMEMaps); every value from the data is escaped.
(function (root) {
    'use strict';
    const EE = root.EmbedEngine, CE = root.ClusterEngine, CH = root.UMECharts, esc = CH.esc;
    const $ = id => document.getElementById(id);
    const yieldUI = () => new Promise(r => setTimeout(r, 0));
    const LIMIT = { tsne: 2000, umap: 5000 };
    // Colour scale (viridis), readable on light and dark backgrounds
    const STOPS = ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'];
    function colourAt(t) {
        const x = Math.min(1, Math.max(0, t)) * (STOPS.length - 1), i = Math.min(STOPS.length - 2, Math.floor(x)), f = x - i;
        const h = s => [1, 3, 5].map(k => parseInt(s.slice(k, k + 2), 16)), a = h(STOPS[i]), b = h(STOPS[i + 1]);
        return '#' + a.map((v, k) => Math.round(v + (b[k] - v) * f).toString(16).padStart(2, '0')).join('');
    }
    const NAMES = { pca: 'PCA', mds: 'Classical scaling', tsne: 't-SNE', umap: 'UMAP' };
    const NOTES = {
        pca: 'Principal components of the standardised variables: a faithful projection, but only the two directions with the most variation are shown, so clusters can overlap here and still be distinct in other directions.',
        mds: 'Classical scaling of the dissimilarities used for clustering (the counterpart of PCA for mixed data): a faithful projection of the two main directions, so clusters can overlap here and still be distinct.',
        tsne: 't-SNE keeps each patient\'s close neighbours together but distorts everything else: the sizes of groups and the distances between them mean nothing, and it can show apparent groups even in data without sub-groups. Use it to see whether the clusters found are coherent, not as evidence that they exist.',
        umap: 'UMAP keeps close neighbours together and more of the overall layout than t-SNE, but still distorts sizes and distances, and apparent groups are not evidence of sub-groups. Use it to see whether the clusters found are coherent.'
    };
    let M = null;   // { key, kind, colourBy, maps: { kind: { Y, idx, trust, xLabel, yLabel } }, busy }

    function dissim(P) {
        if (M.D) return M.D;
        if (!P.impP) { M.D = P.D; return M.D; }
        // With multiple imputation: the dissimilarities averaged over the imputed datasets
        const n = P.rows.length, D = new Float64Array(n * n);
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
            if (S.result.pca) {
                const pca = S.result.pca;
                M.maps.pca = { Y: pca.scores.map(s => [s[0], s[1]]), idx: all, xLabel: `Principal component 1 (${(100 * pca.varianceRatio[0]).toFixed(0)}% of variance)`, yLabel: `Component 2 (${(100 * pca.varianceRatio[1]).toFixed(0)}%)` };
                M.kind = 'pca';
            } else {
                const D = dissim(P);
                if (D) { M.maps.mds = { Y: EE.classicalMDS(D, 2), idx: all, xLabel: 'Dimension 1', yLabel: 'Dimension 2' }; M.kind = 'mds'; }
            }
        }
        M.ctx = ctx;
        draw();
    }
    function trustOf(map) {
        if (map.trust === undefined) { const D = subset(dissim(M.ctx.P), map.idx); map.trust = EE.trustworthiness(D, map.Y, kTrust(D.n)); map.k = kTrust(D.n); }
        return map.trust;
    }
    async function compute(kind) {
        if (M.busy) return;
        const P = M.ctx.P, D0 = dissim(P);
        if (!D0) return;
        M.busy = true;
        const idx = sampleIdx(D0.n, LIMIT[kind], P.seed), D = subset(D0, idx);
        const status = $('map-status');
        try {
            if (kind === 'tsne') {
                const t = EE.createTSNE(EE.squared(D), { perplexity: 30 });
                while (!t.done) { t.step(25); status.textContent = `t-SNE: iteration ${t.iter} of ${t.maxIter}…`; await yieldUI(); }
                M.maps.tsne = { Y: t.Y, idx, xLabel: 't-SNE 1', yLabel: 't-SNE 2', extra: `perplexity ${Math.round(t.perplexity)}, divergence ${t.kl.toFixed(3)}` };
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
        const kinds = [M.maps.pca ? 'pca' : 'mds', 'tsne', 'umap'];
        const vars = [...P.cols, ...P.describe];
        const map = M.maps[M.kind];
        let body = '<p class="muted">No map is available for these data.</p>';
        if (map) {
            const c = colouring(map);
            const svg = CH.scatter({ points: map.idx.map((i, r) => ({ x: map.Y[r][0], y: map.Y[r][1], ...c.pts(i) })), xLabel: map.xLabel, yLabel: map.yLabel, noAxes: M.kind === 'tsne' || M.kind === 'umap', aria: `${NAMES[M.kind]} map of the patients` });
            const tr = trustOf(map);
            body = `${svg}${c.legend}
                <p class="muted small"><strong>${NAMES[M.kind]}</strong>${map.extra ? ` (${esc(map.extra)})` : ''}. ${esc(NOTES[M.kind])}</p>
                <p class="muted small">Trustworthiness ${tr.toFixed(3)}: how well the map keeps each patient's ${map.k} nearest neighbours (1 = perfectly; above 0.9 good).${map.idx.length < n ? ` Map of a random sample of ${map.idx.length} of the ${n} patients.` : ''}</p>`;
        }
        el.innerHTML = `
            <div class="row map-controls" role="group" aria-label="Type of map">${kinds.map(k => `<button type="button" class="btn small${M.kind === k ? ' primary' : ''}" data-map="${k}" aria-pressed="${M.kind === k}"${M.busy ? ' disabled' : ''}>${NAMES[k]}${(k === 'tsne' || k === 'umap') && !M.maps[k] ? ' (make)' : ''}</button>`).join('')}
                <label for="map-colour" class="small">Colour by</label><select id="map-colour"><option value="__cluster">Cluster</option>${vars.map(v => `<option value="${esc(v)}"${M.colourBy === v ? ' selected' : ''}>${esc(v)}</option>`).join('')}</select></div>
            <p id="map-status" class="status small" role="status"></p>
            ${body}`;
        el.querySelectorAll('[data-map]').forEach(b => b.addEventListener('click', () => { const k = b.dataset.map; if (M.maps[k]) { M.kind = k; draw(); } else compute(k); }));
        $('map-colour').addEventListener('change', e => { M.colourBy = e.target.value; draw(); });
    }
    root.UMEMaps = { render, _state: () => M };
})(typeof window !== 'undefined' ? window : globalThis);
