// nn-engine.js: a small autoencoder (deep learning) written directly, so the page keeps a strict security policy
// (no library that evaluates generated code). Fully connected layers, tanh (or linear) activations, a linear
// bottleneck and output, mean squared error over the recorded values only (missing values are left out of the
// loss), Adam with L2 weight decay, mini-batches, early stopping on held-out patients. Plain script
// (window.NNEngine); uses ClusterEngine's random numbers.
(function (root) {
    'use strict';
    const act = {
        tanh: { f: x => Math.tanh(x), d: y => 1 - y * y },   // derivative from the output y
        linear: { f: x => x, d: () => 1 }
    };
    /** Network p -> hidden... -> latent -> hidden (mirrored)... -> p. Glorot-uniform weights, zero biases. */
    function createAutoencoder({ p, hidden = [16], latent = 2, activation = 'tanh', seed = 2024 }) {
        const rng = root.ClusterEngine.createPRNG(seed);
        const sizes = [p, ...hidden, latent, ...hidden.slice().reverse(), p];
        const layers = [];
        for (let l = 0; l < sizes.length - 1; l++) {
            const nin = sizes[l], nout = sizes[l + 1], lim = Math.sqrt(6 / (nin + nout));
            const isLatent = l === hidden.length, isOutput = l === sizes.length - 2;
            layers.push({ W: Array.from({ length: nout }, () => Array.from({ length: nin }, () => (2 * rng() - 1) * lim)), b: new Array(nout).fill(0), act: isLatent || isOutput ? 'linear' : activation });
        }
        return { layers, sizes, latentLayer: hidden.length };
    }
    function forward(net, x) {
        const outs = [x];
        net.layers.forEach(L => {
            const a = outs[outs.length - 1], f = act[L.act].f;
            outs.push(L.W.map((row, o) => { let s = L.b[o]; for (let i = 0; i < row.length; i++) s += row[i] * a[i]; return f(s); }));
        });
        return outs;
    }
    const encode = (net, x) => forward(net, x)[net.latentLayer + 1];
    const reconstruct = (net, x) => { const o = forward(net, x); return o[o.length - 1]; };
    /**
     * Loss and gradients for a batch: mean over the batch of the sum over recorded entries of the squared error,
     * divided by the number of variables, plus (weightDecay / 2) x sum of squared weights (not biases).
     * X: inputs (missing entries 0 = the mean after standardising); M: masks (1 recorded, 0 missing).
     */
    function lossGrad(net, X, M, idx, weightDecay = 0, latentGrad = null) {
        const p = net.sizes[0], gW = net.layers.map(L => L.W.map(r => r.map(() => 0))), gb = net.layers.map(L => L.b.map(() => 0));
        let loss = 0;
        idx.forEach(i => {
            const outs = forward(net, X[i]), y = outs[outs.length - 1];
            let delta = y.map((v, j) => (M[i][j] ? (2 * (v - X[i][j])) / (p * idx.length) : 0));
            y.forEach((v, j) => { if (M[i][j]) loss += (v - X[i][j]) ** 2 / (p * idx.length); });
            for (let l = net.layers.length - 1; l >= 0; l--) {
                const L = net.layers[l], a = outs[l], out = outs[l + 1], d = act[L.act].d;
                const dz = delta.map((v, o) => v * d(out[o]));
                for (let o = 0; o < dz.length; o++) { gb[l][o] += dz[o]; const row = gW[l][o]; for (let k = 0; k < a.length; k++) row[k] += dz[o] * a[k]; }
                if (l > 0) {
                    const nd = new Array(a.length).fill(0); for (let o = 0; o < dz.length; o++) { const w = L.W[o]; for (let k = 0; k < a.length; k++) nd[k] += w[k] * dz[o]; }
                    // Outputs of layer l - 1 are the latent representation: add the gradient of any loss on it
                    if (latentGrad && l - 1 === net.latentLayer) { const gz = latentGrad(i, a); for (let k = 0; k < nd.length; k++) nd[k] += gz[k]; }
                    delta = nd;
                }
            }
        });
        if (weightDecay > 0) net.layers.forEach((L, l) => L.W.forEach((row, o) => row.forEach((w, k) => { loss += 0.5 * weightDecay * w * w; gW[l][o][k] += weightDecay * w; })));
        return { loss, gW, gb };
    }
    /** Adam (Kingma & Ba 2015): one update from gradients; state holds the moments and the step count */
    function adamStep(net, g, state, { lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8 } = {}) {
        if (!state.m) { state.m = { W: g.gW.map(L => L.map(r => r.map(() => 0))), b: g.gb.map(L => L.map(() => 0)) }; state.v = { W: g.gW.map(L => L.map(r => r.map(() => 0))), b: g.gb.map(L => L.map(() => 0)) }; state.t = 0; }
        state.t++;
        const c1 = 1 - Math.pow(beta1, state.t), c2 = 1 - Math.pow(beta2, state.t);
        const upd = (param, grad, m, v) => { m = beta1 * m + (1 - beta1) * grad; v = beta2 * v + (1 - beta2) * grad * grad; return { p: param - (lr * (m / c1)) / (Math.sqrt(v / c2) + eps), m, v }; };
        net.layers.forEach((L, l) => {
            L.W.forEach((row, o) => row.forEach((w, k) => { const r = upd(w, g.gW[l][o][k], state.m.W[l][o][k], state.v.W[l][o][k]); row[k] = r.p; state.m.W[l][o][k] = r.m; state.v.W[l][o][k] = r.v; }));
            L.b.forEach((bb, o) => { const r = upd(bb, g.gb[l][o], state.m.b[l][o], state.v.b[l][o]); L.b[o] = r.p; state.m.b[l][o] = r.m; state.v.b[l][o] = r.v; });
        });
    }
    const clone = net => ({ ...net, layers: net.layers.map(L => ({ W: L.W.map(r => r.slice()), b: L.b.slice(), act: L.act })) });
    /**
     * Step-wise training: 20% of patients held back (validation) for early stopping (patience 40 epochs; the best
     * weights are kept); mini-batches of 32 in a new random order each epoch.
     */
    function createTrainer(X, M, { hidden = [16], latent = 2, activation = 'tanh', seed = 2024, lr = 0.01, weightDecay = 1e-4, batch = 32, maxEpochs = 600, patience = 40, validation = 0.2 } = {}) {
        const n = X.length, p = X[0].length, rng = root.ClusterEngine.createPRNG(seed + 1);
        const net = createAutoencoder({ p, hidden, latent, activation, seed });
        const order = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
        const nVal = n >= 50 ? Math.round(validation * n) : 0, val = order.slice(0, nVal), trainIdx = order.slice(nVal);
        const state = {}, history = [];
        let best = { loss: Infinity, net: clone(net), epoch: 0 }, epoch = 0, done = false;
        const evalLoss = idx => (idx.length ? lossGrad(net, X, M, idx, 0).loss : NaN);
        const T = {
            get epoch() { return epoch; }, maxEpochs, history, get done() { return done; }, get net() { return best.net; }, get bestEpoch() { return best.epoch; }, nVal,
            step(count = 10) {
                for (let s = 0; s < count && !done; s++) {
                    const ord = trainIdx.slice(); for (let i = ord.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [ord[i], ord[j]] = [ord[j], ord[i]]; }
                    for (let b0 = 0; b0 < ord.length; b0 += batch) adamStep(net, lossGrad(net, X, M, ord.slice(b0, b0 + batch), weightDecay), state, { lr });
                    epoch++;
                    const tl = evalLoss(trainIdx), vl = nVal ? evalLoss(val) : tl;
                    history.push({ epoch, train: tl, val: vl });
                    if (vl < best.loss - 1e-9) best = { loss: vl, net: clone(net), epoch };
                    if (epoch >= maxEpochs || epoch - best.epoch >= patience) done = true;
                }
                return T;
            }
        };
        return T;
    }
    function train(X, M, opts = {}) { const t = createTrainer(X, M, opts); while (!t.done) t.step(50); return { net: t.net, history: t.history, bestEpoch: t.bestEpoch }; }
    /** Per patient: mean squared reconstruction error over the recorded values, and per variable squared errors */
    function errors(net, X, M) {
        return X.map((x, i) => { const y = reconstruct(net, x), e = y.map((v, j) => (M[i][j] ? (v - x[j]) ** 2 : null)); const rec = e.filter(v => v !== null); return { mean: rec.length ? rec.reduce((a, b) => a + b, 0) / rec.length : NaN, perVar: e }; });
    }
    /**
     * Coding of clinical variables for the network: numbers and grades standardised; categories as indicators scaled
     * as in FAMD ((indicator - p) / sqrt(p)); missing values 0 with mask 0. Returns X, M and for each column the
     * variable it belongs to.
     */
    function codeData(rows, vars) {
        const cols = [];
        vars.forEach((v, j) => {
            const raw = rows.map(r => r[j]);
            if (v.type !== 'nominal') {
                const val = raw.map(x => { if (x === null || x === undefined || x === '') return null; if (v.type === 'ordinal' && v.levels) { const k = v.levels.indexOf(String(x).trim()); return k < 0 ? null : k; } const y = Number(x); return Number.isFinite(y) ? y : null; });
                const obs = val.filter(x => x !== null), m = obs.reduce((a, b) => a + b, 0) / Math.max(1, obs.length), sd = Math.sqrt(obs.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, obs.length)) || 1;
                const toNum = x => { if (x === null || x === undefined || x === '') return null; if (v.type === 'ordinal' && v.levels) { const k = v.levels.indexOf(String(x).trim()); return k < 0 ? null : k; } const y = Number(x); return Number.isFinite(y) ? y : null; };
                cols.push({ variable: j, name: v.name, x: val.map(x => (x === null ? 0 : (x - m) / sd)), m: val.map(x => (x === null ? 0 : 1)), coder: row => { const y = toNum(row[j]); return y === null ? [0, 0] : [(y - m) / sd, 1]; } });
            } else {
                const val = raw.map(x => (x === null || x === undefined || x === '' ? null : String(x).trim())), levels = [...new Set(val.filter(x => x !== null))].sort(), nObs = val.filter(x => x !== null).length;
                levels.forEach(l => { const pq = val.filter(x => x === l).length / Math.max(1, nObs); cols.push({ variable: j, name: `${v.name} = ${l}`, x: val.map(x => (x === null || !(pq > 0) ? 0 : ((x === l ? 1 : 0) - pq) / Math.sqrt(pq))), m: val.map(x => (x === null ? 0 : 1)),
                    coder: row => { const x = row[j]; if (x === null || x === undefined || x === '') return [0, 0]; return [pq > 0 ? ((String(x).trim() === l ? 1 : 0) - pq) / Math.sqrt(pq) : 0, 1]; } }); });
            }
        });
        const n = rows.length;
        return { X: Array.from({ length: n }, (_, i) => cols.map(c => c.x[i])), M: Array.from({ length: n }, (_, i) => cols.map(c => c.m[i])), columns: cols.map(c => ({ variable: c.variable, name: c.name })), coders: cols.map(c => c.coder) };
    }


    // ------------------------------------------------------------------------------------------- self-organising map
    // Masked squared distance between a record and a prototype, scaled to all columns (missing values left out)
    function mdist(x, m, w) { let s = 0, c = 0; for (let j = 0; j < x.length; j++) if (m[j]) { const t = x[j] - w[j]; s += t * t; c++; } return c ? (s * x.length) / c : Infinity; }
    /**
     * Batch self-organising map (Kohonen 2001), rectangular grid, Gaussian neighbourhood whose radius falls linearly
     * from sigma0 to sigma1 over the epochs. Each epoch every record is assigned to its best-matching unit (masked
     * distance), then every prototype becomes the neighbourhood-weighted mean of the records (recorded values
     * only). Linear initialisation along the first two principal components (deterministic). With sigma0 = sigma1
     * = 0 and an initialisation given, it is exactly Lloyd's k-means.
     * Returns prototypes W, each record's unit, hits per unit, quantisation error (mean distance to the unit) and
     * topographic error (share of records whose two best units are not neighbours).
     */
    function som(X, M, { rows = null, cols = null, epochs = 40, sigma0 = null, sigma1 = 0.2, init = null } = {}) {
        const n = X.length, p = X[0].length;
        if (!rows || !cols) {
            // Grid size: about 5 sqrt(n) units (Vesanto 2000), between 25 and 225, sides in the ratio of the first two
            // principal components
            const units = Math.max(25, Math.min(225, Math.round(5 * Math.sqrt(n)))), ev = pcs(X).values, ratio = Math.sqrt(Math.max(1, ev[0] / Math.max(1e-12, ev[1] || ev[0])));
            rows = Math.max(3, Math.round(Math.sqrt(units / ratio))); cols = Math.max(3, Math.round(units / rows));
        }
        const U = rows * cols, pos = Array.from({ length: U }, (_, u) => [Math.floor(u / cols), u % cols]);
        let W = init ? init.map(r => r.slice()) : (() => {
            const P = pcs(X), mu = P.mean, a = Math.sqrt(Math.max(0, P.values[0])), b = Math.sqrt(Math.max(0, P.values[1] || 0));
            return pos.map(([r, c]) => mu.map((m0, j) => m0 + (rows > 1 ? (2 * r / (rows - 1) - 1) : 0) * a * P.vectors[0][j] + (cols > 1 ? (2 * c / (cols - 1) - 1) : 0) * b * (P.vectors[1] ? P.vectors[1][j] : 0)));
        })();
        const s0 = sigma0 ?? Math.max(rows, cols) / 2, history = [];
        const bmu = new Int32Array(n);
        const assign = () => { let qe = 0; for (let i = 0; i < n; i++) { let best = 0, bd = Infinity; for (let u = 0; u < U; u++) { const d = mdist(X[i], M[i], W[u]); if (d < bd) { bd = d; best = u; } } bmu[i] = best; qe += Math.sqrt(bd); } return qe / n; };
        for (let ep = 0; ep < epochs; ep++) {
            const sigma = epochs > 1 ? s0 + (sigma1 - s0) * (ep / (epochs - 1)) : sigma1;
            const qe = assign();
            history.push({ epoch: ep + 1, sigma, qe });
            // Sums per unit, then neighbourhood smoothing across units
            const S = Array.from({ length: U }, () => new Array(p).fill(0)), N = Array.from({ length: U }, () => new Array(p).fill(0));
            for (let i = 0; i < n; i++) { const b = bmu[i]; for (let j = 0; j < p; j++) if (M[i][j]) { S[b][j] += X[i][j]; N[b][j] += 1; } }
            const h = (u, v) => { if (sigma <= 0) return u === v ? 1 : 0; const dr = pos[u][0] - pos[v][0], dc = pos[u][1] - pos[v][1]; return Math.exp(-(dr * dr + dc * dc) / (2 * sigma * sigma)); };
            W = W.map((w, u) => { const num = new Array(p).fill(0), den = new Array(p).fill(0); for (let v = 0; v < U; v++) { const hv = h(u, v); if (hv < 1e-12) continue; for (let j = 0; j < p; j++) { num[j] += hv * S[v][j]; den[j] += hv * N[v][j]; } } return w.map((old, j) => (den[j] > 1e-12 ? num[j] / den[j] : old)); });
        }
        const qe = assign(), hits = new Array(U).fill(0); for (let i = 0; i < n; i++) hits[bmu[i]]++;
        let te = 0;
        for (let i = 0; i < n; i++) { let b1 = -1, b2 = -1, d1 = Infinity, d2 = Infinity; for (let u = 0; u < U; u++) { const d = mdist(X[i], M[i], W[u]); if (d < d1) { d2 = d1; b2 = b1; d1 = d; b1 = u; } else if (d < d2) { d2 = d; b2 = u; } } if (b2 >= 0 && Math.max(Math.abs(pos[b1][0] - pos[b2][0]), Math.abs(pos[b1][1] - pos[b2][1])) > 1) te++; }
        return { W, rows, cols, pos, bmu: Array.from(bmu), hits, qe, te: te / n, history, epochs };
    }
    // Principal components of the rows of X (for the SOM grid and its start), via ClusterEngine's eigen solver
    function pcs(X) {
        const CEg = root.ClusterEngine, p = X[0].length, n = X.length, mean = new Array(p).fill(0);
        X.forEach(r => r.forEach((v, j) => { mean[j] += v / n; }));
        const C = Array.from({ length: p }, () => new Array(p).fill(0));
        X.forEach(r => { for (let a = 0; a < p; a++) { const da = r[a] - mean[a]; for (let b = a; b < p; b++) C[a][b] += (da * (r[b] - mean[b])) / n; } });
        for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) C[a][b] = C[b][a];
        const e = CEg.symmetricEigen(C);
        return { mean, values: e.values, vectors: e.vectors };
    }
    /** Best-matching unit of a new record (masked distance) */
    function somUnit(map, x, m) { let best = 0, bd = Infinity; map.W.forEach((w, u) => { const d = mdist(x, m, w); if (d < bd) { bd = d; best = u; } }); return best; }
    /**
     * Weighted k-means (k-means++ starts, Lloyd), for grouping the map's units weighted by the records they hold
     * (the second level of Vesanto & Alhoniemi 2000). Units without records keep the group of their nearest centre.
     * Returns unit labels (0 = the group holding most records) and centres.
     */
    function weightedKmeans(P, wts, k, { seed = 2024, nInit = 10, maxIter = 300, init = null } = {}) {
        const rng = root.ClusterEngine.createPRNG(seed), U = P.length, p = P[0].length, d2 = (a, b) => { let s = 0; for (let j = 0; j < p; j++) s += (a[j] - b[j]) ** 2; return s; };
        let best = null;
        for (let t = 0; t < (init ? 1 : nInit); t++) {
            const C = init ? init.map(c => c.slice()) : [], tot = wts.reduce((a, b) => a + b, 0);
            if (!init) { let r = rng() * tot, u0 = 0; while (u0 < U - 1 && r > wts[u0]) { r -= wts[u0]; u0++; } C.push(P[u0].slice()); }
            while (C.length < k) { const dd = P.map((x, u) => wts[u] * Math.min(...C.map(c => d2(x, c)))), sm = dd.reduce((a, b) => a + b, 0); if (!(sm > 0)) { C.push(P[Math.floor(rng() * U)].slice()); continue; } let q = rng() * sm, u = 0; while (u < U - 1 && q > dd[u]) { q -= dd[u]; u++; } C.push(P[u].slice()); }
            let lab = new Array(U).fill(0);
            for (let it = 0; it < maxIter; it++) {
                const nl = P.map(x => { let b = 0, bd = Infinity; C.forEach((c, j) => { const d = d2(x, c); if (d < bd) { bd = d; b = j; } }); return b; });
                const changed = nl.some((l, u) => l !== lab[u]); lab = nl;
                for (let j = 0; j < k; j++) { let w = 0; const s = new Array(p).fill(0); P.forEach((x, u) => { if (lab[u] === j && wts[u] > 0) { w += wts[u]; for (let q = 0; q < p; q++) s[q] += wts[u] * x[q]; } }); if (w > 0) C[j] = s.map(v => v / w); }
                if (!changed && it > 0) break;
            }
            const inertia = P.reduce((a, x, u) => a + wts[u] * d2(x, C[lab[u]]), 0);
            if (!best || inertia < best.inertia - 1e-12) best = { labels: lab, centres: C.map(c => c.slice()), inertia };
        }
        // Order groups by the records they hold
        const size = new Array(k).fill(0); best.labels.forEach((l, u) => { size[l] += wts[u]; });
        const order = size.map((_, j) => j).sort((a, b) => size[b] - size[a] || a - b), inv = new Array(k); order.forEach((j, r) => { inv[j] = r; });
        return { labels: best.labels.map(l => inv[l]), centres: order.map(j => best.centres[j]), inertia: best.inertia };
    }
    /** Code a new patient (raw values in the variables' order) exactly as the training data were coded */
    const codeRow = (coded, row) => { const xm = coded.coders.map(f => f(row)); return { x: xm.map(v => v[0]), m: xm.map(v => v[1]) }; };
    root.NNEngine = { codeRow, som, somUnit, weightedKmeans, pcs, createAutoencoder, forward, encode, reconstruct, lossGrad, adamStep, createTrainer, train, errors, codeData, clone };
})(typeof window !== 'undefined' ? window : globalThis);
