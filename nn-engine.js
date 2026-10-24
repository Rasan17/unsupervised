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
    function lossGrad(net, X, M, idx, weightDecay = 0) {
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
                if (l > 0) { const nd = new Array(a.length).fill(0); for (let o = 0; o < dz.length; o++) { const w = L.W[o]; for (let k = 0; k < a.length; k++) nd[k] += w[k] * dz[o]; } delta = nd; }
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
                cols.push({ variable: j, name: v.name, x: val.map(x => (x === null ? 0 : (x - m) / sd)), m: val.map(x => (x === null ? 0 : 1)) });
            } else {
                const val = raw.map(x => (x === null || x === undefined || x === '' ? null : String(x).trim())), levels = [...new Set(val.filter(x => x !== null))].sort(), nObs = val.filter(x => x !== null).length;
                levels.forEach(l => { const pq = val.filter(x => x === l).length / Math.max(1, nObs); cols.push({ variable: j, name: `${v.name} = ${l}`, x: val.map(x => (x === null || !(pq > 0) ? 0 : ((x === l ? 1 : 0) - pq) / Math.sqrt(pq))), m: val.map(x => (x === null ? 0 : 1)) }); });
            }
        });
        const n = rows.length;
        return { X: Array.from({ length: n }, (_, i) => cols.map(c => c.x[i])), M: Array.from({ length: n }, (_, i) => cols.map(c => c.m[i])), columns: cols.map(c => ({ variable: c.variable, name: c.name })) };
    }
    root.NNEngine = { createAutoencoder, forward, encode, reconstruct, lossGrad, adamStep, createTrainer, train, errors, codeData, clone };
})(typeof window !== 'undefined' ? window : globalThis);
