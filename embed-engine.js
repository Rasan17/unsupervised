// embed-engine.js: two-dimensional maps of patients from a dissimilarity matrix (the same one used for clustering,
// so mixed data and missing values work): classical multidimensional scaling, t-SNE (exact, as scikit-learn's
// method="exact") and UMAP (as umap-learn), plus trustworthiness to compare how faithful maps are.
// Plain script (window.EmbedEngine). Dissimilarities: { n, D } with D flat (D[i * n + j]).
(function (root) {
    'use strict';
    const EPS = 2.220446049250313e-16;   // numpy float64 machine epsilon (scikit-learn's MACHINE_EPSILON)
    function createPRNG(seed) {   // mulberry32, as the cluster engine
        let a = seed >>> 0;
        return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    }
    const gauss = rng => { let u = 0; while (u === 0) u = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng()); };

    // ------------------------------------------------------------------------------------------- classical MDS
    /** Classical (Torgerson) scaling: top eigenvectors of -1/2 J D^2 J by power iteration with deflation.
     *  With Euclidean distances this equals PCA scores (up to sign). Returns [n][dim]. */
    function classicalMDS({ n, D }, dim = 2) {
        const B = new Float64Array(n * n), rowMean = new Float64Array(n);
        let all = 0;
        for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) { const d = D[i * n + j]; s += d * d; } rowMean[i] = s / n; all += s; }
        all /= n * n;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const d = D[i * n + j]; B[i * n + j] = -0.5 * (d * d - rowMean[i] - rowMean[j] + all); }
        const vecs = [], vals = [];
        for (let c = 0; c < dim; c++) {
            // Deterministic start; Rayleigh-quotient power iteration on B minus the found components
            let v = new Float64Array(n); for (let i = 0; i < n; i++) v[i] = Math.sin(1 + i * (c + 1) * 0.7) + 0.01 * i / n;
            let lambda = 0;
            for (let it = 0; it < 1000; it++) {
                const w = new Float64Array(n);
                for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += B[i * n + j] * v[j]; w[i] = s; }
                vecs.forEach((u, q) => { let dot = 0; for (let i = 0; i < n; i++) dot += u[i] * v[i]; for (let i = 0; i < n; i++) w[i] -= vals[q] * dot * u[i]; });
                let norm = 0; for (let i = 0; i < n; i++) norm += w[i] * w[i]; norm = Math.sqrt(norm);
                if (!(norm > 0)) break;
                let diff = 0; for (let i = 0; i < n; i++) { const x = w[i] / norm; diff = Math.max(diff, Math.abs(Math.abs(x) - Math.abs(v[i]))); v[i] = x; }
                lambda = norm;
                if (diff < 1e-10) break;
            }
            // Sign convention: largest absolute coordinate positive
            let big = 0; for (let i = 0; i < n; i++) if (Math.abs(v[i]) > Math.abs(v[big])) big = i;
            if (v[big] < 0) for (let i = 0; i < n; i++) v[i] = -v[i];
            vecs.push(v); vals.push(lambda);
        }
        return Array.from({ length: n }, (_, i) => vecs.map((v, c) => v[i] * Math.sqrt(Math.max(vals[c], 0))));
    }

    // ------------------------------------------------------------------------------------------- t-SNE
    /** Joint probabilities P (flat n x n, symmetric) from squared dissimilarities, as scikit-learn's
     *  _joint_probabilities: per-patient Gaussian bandwidth by binary search to the perplexity. */
    function tsneP({ n, D }, perplexity = 30) {
        const cond = new Float64Array(n * n), desired = Math.log(perplexity);
        for (let i = 0; i < n; i++) {
            let beta = 1, lo = -Infinity, hi = Infinity;
            const row = new Float64Array(n);
            for (let step = 0; step < 100; step++) {
                let sumP = 0;
                for (let j = 0; j < n; j++) { row[j] = j === i ? 0 : Math.exp(-D[i * n + j] * beta); sumP += row[j]; }
                if (sumP === 0) sumP = 1e-8;
                let sumDP = 0;
                for (let j = 0; j < n; j++) { row[j] /= sumP; sumDP += D[i * n + j] * row[j]; }
                const H = Math.log(sumP) + beta * sumDP, diff = H - desired;
                if (Math.abs(diff) <= 1e-5) break;
                if (diff > 0) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; } else { hi = beta; beta = lo === -Infinity ? beta / 2 : (beta + lo) / 2; }
            }
            for (let j = 0; j < n; j++) cond[i * n + j] = row[j];
        }
        const P = new Float64Array(n * n); let sum = 0;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const v = cond[i * n + j] + cond[j * n + i]; P[i * n + j] = v; if (i !== j) sum += v; }
        sum = Math.max(sum, EPS);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) P[i * n + j] = i === j ? 0 : Math.max(P[i * n + j] / sum, EPS);
        return P;
    }
    /** Kullback-Leibler divergence of the map from P and its gradient (Student t, one degree of freedom) */
    function tsneKL(P, Y, n, exaggeration = 1) {
        const dim = Y[0].length, W = new Float64Array(n * n);
        let sumW = 0;
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
            let d = 0; for (let c = 0; c < dim; c++) { const t = Y[i][c] - Y[j][c]; d += t * t; }
            const w = 1 / (1 + d); W[i * n + j] = W[j * n + i] = w; sumW += 2 * w;
        }
        let kl = 0; const grad = Array.from({ length: n }, () => new Array(dim).fill(0));
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
            const q = Math.max(W[i * n + j] / sumW, EPS), pij = exaggeration * P[i * n + j];
            kl += 2 * pij * Math.log(Math.max(pij, EPS) / q);
            const f = 4 * (pij - q) * W[i * n + j];
            for (let c = 0; c < dim; c++) { const t = f * (Y[i][c] - Y[j][c]); grad[i][c] += t; grad[j][c] -= t; }
        }
        return { kl, grad };
    }
    /** Step-wise t-SNE (scikit-learn defaults: perplexity 30, early exaggeration 12 for 250 iterations, learning rate
     *  max(n / 12 / 4, 50), momentum 0.5 then 0.8, adaptive gains with minimum 0.01, 1000 iterations).
     *  init: [n][2] start (default classical MDS scaled to standard deviation 1e-4, as scikit-learn's PCA start). */
    function createTSNE(Dsq, { perplexity = 30, maxIter = 1000, init = null, earlyExaggeration = 12, exploreIter = 250, learningRate = null } = {}) {
        const n = Dsq.n;
        if (n < 4) throw new Error('At least 4 patients are needed for a t-SNE map.');
        const perp = Math.min(perplexity, (n - 1) / 3);
        const P = tsneP(Dsq, perp);
        let Y = init ? init.map(r => r.slice()) : (() => {
            const M = classicalMDS({ n, D: Dsq.D.map(Math.sqrt) }, 2);
            const sd = Math.sqrt(M.reduce((a, r) => a + r[0] * r[0], 0) / n) || 1;
            return M.map(r => r.map(v => v / sd * 1e-4));
        })();
        const lr = learningRate ?? Math.max(n / earlyExaggeration / 4, 50);
        const update = Array.from({ length: n }, () => [0, 0]), gains = Array.from({ length: n }, () => [1, 1]);
        const state = { iter: 0, maxIter, kl: null, perplexity: perp, get Y() { return Y; }, done: false };
        state.step = (count = 50) => {
            for (let s = 0; s < count && state.iter < maxIter; s++, state.iter++) {
                const explore = state.iter < exploreIter, mom = explore ? 0.5 : 0.8;
                // scikit-learn runs the two stages as separate descents: momentum and gains restart after exaggeration
                if (state.iter === exploreIter) for (let i = 0; i < n; i++) { update[i][0] = update[i][1] = 0; gains[i][0] = gains[i][1] = 1; }
                const { kl, grad } = tsneKL(P, Y, n, explore ? earlyExaggeration : 1);
                state.kl = kl;
                for (let i = 0; i < n; i++) for (let c = 0; c < 2; c++) {
                    const g = grad[i][c];
                    gains[i][c] = update[i][c] * g < 0 ? gains[i][c] + 0.2 : gains[i][c] * 0.8;
                    if (gains[i][c] < 0.01) gains[i][c] = 0.01;
                    update[i][c] = mom * update[i][c] - lr * gains[i][c] * g;
                    Y[i][c] += update[i][c];
                }
            }
            if (state.iter >= maxIter) { state.done = true; state.kl = tsneKL(P, Y, n, 1).kl; }
            return state;
        };
        return state;
    }
    function tsne(Dsq, opts = {}) { const t = createTSNE(Dsq, opts); while (!t.done) t.step(100); return { Y: t.Y, kl: t.kl, perplexity: t.perplexity }; }

    // ------------------------------------------------------------------------------------------- UMAP
    /** Fuzzy neighbour graph (umap-learn fuzzy_simplicial_set, set_op_mix_ratio 1, local_connectivity 1), from exact
     *  nearest neighbours of the dissimilarity matrix (the patient itself counts as its first neighbour).
     *  Returns { n, rows, cols, vals } (symmetric, both directions), plus sigmas and rhos. */
    function umapGraph({ n, D }, nNeighbors = 15) {
        const k = Math.min(nNeighbors, n);
        const knn = [], knd = [];
        for (let i = 0; i < n; i++) {
            const idx = Array.from({ length: n }, (_, j) => j).sort((a, b) => (D[i * n + a] - D[i * n + b]) || (a === i ? -1 : b === i ? 1 : a - b));
            if (idx[0] !== i) { const p = idx.indexOf(i); idx.splice(p, 1); idx.unshift(i); }
            knn.push(idx.slice(0, k)); knd.push(idx.slice(0, k).map(j => (j === i ? 0 : D[i * n + j])));
        }
        const target = Math.log2(k);
        let meanAll = 0; knd.forEach(r => r.forEach(v => { meanAll += v; })); meanAll /= n * k;
        const sigmas = new Float64Array(n), rhos = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            const d = knd[i], nz = d.filter(v => v > 0);
            rhos[i] = nz.length ? nz[0] : 0;
            let lo = 0, hi = Infinity, mid = 1;
            for (let it = 0; it < 64; it++) {
                let ps = 0; for (let j = 1; j < k; j++) { const t = d[j] - rhos[i]; ps += t > 0 ? Math.exp(-(t / mid)) : 1; }
                if (Math.abs(ps - target) < 1e-5) break;
                if (ps > target) { hi = mid; mid = (lo + hi) / 2; } else { lo = mid; mid = hi === Infinity ? mid * 2 : (lo + hi) / 2; }
            }
            const meanI = d.reduce((a, b) => a + b, 0) / k;
            sigmas[i] = rhos[i] > 0 ? Math.max(mid, 1e-3 * meanI) : Math.max(mid, 1e-3 * meanAll);
        }
        const W = new Map();
        for (let i = 0; i < n; i++) for (let jj = 0; jj < k; jj++) {
            const j = knn[i][jj]; if (j === i) continue;
            const t = knd[i][jj] - rhos[i], v = t <= 0 || sigmas[i] === 0 ? 1 : Math.exp(-(t / sigmas[i]));
            W.set(i * n + j, v);
        }
        const rows = [], cols = [], vals = [];
        const seen = new Set();
        W.forEach((v, key) => {
            const i = Math.floor(key / n), j = key % n, a = Math.min(i, j), b = Math.max(i, j), pk = a * n + b;
            if (seen.has(pk)) return; seen.add(pk);
            const wij = W.get(i * n + j) || 0, wji = W.get(j * n + i) || 0, s = wij + wji - wij * wji;
            if (s > 0) { rows.push(a, b); cols.push(b, a); vals.push(s, s); }
        });
        return { n, rows, cols, vals, sigmas, rhos };
    }
    /** a and b of the curve 1 / (1 + a d^(2b)) that best fits UMAP's target (umap-learn find_ab_params), by
     *  Levenberg-Marquardt least squares on 300 points over [0, 3 spread] */
    function findAB(spread = 1, minDist = 0.1) {
        const xs = Array.from({ length: 300 }, (_, i) => (3 * spread * i) / 299), ys = xs.map(x => (x < minDist ? 1 : Math.exp(-(x - minDist) / spread)));
        let a = 1, b = 1, lam = 1e-3;
        const sse = (A, B) => xs.reduce((s, x, i) => { const f = 1 / (1 + A * Math.pow(x, 2 * B)); return s + (f - ys[i]) ** 2; }, 0);
        let cur = sse(a, b);
        for (let it = 0; it < 500; it++) {
            let JTJ00 = 0, JTJ01 = 0, JTJ11 = 0, g0 = 0, g1 = 0;
            xs.forEach((x, i) => {
                if (x === 0) return;
                const x2b = Math.pow(x, 2 * b), den = 1 + a * x2b, f = 1 / den, r = f - ys[i];
                const da = -x2b / (den * den), db = -a * x2b * 2 * Math.log(x) / (den * den);
                JTJ00 += da * da; JTJ01 += da * db; JTJ11 += db * db; g0 += da * r; g1 += db * r;
            });
            let improved = false;
            for (let tries = 0; tries < 30; tries++) {
                const m00 = JTJ00 * (1 + lam), m11 = JTJ11 * (1 + lam), det = m00 * m11 - JTJ01 * JTJ01;
                const sa = -(m11 * g0 - JTJ01 * g1) / det, sb = -(-JTJ01 * g0 + m00 * g1) / det;
                const nv = sse(a + sa, b + sb);
                if (nv < cur) { a += sa; b += sb; const rel = (cur - nv) / Math.max(cur, 1e-300); cur = nv; lam = Math.max(lam / 10, 1e-12); improved = rel > 1e-15; break; }
                lam *= 10;
            }
            if (!improved) break;
        }
        return { a, b };
    }
    /** Step-wise UMAP layout (umap-learn optimize_layout_euclidean: 500 epochs for up to 10000 patients, learning rate
     *  1 decreasing linearly, 5 negative samples per positive one, gradients clipped at 4). Start: classical MDS
     *  scaled to coordinates within 10, plus noise (standard deviation 1e-4), as umap-learn's init="pca". */
    function createUMAP(Dm, { nNeighbors = 15, minDist = 0.1, spread = 1, nEpochs = null, seed = 2024, negativeSampleRate = 5, init = null } = {}) {
        const n = Dm.n;
        if (n < 4) throw new Error('At least 4 patients are needed for a UMAP map.');
        const G = umapGraph(Dm, nNeighbors), { a, b } = findAB(spread, minDist);
        const epochs = nEpochs ?? (n <= 10000 ? 500 : 200);
        const maxW = Math.max(...G.vals);
        const keep = G.vals.map(v => v >= maxW / epochs);
        const head = G.rows.filter((_, e) => keep[e]), tail = G.cols.filter((_, e) => keep[e]), w = G.vals.filter((_, e) => keep[e]);
        const wMax = Math.max(...w), eps = w.map(v => { const ns = epochs * (v / wMax); return ns > 0 ? epochs / ns : -1; });
        const epsNeg = eps.map(v => v / negativeSampleRate), nextNeg = epsNeg.slice(), next = eps.slice();
        const rng = createPRNG(seed);
        let Y = init ? init.map(r => r.slice()) : (() => {
            const M = classicalMDS(Dm, 2), mx = Math.max(...M.flat().map(Math.abs)) || 1;
            return M.map(r => r.map(v => (10 * v) / mx + 1e-4 * gauss(rng)));
        })();
        const clip = v => (v > 4 ? 4 : v < -4 ? -4 : v);
        const state = { epoch: 0, nEpochs: epochs, a, b, graph: G, get Y() { return Y; }, done: false };
        state.step = (count = 25) => {
            for (let s = 0; s < count && state.epoch < epochs; s++, state.epoch++) {
                const ep = state.epoch, alpha = 1 - ep / epochs;
                for (let e = 0; e < head.length; e++) {
                    if (next[e] > ep) continue;
                    const j = head[e], k = tail[e], cur = Y[j], oth = Y[k];
                    let d2 = (cur[0] - oth[0]) ** 2 + (cur[1] - oth[1]) ** 2;
                    let gc = d2 > 0 ? (-2 * a * b * Math.pow(d2, b - 1)) / (a * Math.pow(d2, b) + 1) : 0;
                    for (let c = 0; c < 2; c++) { const g = clip(gc * (cur[c] - oth[c])); cur[c] += g * alpha; oth[c] -= g * alpha; }
                    next[e] += eps[e];
                    const nNeg = Math.floor((ep - nextNeg[e]) / epsNeg[e]);
                    for (let q = 0; q < nNeg; q++) {
                        const kk = Math.floor(rng() * n), o = Y[kk];
                        d2 = (cur[0] - o[0]) ** 2 + (cur[1] - o[1]) ** 2;
                        if (d2 > 0) gc = (2 * b) / ((0.001 + d2) * (a * Math.pow(d2, b) + 1)); else if (kk === j) continue; else gc = 0;
                        for (let c = 0; c < 2; c++) cur[c] += (gc > 0 ? clip(gc * (cur[c] - o[c])) : 0) * alpha;
                    }
                    nextNeg[e] += nNeg * epsNeg[e];
                }
            }
            if (state.epoch >= epochs) state.done = true;
            return state;
        };
        return state;
    }
    function umap(Dm, opts = {}) { const u = createUMAP(Dm, opts); while (!u.done) u.step(100); return { Y: u.Y, a: u.a, b: u.b }; }

    // ------------------------------------------------------------------------------------------- faithfulness
    /** Trustworthiness (Venna & Kaski; scikit-learn manifold.trustworthiness): penalises patients that are near
     *  neighbours on the map but not in the data, by how far down the data ranking they are. 1 = perfect. */
    function trustworthiness({ n, D }, Y, k = 5) {
        if (k >= n / 2) k = Math.max(1, Math.floor(n / 2) - 1);
        let t = 0;
        for (let i = 0; i < n; i++) {
            const byData = Array.from({ length: n }, (_, j) => j).sort((a, b) => (a === i ? -1 : b === i ? 1 : (D[i * n + a] - D[i * n + b]) || a - b));
            const rank = new Int32Array(n); byData.forEach((j, r) => { rank[j] = r; });
            const dY = j => { let s = 0; for (let c = 0; c < Y[i].length; c++) s += (Y[i][c] - Y[j][c]) ** 2; return s; };
            const byMap = Array.from({ length: n }, (_, j) => j).filter(j => j !== i).map(j => [j, dY(j)]).sort((a, b) => a[1] - b[1] || a[0] - b[0]).slice(0, k);
            byMap.forEach(([j]) => { const r = rank[j]; if (r > k) t += r - k; });
        }
        return 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * t;
    }
    const squared = ({ n, D }) => ({ n, D: D.map(v => v * v) });
    root.EmbedEngine = { createPRNG, classicalMDS, tsneP, tsneKL, createTSNE, tsne, umapGraph, findAB, createUMAP, umap, trustworthiness, squared };
})(typeof window !== 'undefined' ? window : globalThis);
