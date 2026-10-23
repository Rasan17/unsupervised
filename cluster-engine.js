// cluster-engine.js: the statistical engine of the Unsupervised Medical Explorer (no interface code).
// Clustering (k-means, k-medoids/PAM, hierarchical, Gaussian mixtures), dissimilarities (Euclidean, Gower for
// mixed clinical data), validity indices (silhouette, Calinski-Harabasz, Davies-Bouldin, gap statistic, BIC),
// agreement (adjusted Rand index), stability (bootstrap Jaccard, Hennig's clusterboot), a test of "no real
// clusters", and principal component analysis. Every routine is deterministic given its seed and is checked
// against scikit-learn, SciPy or R's cluster package in tests/reference_check.*.
// Loaded as a plain script in the page and in the background worker (window.ClusterEngine) and in Node tests.
(function (root) {
    'use strict';

    // ------------------------------------------------------------------------------------------- utilities
    function createPRNG(seed) {
        let s = (Number.isFinite(Number(seed)) ? Number(seed) : 2024) >>> 0;
        return function () { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    }
    function gaussian(rng) { let u = 0, v = 0; while (u === 0) u = rng(); v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
    const sq = v => v * v;
    function sqDist(a, b) { let s = 0; for (let j = 0; j < a.length; j++) { const d = a[j] - b[j]; s += d * d; } return s; }
    function colMeans(X) { const n = X.length, p = X[0].length, m = new Array(p).fill(0); for (const r of X) for (let j = 0; j < p; j++) m[j] += r[j] / n; return m; }
    function checkMatrix(X) {
        if (!Array.isArray(X) || !X.length) throw new Error('No patients to analyse.');
        const p = X[0].length;
        if (!p) throw new Error('No variables to analyse.');
        X.forEach((r, i) => { if (r.length !== p || r.some(v => !Number.isFinite(v))) throw new Error(`Row ${i + 1} has a missing or non-numeric value: fill or remove missing values first.`); });
        return p;
    }

    /** Standardisation to mean 0 and population SD 1 (as scikit-learn's StandardScaler); constant columns are only centred */
    function standardize(X) {
        const p = checkMatrix(X), n = X.length, mean = colMeans(X), sd = new Array(p).fill(0);
        for (const r of X) for (let j = 0; j < p; j++) sd[j] += sq(r[j] - mean[j]) / n;
        for (let j = 0; j < p; j++) { sd[j] = Math.sqrt(sd[j]); if (!(sd[j] > 1e-12 * Math.max(1, Math.abs(mean[j])))) sd[j] = 1; }
        return { Z: X.map(r => r.map((v, j) => (v - mean[j]) / sd[j])), mean, sd };
    }

    // A dissimilarity matrix: n x n, stored flat (D[i * n + j])
    function euclideanMatrix(X) {
        checkMatrix(X);
        const n = X.length, D = new Float64Array(n * n);
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const d = Math.sqrt(sqDist(X[i], X[j])); D[i * n + j] = d; D[j * n + i] = d; }
        return { n, D };
    }

    /**
     * Gower dissimilarity for mixed clinical data (as R's cluster::daisy(metric = "gower")): numbers and ordinal
     * ranks scaled by their range, categories compared as same / different; a variable missing in either patient
     * is left out of that pair (the remaining variables are averaged). vars: [{ type: 'continuous' | 'ordinal' |
     * 'nominal', levels? (ordinal order) }]; rows: arrays of raw values (null for missing).
     */
    function gowerMatrix(rows, vars) {
        const n = rows.length, p = vars.length;
        if (!n || !p) throw new Error('No patients or variables to analyse.');
        const val = vars.map((v, j) => rows.map(r => {
            const x = r[j];
            if (x === null || x === undefined || x === '' || (typeof x === 'number' && !Number.isFinite(x))) return null;
            if (v.type === 'nominal') return String(x);
            if (v.type === 'ordinal') { const k = v.levels ? v.levels.indexOf(String(x)) : Number(x); return k < 0 || !Number.isFinite(k) ? null : k; }
            const num = Number(x); return Number.isFinite(num) ? num : null;
        }));
        const range = vars.map((v, j) => { if (v.type === 'nominal') return 1; const xs = val[j].filter(x => x !== null); if (!xs.length) return 0; let lo = Infinity, hi = -Infinity; for (const x of xs) { if (x < lo) lo = x; if (x > hi) hi = x; } return hi - lo; });
        const D = new Float64Array(n * n);
        let undefinedPairs = 0;
        for (let i = 0; i < n; i++) for (let k = i + 1; k < n; k++) {
            let s = 0, w = 0;
            for (let j = 0; j < p; j++) {
                const a = val[j][i], b = val[j][k];
                if (a === null || b === null) continue;
                if (vars[j].type === 'nominal') { s += a === b ? 0 : 1; w += 1; } else if (range[j] > 0) { s += Math.abs(a - b) / range[j]; w += 1; } else { w += 1; }
            }
            const d = w ? s / w : NaN;
            if (!w) undefinedPairs++;
            D[i * n + k] = d; D[k * n + i] = d;
        }
        if (undefinedPairs) throw new Error(`${undefinedPairs} pair(s) of patients share no recorded variable, so their dissimilarity is undefined: remove patients with very few recorded values.`);
        return { n, D };
    }

    // ------------------------------------------------------------------------------------------- k-means
    // Lloyd's algorithm from given centres, to a fixed point (labels stop changing) or until the centre shift falls
    // below tol (scaled by the mean variance of the data, as scikit-learn). Empty clusters take the point farthest
    // from its centre.
    function lloyd(X, centres, { maxIter = 300, tol = 1e-4 } = {}) {
        const n = X.length, p = X[0].length, k = centres.length;
        let C = centres.map(c => c.slice());
        const mean = colMeans(X); let varSum = 0;
        for (const r of X) for (let j = 0; j < p; j++) varSum += sq(r[j] - mean[j]);
        const tolScaled = tol * varSum / (n * p);
        const labels = new Int32Array(n).fill(-1);
        let iter = 0;
        for (; iter < maxIter; iter++) {
            let changed = 0;
            for (let i = 0; i < n; i++) {
                let best = 0, bd = Infinity;
                for (let c = 0; c < k; c++) { const d = sqDist(X[i], C[c]); if (d < bd) { bd = d; best = c; } }
                if (labels[i] !== best) { labels[i] = best; changed++; }
            }
            const S = Array.from({ length: k }, () => new Array(p).fill(0)), cnt = new Array(k).fill(0);
            for (let i = 0; i < n; i++) { cnt[labels[i]]++; const s = S[labels[i]]; for (let j = 0; j < p; j++) s[j] += X[i][j]; }
            for (let c = 0; c < k; c++) if (!cnt[c]) {
                let far = 0, fd = -1;
                for (let i = 0; i < n; i++) { const d = sqDist(X[i], C[labels[i]]); if (d > fd) { fd = d; far = i; } }
                S[c] = X[far].slice(); cnt[c] = 1; S[labels[far]] = S[labels[far]].map((v, j) => v - X[far][j]); cnt[labels[far]]--; labels[far] = c;
            }
            const next = S.map((s, c) => s.map(v => v / cnt[c]));
            let shift = 0;
            for (let c = 0; c < k; c++) shift += sqDist(next[c], C[c]);
            C = next;
            if (changed === 0 || shift <= tolScaled) { iter++; break; }
        }
        for (let i = 0; i < n; i++) { let best = 0, bd = Infinity; for (let c = 0; c < k; c++) { const d = sqDist(X[i], C[c]); if (d < bd) { bd = d; best = c; } } labels[i] = best; }
        let inertia = 0;
        for (let i = 0; i < n; i++) inertia += sqDist(X[i], C[labels[i]]);
        return { labels: Array.from(labels), centres: C, inertia, iterations: iter };
    }

    // k-means++ seeding (greedy, 2 + floor(ln k) candidates per step, as scikit-learn)
    function kmeansPlusPlus(X, k, rng) {
        const n = X.length, trials = 2 + Math.floor(Math.log(k));
        const C = [X[Math.floor(rng() * n)].slice()];
        let closest = X.map(x => sqDist(x, C[0]));
        for (let c = 1; c < k; c++) {
            const total = closest.reduce((a, b) => a + b, 0);
            let bestIdx = -1, bestPot = Infinity, bestClosest = null;
            for (let t = 0; t < trials; t++) {
                let r = rng() * total, idx = 0;
                while (idx < n - 1 && r >= closest[idx]) { r -= closest[idx]; idx++; }
                const cand = X.map((x, i) => Math.min(closest[i], sqDist(x, X[idx])));
                const pot = cand.reduce((a, b) => a + b, 0);
                if (pot < bestPot) { bestPot = pot; bestIdx = idx; bestClosest = cand; }
            }
            C.push(X[bestIdx].slice()); closest = bestClosest;
        }
        return C;
    }

    /** k-means: the best (lowest within-cluster sum of squares) of nInit k-means++ starts */
    function kmeans(X, k, { nInit = 10, seed = 2024, maxIter = 300, tol = 1e-4 } = {}) {
        checkMatrix(X);
        if (!(k >= 1) || k > X.length) throw new Error(`k must be between 1 and the number of patients (${X.length}).`);
        const rng = createPRNG(seed);
        let best = null;
        for (let s = 0; s < nInit; s++) { const r = lloyd(X, kmeansPlusPlus(X, k, rng), { maxIter, tol }); if (!best || r.inertia < best.inertia - 1e-12) best = r; }
        return relabelBySize(best);
    }

    // Cluster numbers ordered by size (1 = largest), so results are reproducible and easy to read
    function relabelBySize(res) {
        const k = Math.max(...res.labels) + 1, cnt = new Array(k).fill(0);
        res.labels.forEach(l => { cnt[l]++; });
        const order = cnt.map((c, i) => i).sort((a, b) => cnt[b] - cnt[a] || a - b), map = new Array(k);
        order.forEach((old, nw) => { map[old] = nw; });
        const out = { ...res, labels: res.labels.map(l => map[l]) };
        if (res.centres) out.centres = order.map(o => res.centres[o]);
        if (res.medoids) out.medoids = order.map(o => res.medoids[o]);
        if (res.means) { out.means = order.map(o => res.means[o]); out.weights = order.map(o => res.weights[o]); out.covs = order.map(o => res.covs[o]); }
        if (res.resp) out.resp = res.resp.map(r => order.map(o => r[o]));
        return out;
    }

    // ------------------------------------------------------------------------------------------- k-medoids (PAM)
    /**
     * Partitioning around medoids (Kaufman and Rousseeuw): BUILD then SWAP on any dissimilarity matrix (so it
     * works with Gower dissimilarities for mixed data), as R's cluster::pam. Returns medoids (row indices),
     * labels and the objective (mean dissimilarity of each patient to its medoid).
     */
    function pam({ n, D }, k, { maxSwaps = 1000 } = {}) {
        if (!(k >= 1) || k > n) throw new Error(`k must be between 1 and the number of patients (${n}).`);
        const d = (i, j) => D[i * n + j];
        const med = [];
        // BUILD: first the object with the smallest total dissimilarity, then the one giving the largest gain
        let first = 0, fs = Infinity;
        for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += d(i, j); if (s < fs - 1e-12) { fs = s; first = i; } }
        med.push(first);
        const near = new Float64Array(n);
        for (let j = 0; j < n; j++) near[j] = d(j, first);
        while (med.length < k) {
            let bi = -1, bg = -Infinity;
            for (let i = 0; i < n; i++) {
                if (med.includes(i)) continue;
                let g = 0; for (let j = 0; j < n; j++) { const diff = near[j] - d(j, i); if (diff > 0) g += diff; }
                if (g > bg + 1e-12) { bg = g; bi = i; }
            }
            med.push(bi);
            for (let j = 0; j < n; j++) near[j] = Math.min(near[j], d(j, bi));
        }
        const buildObjective = near.reduce((a, b) => a + b, 0) / n;
        // SWAP: best improving exchange of a medoid and a non-medoid, until none improves
        const nearest = () => {
            const d1 = new Float64Array(n), d2 = new Float64Array(n), l1 = new Int32Array(n);
            for (let j = 0; j < n; j++) {
                let a = Infinity, b = Infinity, la = 0;
                med.forEach((m, c) => { const v = d(j, m); if (v < a) { b = a; a = v; la = c; } else if (v < b) b = v; });
                d1[j] = a; d2[j] = b; l1[j] = la;
            }
            return { d1, d2, l1 };
        };
        let swaps = 0;
        for (; swaps < maxSwaps; swaps++) {
            const { d1, d2, l1 } = nearest();
            let best = 0, bm = -1, bh = -1;
            for (let mi = 0; mi < k; mi++) for (let h = 0; h < n; h++) {
                if (med.includes(h)) continue;
                let t = 0;
                for (let j = 0; j < n; j++) {
                    const djh = d(j, h);
                    if (l1[j] === mi) t += Math.min(djh, d2[j]) - d1[j];
                    else if (djh < d1[j]) t += djh - d1[j];
                }
                if (t < best - 1e-12) { best = t; bm = mi; bh = h; }
            }
            if (bm < 0) break;
            med[bm] = bh;
        }
        const { d1, l1 } = nearest();
        return relabelBySize({ medoids: med.slice(), labels: Array.from(l1), objective: d1.reduce((a, b) => a + b, 0) / n, buildObjective, swaps });
    }

    // ------------------------------------------------------------------------------------------- hierarchical
    /**
     * Agglomerative hierarchical clustering by the nearest-neighbour chain algorithm with Lance-Williams updates:
     * 'ward' (Euclidean data: use the Euclidean dissimilarity matrix), 'average', 'complete' or 'single'. Returns
     * the merges in SciPy's linkage format (sorted by height): [a, b, height, size] with new clusters numbered n, n+1, ...
     */
    function hierarchical({ n, D }, method = 'ward') {
        if (!['ward', 'average', 'complete', 'single'].includes(method)) throw new Error(`Unknown linkage "${method}".`);
        if (n < 2) throw new Error('At least two patients are needed.');
        const ward = method === 'ward';
        const M = new Float64Array(n * n);
        for (let i = 0; i < n * n; i++) M[i] = ward ? D[i] * D[i] : D[i];
        const size = new Array(n).fill(1), active = new Array(n).fill(true);
        const merges = [];
        const chain = [];
        let remaining = n;
        while (remaining > 1) {
            if (!chain.length) { let s = 0; while (!active[s]) s++; chain.push(s); }
            const a = chain[chain.length - 1];
            let b = -1, bd = Infinity;
            const prev = chain.length > 1 ? chain[chain.length - 2] : -1;
            for (let j = 0; j < n; j++) {
                if (!active[j] || j === a) continue;
                const v = M[a * n + j];
                if (v < bd || (v === bd && j === prev)) { bd = v; b = j; }
            }
            if (b === prev) {
                chain.pop(); chain.pop();
                const ni = size[a], nj = size[b];
                merges.push({ a, b, h: ward ? Math.sqrt(Math.max(0, bd)) : bd });
                for (let k = 0; k < n; k++) {
                    if (!active[k] || k === a || k === b) continue;
                    const da = M[a * n + k], db = M[b * n + k], nk = size[k];
                    let v;
                    if (method === 'single') v = Math.min(da, db);
                    else if (method === 'complete') v = Math.max(da, db);
                    else if (method === 'average') v = (ni * da + nj * db) / (ni + nj);
                    else v = ((ni + nk) * da + (nj + nk) * db - nk * bd) / (ni + nj + nk);
                    M[b * n + k] = v; M[k * n + b] = v;
                }
                active[a] = false; size[b] = ni + nj; remaining--;
            } else chain.push(b);
        }
        // SciPy numbering: sort merges by height and relabel the merged clusters n, n+1, ...
        merges.sort((x, y) => x.h - y.h);
        const parent = Array.from({ length: n }, (_, i) => i), label = Array.from({ length: n }, (_, i) => i), sz = new Array(n).fill(1);
        const find = x => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
        const Z = merges.map((m, t) => {
            const ra = find(m.a), rb = find(m.b);
            const la = label[ra], lb = label[rb];
            parent[ra] = rb; sz[rb] += sz[ra]; label[rb] = n + t;
            return [Math.min(la, lb), Math.max(la, lb), m.h, sz[rb]];
        });
        return { Z, method };
    }

    /** Cut a hierarchy into k clusters (the first n - k merges) */
    function cutTree({ Z }, n, k) {
        if (!(k >= 1) || k > n) throw new Error(`k must be between 1 and ${n}.`);
        const parent = Array.from({ length: 2 * n }, (_, i) => i);
        const find = x => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
        for (let t = 0; t < n - k; t++) { const [a, b] = Z[t]; parent[find(a)] = n + t; parent[find(b)] = n + t; }
        const roots = new Map(), labels = [];
        for (let i = 0; i < n; i++) { const r = find(i); if (!roots.has(r)) roots.set(r, roots.size); labels.push(roots.get(r)); }
        return relabelBySize({ labels }).labels;
    }

    // ------------------------------------------------------------------------------------------- Gaussian mixtures
    function cholesky(A, p) {
        const L = Array.from({ length: p }, () => new Array(p).fill(0));
        for (let i = 0; i < p; i++) for (let j = 0; j <= i; j++) {
            let s = A[i][j];
            for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
            if (i === j) { if (!(s > 0)) return null; L[i][i] = Math.sqrt(s); } else L[i][j] = s / L[j][j];
        }
        return L;
    }
    // log N(x | mu, Sigma) from the Cholesky factor of Sigma
    function logNormal(x, mu, L, p) {
        const z = new Array(p);
        for (let i = 0; i < p; i++) { let s = x[i] - mu[i]; for (let k = 0; k < i; k++) s -= L[i][k] * z[k]; z[i] = s / L[i][i]; }
        let q = 0, logDet = 0;
        for (let i = 0; i < p; i++) { q += z[i] * z[i]; logDet += Math.log(L[i][i]); }
        return -0.5 * (p * Math.log(2 * Math.PI) + q) - logDet;
    }
    function mStep(X, R, k, covType, reg) {
        const n = X.length, p = X[0].length;
        const Nk = new Array(k).fill(0), means = Array.from({ length: k }, () => new Array(p).fill(0));
        for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) { const r = R[i][c]; Nk[c] += r; for (let j = 0; j < p; j++) means[c][j] += r * X[i][j]; }
        for (let c = 0; c < k; c++) { Nk[c] += 10 * Number.EPSILON; for (let j = 0; j < p; j++) means[c][j] /= Nk[c]; }
        const covs = means.map((mu, c) => {
            const S = Array.from({ length: p }, () => new Array(p).fill(0));
            for (let i = 0; i < n; i++) { const r = R[i][c]; if (!r) continue; for (let a = 0; a < p; a++) { const da = X[i][a] - mu[a]; if (covType === 'diag') S[a][a] += r * da * da; else for (let b = 0; b <= a; b++) S[a][b] += r * da * (X[i][b] - mu[b]); } }
            for (let a = 0; a < p; a++) { for (let b = 0; b <= a; b++) { S[a][b] /= Nk[c]; S[b][a] = S[a][b]; } S[a][a] += reg; }
            return S;
        });
        return { weights: Nk.map(v => v / n), means, covs };
    }
    function eStep(X, params, k) {
        const n = X.length, p = X[0].length;
        const Ls = params.covs.map(S => cholesky(S, p));
        if (Ls.some(L => !L)) throw new Error('A mixture component collapsed (its covariance is singular): try fewer clusters or the diagonal model.');
        const R = new Array(n); let ll = 0;
        for (let i = 0; i < n; i++) {
            const lp = params.means.map((mu, c) => Math.log(params.weights[c]) + logNormal(X[i], mu, Ls[c], p));
            const m = Math.max(...lp); const s = lp.reduce((a, v) => a + Math.exp(v - m), 0); const lse = m + Math.log(s);
            ll += lse; R[i] = lp.map(v => Math.exp(v - lse));
        }
        return { R, ll };
    }
    /** EM from given parameters (or responsibilities) to convergence of the mean log-likelihood (as scikit-learn) */
    function gmmFromParams(X, params, { covType = 'full', reg = 1e-6, maxIter = 500, tol = 1e-6 } = {}) {
        const n = X.length, k = params.means.length;
        let prm = params, prev = -Infinity, ll = -Infinity, iter = 0, converged = false, R;
        for (; iter < maxIter; iter++) {
            const e = eStep(X, prm, k); R = e.R; ll = e.ll / n;
            prm = mStep(X, R, k, covType, reg);
            if (Math.abs(ll - prev) < tol) { converged = true; iter++; break; }
            prev = ll;
        }
        const fin = eStep(X, prm, k);
        return { ...prm, resp: fin.R, logLik: fin.ll, meanLogLik: fin.ll / n, iterations: iter, converged, labels: fin.R.map(r => r.indexOf(Math.max(...r))) };
    }
    /**
     * Gaussian mixture model ('full' or 'diag' covariance), started from k-means (as scikit-learn), best of nInit
     * starts. BIC = -2 log-likelihood + parameters x ln(n) (lower is better).
     */
    function gmm(X, k, { covType = 'full', nInit = 5, seed = 2024, reg = 1e-6, maxIter = 500, tol = 1e-6 } = {}) {
        const p = checkMatrix(X), n = X.length;
        let best = null;
        for (let s = 0; s < nInit; s++) {
            const km = kmeans(X, k, { nInit: 1, seed: seed + s * 7919 });
            const R = km.labels.map(l => Array.from({ length: k }, (_, c) => (c === l ? 1 : 0)));
            try { const r = gmmFromParams(X, mStep(X, R, k, covType, reg), { covType, reg, maxIter, tol }); if (!best || r.logLik > best.logLik) best = r; } catch (e) { if (!best && s === nInit - 1) throw e; }
        }
        if (!best) throw new Error('The mixture model could not be fitted.');
        const nParams = (k - 1) + k * p + (covType === 'diag' ? k * p : k * p * (p + 1) / 2);
        return relabelBySize({ ...best, nParams, bic: -2 * best.logLik + nParams * Math.log(n), aic: -2 * best.logLik + 2 * nParams });
    }

    // ------------------------------------------------------------------------------------------- validity indices
    /** Silhouette (Rousseeuw): per patient and mean; patients alone in their cluster score 0 (as scikit-learn) */
    function silhouette({ n, D }, labels) {
        const k = Math.max(...labels) + 1;
        if (k < 2) return { mean: null, values: [] };
        const size = new Array(k).fill(0); labels.forEach(l => { size[l]++; });
        const values = new Array(n);
        for (let i = 0; i < n; i++) {
            const sums = new Array(k).fill(0);
            for (let j = 0; j < n; j++) if (j !== i) sums[labels[j]] += D[i * n + j];
            const own = labels[i];
            if (size[own] === 1) { values[i] = 0; continue; }
            const a = sums[own] / (size[own] - 1);
            let b = Infinity; for (let c = 0; c < k; c++) if (c !== own && size[c]) b = Math.min(b, sums[c] / size[c]);
            values[i] = (b - a) / Math.max(a, b);
        }
        return { mean: values.reduce((x, y) => x + y, 0) / n, values };
    }
    function centroids(X, labels, k) {
        const p = X[0].length, C = Array.from({ length: k }, () => new Array(p).fill(0)), cnt = new Array(k).fill(0);
        X.forEach((x, i) => { cnt[labels[i]]++; for (let j = 0; j < p; j++) C[labels[i]][j] += x[j]; });
        return { C: C.map((c, l) => c.map(v => v / (cnt[l] || 1))), cnt };
    }
    /** Calinski-Harabasz index (higher = better separated) */
    function calinskiHarabasz(X, labels) {
        const n = X.length, k = Math.max(...labels) + 1;
        if (k < 2 || k >= n) return null;
        const mean = colMeans(X), { C, cnt } = centroids(X, labels, k);
        let B = 0, W = 0;
        for (let c = 0; c < k; c++) B += cnt[c] * sqDist(C[c], mean);
        X.forEach((x, i) => { W += sqDist(x, C[labels[i]]); });
        return W === 0 ? Infinity : (B / (k - 1)) / (W / (n - k));
    }
    /** Davies-Bouldin index (lower = better) */
    function daviesBouldin(X, labels) {
        const k = Math.max(...labels) + 1;
        if (k < 2) return null;
        const { C, cnt } = centroids(X, labels, k), s = new Array(k).fill(0);
        X.forEach((x, i) => { s[labels[i]] += Math.sqrt(sqDist(x, C[labels[i]])) / cnt[labels[i]]; });
        let tot = 0;
        for (let a = 0; a < k; a++) { let m = 0; for (let b = 0; b < k; b++) if (b !== a) { const d = Math.sqrt(sqDist(C[a], C[b])); m = Math.max(m, d > 0 ? (s[a] + s[b]) / d : Infinity); } tot += m; }
        return tot / k;
    }
    /** Within-cluster sum of squares (pooled), used by the gap statistic */
    function withinSS(X, labels) { const k = Math.max(...labels) + 1, { C } = centroids(X, labels, k); let W = 0; X.forEach((x, i) => { W += sqDist(x, C[labels[i]]); }); return W; }

    /** Adjusted Rand index (Hubert and Arabie): agreement between two partitions, 0 = chance, 1 = identical */
    function adjustedRand(a, b) {
        const n = a.length, ka = Math.max(...a) + 1, kb = Math.max(...b) + 1;
        const T = Array.from({ length: ka }, () => new Array(kb).fill(0)), ra = new Array(ka).fill(0), rb = new Array(kb).fill(0);
        for (let i = 0; i < n; i++) { T[a[i]][b[i]]++; ra[a[i]]++; rb[b[i]]++; }
        const c2 = x => x * (x - 1) / 2;
        let sij = 0; T.forEach(r => r.forEach(v => { sij += c2(v); }));
        const sa = ra.reduce((s, v) => s + c2(v), 0), sb = rb.reduce((s, v) => s + c2(v), 0), exp = sa * sb / c2(n), mx = (sa + sb) / 2;
        return mx === exp ? 1 : (sij - exp) / (mx - exp);
    }

    // ------------------------------------------------------------------------------------------- PCA
    // Eigen-decomposition of a symmetric matrix (cyclic Jacobi); eigenvalues in decreasing order
    function symmetricEigen(A) {
        const p = A.length, M = A.map(r => r.slice()), V = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => (i === j ? 1 : 0)));
        for (let sweep = 0; sweep < 100; sweep++) {
            let off = 0; for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) off += M[i][j] * M[i][j];
            if (off < 1e-22) break;
            for (let i = 0; i < p; i++) for (let j = i + 1; j < p; j++) {
                if (Math.abs(M[i][j]) < 1e-300) continue;
                const th = (M[j][j] - M[i][i]) / (2 * M[i][j]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
                for (let k = 0; k < p; k++) { const a = M[k][i], b = M[k][j]; M[k][i] = c * a - s * b; M[k][j] = s * a + c * b; }
                for (let k = 0; k < p; k++) { const a = M[i][k], b = M[j][k]; M[i][k] = c * a - s * b; M[j][k] = s * a + c * b; }
                for (let k = 0; k < p; k++) { const a = V[k][i], b = V[k][j]; V[k][i] = c * a - s * b; V[k][j] = s * a + c * b; }
            }
        }
        const order = Array.from({ length: p }, (_, i) => i).sort((x, y) => M[y][y] - M[x][x]);
        return { values: order.map(i => M[i][i]), vectors: order.map(i => V.map(r => r[i])) };
    }
    /**
     * Principal component analysis of (already standardised, if wanted) data: components (unit loadings, sign
     * fixed so the largest absolute loading is positive), variance explained, and the patients' scores.
     */
    function pca(X, nComp = null) {
        const p = checkMatrix(X), n = X.length, mean = colMeans(X);
        const S = Array.from({ length: p }, () => new Array(p).fill(0));
        for (const r of X) for (let a = 0; a < p; a++) { const da = r[a] - mean[a]; for (let b = 0; b <= a; b++) S[a][b] += da * (r[b] - mean[b]) / (n - 1); }
        for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) S[b][a] = S[a][b];
        const { values, vectors } = symmetricEigen(S);
        const m = Math.min(nComp || p, p);
        const comps = vectors.slice(0, m).map(v => { let big = 0; v.forEach((x, j) => { if (Math.abs(x) > Math.abs(v[big])) big = j; }); return v[big] < 0 ? v.map(x => -x) : v; });
        const total = values.reduce((a, b) => a + Math.max(0, b), 0);
        return { components: comps, variance: values.slice(0, m).map(v => Math.max(0, v)), varianceRatio: values.slice(0, m).map(v => Math.max(0, v) / total), mean,
            scores: X.map(r => comps.map(c => c.reduce((s, w, j) => s + w * (r[j] - mean[j]), 0))) };
    }

    // ------------------------------------------------------------------------------------------- choosing k
    /**
     * Gap statistic (Tibshirani, Walther and Hastie 2001) for k = 1..kMax with k-means: log within-cluster sum of
     * squares compared with B reference datasets drawn uniformly in the box aligned with the data's principal
     * components (as R's cluster::clusGap, spaceH0 = "scaledPCA"). Recommended k: the smallest k with
     * gap(k) >= gap(k+1) - se(k+1) (Tibshirani's 1-SE rule).
     */
    function gapStatistic(X, kMax, { B = 50, seed = 2024, nInit = 5 } = {}) {
        checkMatrix(X);
        const n = X.length, rng = createPRNG(seed);
        const logW = data => { const out = []; for (let k = 1; k <= kMax; k++) out.push(Math.log(k === 1 ? withinSS(data, new Array(n).fill(0)) : kmeans(data, k, { nInit, seed: Math.floor(rng() * 1e9) }).inertia)); return out; };
        const obs = logW(X);
        const P = pca(X), mn = P.mean, comps = P.components;
        const lo = comps.map((_, c) => Math.min(...P.scores.map(s => s[c]))), hi = comps.map((_, c) => Math.max(...P.scores.map(s => s[c])));
        const ref = [];
        for (let b = 0; b < B; b++) {
            const Y = Array.from({ length: n }, () => { const sc = lo.map((l, c) => l + rng() * (hi[c] - l)); return mn.map((m, j) => m + comps.reduce((s, v, c) => s + v[j] * sc[c], 0)); });
            ref.push(logW(Y));
        }
        const res = obs.map((o, i) => { const v = ref.map(r => r[i]), m = v.reduce((a, x) => a + x, 0) / B, sd = Math.sqrt(v.reduce((a, x) => a + (x - m) ** 2, 0) / B); return { k: i + 1, logW: o, expected: m, gap: m - o, se: sd * Math.sqrt(1 + 1 / B) }; });
        let kBest = kMax;
        for (let i = 0; i < kMax - 1; i++) if (res[i].gap >= res[i + 1].gap - res[i + 1].se) { kBest = i + 1; break; }
        return { table: res, kBest };
    }

    // ------------------------------------------------------------------------------------------- stability
    /**
     * Cluster stability by the bootstrap (Hennig 2007, as fpc::clusterboot): the clustering is repeated on B
     * resamples of patients; each original cluster's stability is its mean Jaccard similarity with the most
     * similar cluster in the resample (on the patients drawn). Mean Jaccard >= 0.75: stable; 0.6-0.75: indicates
     * patterns; < 0.5: dissolved (not reproducible). clusterFn(indices) -> labels for those patients.
     * groups (optional, one key per patient row): whole patients are resampled.
     */
    function bootstrapStability(n, labels, clusterFn, { B = 100, seed = 2024, groups = null } = {}) {
        const rng = createPRNG(seed), k = Math.max(...labels) + 1;
        const clusters = Array.from({ length: k }, (_, c) => labels.map((l, i) => (l === c ? i : -1)).filter(i => i >= 0));
        const byGroup = groups ? [...groups.reduce((m, g, i) => { const key = String(g); if (!m.has(key)) m.set(key, []); m.get(key).push(i); return m; }, new Map()).values()] : null;
        const jac = Array.from({ length: k }, () => []), aris = [];
        let failed = 0;
        for (let b = 0; b < B; b++) {
            const idx = byGroup ? [].concat(...Array.from({ length: byGroup.length }, () => byGroup[Math.floor(rng() * byGroup.length)])) : Array.from({ length: n }, () => Math.floor(rng() * n));
            let lb;
            try { lb = clusterFn(idx); } catch (e) { failed++; continue; }
            // the resample's clusters, on the distinct patients drawn (a patient drawn twice keeps its first label)
            const lab = new Map(); idx.forEach((i, t) => { if (!lab.has(i)) lab.set(i, lb[t]); });
            const kb = Math.max(...lab.values()) + 1;
            const bc = Array.from({ length: kb }, () => new Set());
            lab.forEach((l, i) => { bc[l].add(i); });
            clusters.forEach((C, c) => {
                const inC = C.filter(i => lab.has(i));
                if (!inC.length) return;
                const setC = new Set(inC);
                let best = 0;
                bc.forEach(S => { let inter = 0; S.forEach(i => { if (setC.has(i)) inter++; }); const uni = setC.size + S.size - inter; if (uni) best = Math.max(best, inter / uni); });
                jac[c].push(best);
            });
            const shared = [...lab.keys()];
            aris.push(adjustedRand(shared.map(i => labels[i]), shared.map(i => lab.get(i))));
        }
        const mean = a => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
        return { perCluster: jac.map((a, c) => ({ cluster: c, size: clusters[c].length, meanJaccard: mean(a), dissolved: a.filter(v => v < 0.5).length / (a.length || 1) })), meanAdjustedRand: mean(aris), resamples: B - failed, failed };
    }

    // ------------------------------------------------------------------------------------------- is there structure?
    /**
     * Test of "no real clusters": the separation of the clustering (R^2 = between / total sum of squares, for
     * k-means on numerical data) is compared with clusterings of B datasets without clusters but with the same
     * means and covariance (a single multivariate normal), as in SigClust (Liu et al. 2008). Returns the observed
     * R^2, the null distribution and a Monte Carlo p-value; a small p means more separation than one Gaussian
     * cloud with the same covariance gives by chance. Skewed or heavy-tailed data without clusters can still give
     * small p-values (the test is about departure from one Gaussian cloud).
     */
    function nullClusterTest(X, k, { B = 100, seed = 2024, nInit = 5 } = {}) {
        checkMatrix(X);
        const n = X.length, p = X[0].length, rng = createPRNG(seed);
        if (n <= p + 1) throw new Error('The test needs more patients than variables.');
        const r2 = data => { const tss = withinSS(data, new Array(n).fill(0)); return tss > 0 ? 1 - kmeans(data, k, { nInit, seed: Math.floor(rng() * 1e9) }).inertia / tss : 0; };
        const obs = r2(X);
        // Each reference dataset is Gaussian and transformed to have exactly the data's mean and sample covariance:
        // conditioning on the covariance makes the test exact for Gaussian data (drawing from a Gaussian with the
        // sample covariance instead is conservative, because the largest sample eigenvalues are biased upwards)
        const P = pca(X), U = P.components, sdObs = P.variance.map(v => Math.sqrt(v));
        const nulls = [];
        for (let b = 0; b < B; b++) {
            const Y = Array.from({ length: n }, () => Array.from({ length: p }, () => gaussian(rng)));
            const Q = pca(Y), sdY = Q.variance.map(v => Math.sqrt(v));
            // scores of Y on its own principal axes, scaled to unit variance, then given the data's axes and variances
            const Yt = Q.scores.map(sc => { const z = sc.map((v, c) => (sdY[c] > 0 ? v / sdY[c] : 0) * sdObs[c]); return P.mean.map((m, j) => m + U.reduce((acc, u, c) => acc + u[j] * z[c], 0)); });
            nulls.push(r2(Yt));
        }
        const ge = nulls.filter(v => v >= obs).length;
        return { observed: obs, nullMean: nulls.reduce((a, b) => a + b, 0) / B, nullQuantile95: nulls.slice().sort((a, b) => a - b)[Math.floor(0.95 * (B - 1))], pValue: (1 + ge) / (1 + B), B, p };
    }

    const ClusterEngine = { createPRNG, standardize, euclideanMatrix, gowerMatrix, lloyd, kmeansPlusPlus, kmeans, pam, hierarchical, cutTree, gmm, gmmFromParams, mStep,
        silhouette, calinskiHarabasz, daviesBouldin, withinSS, adjustedRand, symmetricEigen, pca, gapStatistic, bootstrapStability, nullClusterTest, relabelBySize };
    root.ClusterEngine = ClusterEngine;
})(typeof window !== 'undefined' ? window : globalThis);
