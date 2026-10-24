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
     * weights: null (equal, classic Gower), an array, or 'balanced': each variable weighted by 1 / its mean
     * contribution over all pairs, so that every variable contributes equally on average (Hennig & Liao 2013).
     * Without balancing, a yes/no variable (mean contribution about 0.5) outweighs a measurement (about 0.2-0.3)
     * and can split patients into artefactual groups. Returns { n, D, weights }.
     */
    function gowerMatrix(rows, vars, { weights = null } = {}) {
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
        const dj = (j, a, b) => (vars[j].type === 'nominal' ? (a === b ? 0 : 1) : range[j] > 0 ? Math.abs(a - b) / range[j] : 0);
        let wt = new Array(p).fill(1);
        if (Array.isArray(weights)) wt = weights.slice();
        else if (weights === 'balanced') {
            wt = vars.map((v, j) => {
                let s = 0, c = 0;
                for (let i = 0; i < n; i++) { const a = val[j][i]; if (a === null) continue; for (let k = i + 1; k < n; k++) { const b = val[j][k]; if (b === null) continue; s += dj(j, a, b); c++; } }
                return s > 0 ? c / s : 0;   // a constant variable gets no weight
            });
        }
        const D = new Float64Array(n * n);
        let undefinedPairs = 0;
        for (let i = 0; i < n; i++) for (let k = i + 1; k < n; k++) {
            let s = 0, w = 0;
            for (let j = 0; j < p; j++) {
                const a = val[j][i], b = val[j][k];
                if (a === null || b === null || !(wt[j] > 0)) continue;
                s += wt[j] * dj(j, a, b); w += wt[j];
            }
            const d = w ? s / w : NaN;
            if (!w) undefinedPairs++;
            D[i * n + k] = d; D[k * n + i] = d;
        }
        if (undefinedPairs) throw new Error(`${undefinedPairs} pair(s) of patients share no recorded variable, so their dissimilarity is undefined: remove patients with very few recorded values.`);
        return { n, D, weights: wt };
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
        const p = A.length, M = A.map(r => r.slice()), V = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (__, j) => (i === j ? 1 : 0)));
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
        const jac = Array.from({ length: k }, () => []), sizes = Array.from({ length: k }, () => []), aris = [];
        // Patient-level stability: in how many resamples containing the patient it falls in the resample cluster that
        // best matches its own cluster (largest Jaccard)
        const seen = new Float64Array(n), same = new Float64Array(n);
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
                let best = 0, bestS = null;
                bc.forEach(S => { let inter = 0; S.forEach(i => { if (setC.has(i)) inter++; }); const uni = setC.size + S.size - inter; if (uni && inter / uni > best) { best = inter / uni; bestS = S; } });
                jac[c].push(best);
                sizes[c].push(bestS ? bestS.size / lab.size : 0);
                inC.forEach(i => { seen[i]++; if (bestS && bestS.has(i)) same[i]++; });
            });
            const shared = [...lab.keys()];
            aris.push(adjustedRand(shared.map(i => labels[i]), shared.map(i => lab.get(i))));
        }
        const mean = a => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
        const sd = a => { if (a.length < 2) return null; const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1)); };
        const quant = (a, f) => { if (!a.length) return null; const t = a.slice().sort((x, y) => x - y), h = (t.length - 1) * f, l = Math.floor(h); return t[l] + (t[Math.ceil(h)] - t[l]) * (h - l); };
        return {
            perCluster: jac.map((a, c) => {
                const m = mean(a), s = sd(a), se = s !== null ? s / Math.sqrt(a.length) : null;
                // ciMean: Monte Carlo 95% interval of the mean Jaccard (how precisely the B resamples estimate it);
                // range95: the middle 95% of the Jaccard values of single resamples; shareRange: share of the resampled
                // patients in the matching cluster (compare with size / n)
                return { cluster: c, size: clusters[c].length, meanJaccard: m, sdJaccard: s, ciMean: se !== null ? [m - 1.96 * se, m + 1.96 * se] : null,
                    range95: [quant(a, 0.025), quant(a, 0.975)], shareRange: [quant(sizes[c], 0.025), quant(sizes[c], 0.975)], dissolved: a.filter(v => v < 0.5).length / (a.length || 1) };
            }),
            patientStability: Array.from(same, (v, i) => (seen[i] ? v / seen[i] : null)),
            meanAdjustedRand: mean(aris), resamples: B - failed, failed };
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

    // Inverse of the standard normal CDF (Acklam's rational approximation, relative error < 1.2e-9)
    function normInv(p) {
        const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
        const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
        const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549671010599464e+00, 4.374664141464968e+00, 2.938163982698783e+00];
        const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
        const pl = 0.02425;
        if (p < pl) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
        if (p > 1 - pl) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
        const q = p - 0.5, r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
    function normCdf(x) { // Abramowitz-Stegun 7.1.26 via erf, accurate to ~1e-7
        const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2), y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x / 2);
        return 0.5 * (1 + (x >= 0 ? y : -y));
    }

    /**
     * Test of "no real sub-types" for clinical data with yes/no, graded and skewed variables: the reference datasets
     * come from a Gaussian copula, i.e. one continuous latent population with the data's rank correlations, mapped
     * onto each variable's own observed distribution (a yes/no variable stays yes/no with the same prevalence).
     * Discrete variables make any data look "clustered" against a Gaussian cloud; this reference has exactly the
     * same discreteness, so only sub-types beyond it count. X: the numerically coded data (before standardisation);
     * the clustering statistic is k-means R^2 on standardised data, as for the observed data.
     */
    function copulaNullTest(X0, k, { B = 100, seed = 2024, nInit = 5, groups = [] } = {}) {
        checkMatrix(X0);
        const n = X0.length, rng = createPRNG(seed);
        const stat = data => { const Z = standardize(data).Z, tss = withinSS(Z, new Array(n).fill(0)); return tss > 0 ? 1 - kmeans(Z, k, { nInit, seed: Math.floor(rng() * 1e9) }).inertia / tss : 0; };
        const obs = stat(X0);
        // Categories with several levels (groups: the indicator columns of one variable) are simulated from a single
        // latent value cut at the observed category shares, so every patient has exactly one category. The levels are
        // ordered along the first principal component of the other columns, which keeps their main association.
        const inGroup = new Set(groups.flat()), others = X0[0].map((_, j) => j).filter(j => !inGroup.has(j));
        let pc1 = null;
        if (groups.length && others.length >= 1) { const Zo = standardize(X0.map(r => others.map(j => r[j]))).Z, P1 = pca(Zo, 1); pc1 = P1.scores.map(sv => sv[0]); }
        const gInfo = groups.map(cols => {
            const pattern = r => cols.map(c => r[c]).join(','), pats = new Map();
            X0.forEach((r, i) => { const key = pattern(r); if (!pats.has(key)) pats.set(key, { key, vals: cols.map(c => r[c]), n: 0, s: 0 }); const e = pats.get(key); e.n++; e.s += pc1 ? pc1[i] : 0; });
            const levels = [...pats.values()].sort((a, b) => a.s / a.n - b.s / b.n);
            return { cols, levels, code: r => levels.findIndex(l => l.key === pattern(r)) };
        });
        // Reduced matrix: other columns, then one ordered code per category
        const toReduced = r => [...others.map(j => r[j]), ...gInfo.map(g => g.code(r))];
        const fromReduced = z => { const out = new Array(X0[0].length); others.forEach((j, q) => { out[j] = z[q]; }); gInfo.forEach((g, q) => { const lv = g.levels[Math.round(z[others.length + q])]; g.cols.forEach((c, t) => { out[c] = lv.vals[t]; }); }); return out; };
        const X = groups.length ? X0.map(toReduced) : X0, p = X[0].length;
        // NORTA ("normal to anything"): latent Gaussian correlations adjusted until data mapped onto each variable's
        // observed distribution reproduce the observed (Pearson) correlations, the quantity the statistic depends on
        const cols = Array.from({ length: p }, (_, j) => X.map(r => r[j]));
        const sorted = cols.map(col => col.slice().sort((a, b) => a - b));
        const pearson = M => { const m = M.length, mean = new Array(p).fill(0); M.forEach(r => r.forEach((v, j) => { mean[j] += v / m; }));
            const S = Array.from({ length: p }, () => new Array(p).fill(0)); M.forEach(r => { for (let a = 0; a < p; a++) for (let b = 0; b <= a; b++) S[a][b] += (r[a] - mean[a]) * (r[b] - mean[b]); });
            return S.map((row, a) => row.map((v, b) => { const x = b <= a ? v : S[b][a]; return S[a][a] > 0 && S[b][b] > 0 ? x / Math.sqrt(S[a][a] * S[b][b]) : (a === b ? 1 : 0); })); };
        const target = pearson(X);
        const mapRow = (z) => z.map((v, j) => sorted[j][Math.min(n - 1, Math.floor(normCdf(v) * n))]);
        // Cholesky factor of a correlation matrix; if it is not positive definite (as with the indicators of one
        // category, which can never be 1 together), first project it onto the nearest valid correlation matrix:
        // eigenvalues clipped at 1e-4, then the diagonal rescaled to 1
        const project = R => {
            const e = symmetricEigen(R), lam = e.values.map(v => Math.max(v, 1e-4));
            const A = Array.from({ length: p }, (_, a) => Array.from({ length: p }, (__, b) => e.vectors.reduce((s, v, q) => s + v[a] * v[b] * lam[q], 0)));
            return A.map((row, a) => row.map((v, b) => v / Math.sqrt(A[a][a] * A[b][b])));
        };
        const factor = R => cholesky(R, p) || cholesky(project(R), p) || cholesky(project(project(R)).map((r, a) => r.map((v, b) => (a === b ? v + 1e-6 : v))), p);
        const drawLatent = (L, r) => { const e = Array.from({ length: p }, () => gaussian(r)), z = new Array(p).fill(0); for (let i = 0; i < p; i++) for (let j = 0; j <= i; j++) z[i] += L[i][j] * e[j]; return z; };
        let Rl = target.map(r => r.slice()), L = factor(Rl);
        const crng = createPRNG(seed + 1), m = Math.max(4000, 20 * n);
        for (let it = 0; it < 12 && L; it++) {
            const ref = pearson(Array.from({ length: m }, () => mapRow(drawLatent(L, crng))));
            let worst = 0;
            Rl = Rl.map((row, a) => row.map((v, b) => { if (a === b) return 1; const d = target[a][b] - ref[a][b]; worst = Math.max(worst, Math.abs(d)); return Math.max(-0.999, Math.min(0.999, v + d)); }));
            if (!cholesky(Rl, p)) Rl = project(Rl);
            L = factor(Rl);
            if (worst < 0.005) break;
        }
        if (!L) throw new Error('The correlation of the variables could not be modelled.');
        const nulls = [];
        for (let b = 0; b < B; b++) {
            const Y = Array.from({ length: n }, () => mapRow(drawLatent(L, rng)));
            nulls.push(stat(groups.length ? Y.map(fromReduced) : Y));
        }
        const ge = nulls.filter(v => v >= obs).length;
        return { observed: obs, nullMean: nulls.reduce((a, b) => a + b, 0) / B, nullQuantile95: nulls.slice().sort((a, b) => a - b)[Math.floor(0.95 * (B - 1))], pValue: (1 + ge) / (1 + B), B, p: X0[0].length, reference: 'copula' };
    }

    // ------------------------------------------------------------------------------------------- describing clusters
    // Regularised upper incomplete gamma Q(a, x) (Numerical Recipes: series or continued fraction)
    function gammaln(x) { const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]; let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let ser = 1.000000000190015; for (const cj of c) ser += cj / ++y; return -t + Math.log(2.5066282746310005 * ser / x); }
    function gammaQ(a, x) {
        if (x <= 0) return 1;
        if (x < a + 1) { let sum = 1 / a, del = sum, ap = a; for (let i = 0; i < 1000; i++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-15) break; } return 1 - sum * Math.exp(-x + a * Math.log(x) - gammaln(a)); }
        let b = x + 1 - a, c = 1 / 1e-300, d = 1 / b, h = d;
        for (let i = 1; i < 1000; i++) { const an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300; c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-15) break; }
        return Math.exp(-x + a * Math.log(x) - gammaln(a)) * h;
    }
    const chiSquareP = (x, df) => (df > 0 ? gammaQ(df / 2, x / 2) : null);
    /** Pearson chi-square test of independence (no continuity correction) for categories x groups */
    function chiSquareTest(values, groups) {
        const cats = [...new Set(values)], gs = [...new Set(groups)];
        if (cats.length < 2 || gs.length < 2) return { statistic: null, df: null, p: null };
        const T = cats.map(c => gs.map(g => values.filter((v, i) => v === c && groups[i] === g).length));
        const n = values.length, rs = T.map(r => r.reduce((a, b) => a + b, 0)), cs = gs.map((_, j) => T.reduce((a, r) => a + r[j], 0));
        let x2 = 0, minExp = Infinity;
        T.forEach((r, i) => r.forEach((o, j) => { const e = rs[i] * cs[j] / n; minExp = Math.min(minExp, e); x2 += (o - e) ** 2 / e; }));
        const df = (cats.length - 1) * (gs.length - 1);
        return { statistic: x2, df, p: chiSquareP(x2, df), minExpected: minExp };
    }
    /** Kruskal-Wallis test (with the correction for ties), as scipy.stats.kruskal */
    function kruskalWallis(values, groups) {
        const n = values.length, gs = [...new Set(groups)];
        if (gs.length < 2) return { statistic: null, df: null, p: null };
        const idx = values.map((v, i) => i).sort((a, b) => values[a] - values[b]), rank = new Array(n);
        let ties = 0;
        for (let i = 0; i < n;) { let j = i; while (j + 1 < n && values[idx[j + 1]] === values[idx[i]]) j++; const r = (i + j) / 2 + 1, t = j - i + 1; ties += t ** 3 - t; for (let q = i; q <= j; q++) rank[idx[q]] = r; i = j + 1; }
        let H = 0;
        gs.forEach(g => { const rr = rank.filter((_, i) => groups[i] === g); const m = rr.length; if (m) H += (rr.reduce((a, b) => a + b, 0)) ** 2 / m; });
        H = 12 / (n * (n + 1)) * H - 3 * (n + 1);
        const corr = 1 - ties / (n ** 3 - n);
        if (corr <= 0) return { statistic: null, df: gs.length - 1, p: null };
        H /= corr;
        return { statistic: H, df: gs.length - 1, p: chiSquareP(H, gs.length - 1) };
    }

    // ------------------------------------------------------------------------------------------- latent class analysis
    /**
     * Latent class analysis for mixed clinical data (latent profile analysis for numbers): within each class the
     * variables are independent (local independence); numbers are normal with class-specific mean and variance
     * (mclust model "VVI"), grades and categories have class-specific proportions (as R's poLCA, grades treated as
     * categories). Missing values are left out of each patient's likelihood (full-information, valid when missing
     * at random), so no imputation is needed. EM from nInit random starts (random hard assignment), best kept.
     * rows: patients x variables (raw values, null for missing); vars: [{ type, levels }].
     * Returns labels (modal class, 0 = largest), posterior probabilities, log-likelihood on the original scale,
     * number of parameters, BIC, relative entropy, and the model (for classifying new patients).
     */
    function lca(rows, vars, k, { nInit = 10, seed = 2024, maxIter = 2000, tol = 1e-10, reg = 1e-6 } = {}) {
        const n = rows.length, p = vars.length;
        if (k < 1 || k > n) throw new Error('Invalid number of classes.');
        const levels = vars.map((v, j) => {
            if (v.type === 'continuous') return null;
            if (v.levels) return v.levels.map(String);
            return [...new Set(rows.map(r => codeValue(r[j], v)).filter(x => x !== null))].sort();
        });
        // Coded data: z-scores for numbers, level index for grades and categories, null if missing
        const mean = new Array(p).fill(0), sd = new Array(p).fill(1), nObs = new Array(p).fill(0);
        const C = vars.map((v, j) => rows.map(r => {
            const x = codeValue(r[j], v);
            if (x === null) return null;
            if (v.type === 'continuous') return x;
            const l = levels[j].indexOf(String(v.type === 'ordinal' && v.levels ? v.levels[x] : x));
            return l < 0 ? null : l;
        }));
        vars.forEach((v, j) => {
            const xs = C[j].filter(x => x !== null); nObs[j] = xs.length;
            if (!xs.length) throw new Error(`Variable ${j + 1} has no recorded values.`);
            if (v.type !== 'continuous') return;
            const m = xs.reduce((a, b) => a + b, 0) / xs.length, s = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length) || 1;
            mean[j] = m; sd[j] = s; C[j] = C[j].map(x => (x === null ? null : (x - m) / s));
        });
        const cont = vars.map(v => v.type === 'continuous');
        const LOG2PI = Math.log(2 * Math.PI);
        function lcaM(R) {
            const pi = new Array(k).fill(0), mu = [], va = [], pr = [];
            for (let c = 0; c < k; c++) {
                let wc = 0; for (let i = 0; i < n; i++) wc += R[i][c];
                pi[c] = Math.max(wc / n, 1e-300);
                mu.push(new Array(p).fill(0)); va.push(new Array(p).fill(1)); pr.push(new Array(p).fill(null));
                for (let j = 0; j < p; j++) {
                    if (cont[j]) {
                        let sw = 0, s1 = 0;
                        for (let i = 0; i < n; i++) { const x = C[j][i]; if (x === null) continue; sw += R[i][c]; s1 += R[i][c] * x; }
                        const m = sw > 0 ? s1 / sw : 0; let s2 = 0;
                        for (let i = 0; i < n; i++) { const x = C[j][i]; if (x === null) continue; s2 += R[i][c] * (x - m) ** 2; }
                        mu[c][j] = m; va[c][j] = (sw > 0 ? s2 / sw : 1) + reg;
                    } else {
                        const L = levels[j].length, cnt = new Array(L).fill(0); let sw = 0;
                        for (let i = 0; i < n; i++) { const x = C[j][i]; if (x === null) continue; cnt[x] += R[i][c]; sw += R[i][c]; }
                        pr[c][j] = cnt.map(v => (sw > 0 ? v / sw : 1 / L));
                    }
                }
            }
            return { pi, mu, va, pr };
        }
        function lcaE(par) {
            // Logarithms of the parameters once per iteration
            const lpi = par.pi.map(Math.log), lnorm = par.va.map(row => row.map(v => LOG2PI + Math.log(v)));
            const lpr = par.pr.map(row => row.map(q => (q ? q.map(v => Math.log(Math.max(v, 1e-300))) : null)));
            let ll = 0; const R = [];
            for (let i = 0; i < n; i++) {
                const lp = []; let mx = -Infinity;
                for (let c = 0; c < k; c++) {
                    let v = lpi[c];
                    for (let j = 0; j < p; j++) {
                        const x = C[j][i]; if (x === null) continue;
                        if (cont[j]) { const d = x - par.mu[c][j]; v += -0.5 * (lnorm[c][j] + d * d / par.va[c][j]); } else v += lpr[c][j][x];
                    }
                    lp.push(v); if (v > mx) mx = v;
                }
                let se = 0; for (let c = 0; c < k; c++) se += Math.exp(lp[c] - mx);
                const lse = mx + Math.log(se); ll += lse;
                R.push(lp.map(v => Math.exp(v - lse)));
            }
            return { R, ll };
        }
        const rng = createPRNG(seed);
        // Start: k patients chosen at random as seeds; every patient is assigned to the most similar seed
        // (mean over recorded variables of the absolute z-score difference, capped at 2, or category mismatch)
        const dissim = (a, b) => { let s = 0, w = 0; for (let j = 0; j < p; j++) { const x = C[j][a], y = C[j][b]; if (x === null || y === null) continue; s += cont[j] ? Math.min(2, Math.abs(x - y)) / 2 : (x === y ? 0 : 1); w++; } return w ? s / w : 1; };
        let best = null;
        for (let s = 0; s < nInit; s++) {
            const seeds = []; while (seeds.length < k) { const c = Math.floor(rng() * n); if (!seeds.includes(c)) seeds.push(c); }
            const R0 = Array.from({ length: n }, (_, i) => { const d = seeds.map(sd0 => dissim(i, sd0)); let b = 0; d.forEach((v, c) => { if (v < d[b] || (v === d[b] && rng() < 0.5)) b = c; }); const r = new Array(k).fill(0); r[b] = 1; return r; });
            let par = lcaM(R0), prev = -Infinity, cur = null, it = 0, converged = false;
            for (it = 0; it < maxIter; it++) {
                cur = lcaE(par);
                if (cur.ll - prev < tol * Math.abs(cur.ll)) { converged = true; break; }
                prev = cur.ll; par = lcaM(cur.R);
            }
            if (!best || cur.ll > best.ll + 1e-9) best = { ll: cur.ll, R: cur.R, par, converged, iterations: it };
        }
        // Order classes by size (modal assignment), largest first
        const modal = best.R.map(r => r.reduce((b, v, c) => (v > r[b] ? c : b), 0));
        const size = new Array(k).fill(0); modal.forEach(l => { size[l]++; });
        const order = size.map((_, c) => c).sort((a, b) => size[b] - size[a] || a - b), inv = new Array(k); order.forEach((c, r) => { inv[c] = r; });
        const posterior = best.R.map(r => order.map(c => r[c]));
        const labels = modal.map(l => inv[l]);
        const par = { pi: order.map(c => best.par.pi[c]), mu: order.map(c => best.par.mu[c]), va: order.map(c => best.par.va[c]), pr: order.map(c => best.par.pr[c]) };
        let logLik = best.ll; vars.forEach((v, j) => { if (cont[j]) logLik -= nObs[j] * Math.log(sd[j]); });
        const nParams = (k - 1) + k * vars.reduce((a, v, j) => a + (cont[j] ? 2 : levels[j].length - 1), 0);
        let ent = 0; posterior.forEach(r => r.forEach(v => { if (v > 0) ent -= v * Math.log(v); }));
        return { labels, posterior, logLik, nParams, bic: -2 * logLik + nParams * Math.log(n), entropy: k > 1 ? 1 - ent / (n * Math.log(k)) : 1,
            converged: best.converged, iterations: best.iterations,
            model: { k, vars, levels, mean, sd, pi: par.pi, mu: par.mu, va: par.va, pr: par.pr } };
    }
    /** Class membership probabilities of a new patient (raw values, null for missing) under a fitted LCA model */
    function lcaPosterior(model, x) {
        const { k, vars, levels, mean, sd } = model, lp = [];
        for (let c = 0; c < k; c++) {
            let s = Math.log(model.pi[c]);
            vars.forEach((v, j) => {
                const a = codeValue(x[j], v); if (a === null) return;
                if (v.type === 'continuous') { const z = (a - mean[j]) / sd[j], va = model.va[c][j]; s += -0.5 * (Math.log(2 * Math.PI) + Math.log(va) + (z - model.mu[c][j]) ** 2 / va); } else {
                    const l = levels[j].indexOf(String(v.type === 'ordinal' && v.levels ? v.levels[a] : a));
                    if (l >= 0) s += Math.log(Math.max(model.pr[c][j][l], 1e-300));
                }
            });
            lp.push(s);
        }
        const mx = Math.max(...lp), se = lp.reduce((a, v) => a + Math.exp(v - mx), 0);
        return lp.map(v => Math.exp(v - mx) / se);
    }

    // ------------------------------------------------------------------------------------------- using the clusters
    // Coded value of one variable for the tree and Gower distances: number, grade index or category string; null if missing.
    function codeValue(x, v) {
        if (x === null || x === undefined || x === '' || (typeof x === 'number' && !Number.isFinite(x))) return null;
        if (v.type === 'nominal') return String(x).trim();
        if (v.type === 'ordinal') { const k = v.levels ? v.levels.indexOf(String(x).trim()) : Number(x); return k < 0 || !Number.isFinite(k) ? null : k; }
        const num = Number(x); return Number.isFinite(num) ? num : null;
    }
    /** Gower dissimilarity of one new patient to each reference patient, with the variable ranges of the reference data */
    function gowerDistances(x, rows, vars, weights = null) {
        const val = vars.map((v, j) => rows.map(r => codeValue(r[j], v))), xv = vars.map((v, j) => codeValue(x[j], v));
        const range = vars.map((v, j) => { if (v.type === 'nominal') return 1; const xs = val[j].filter(a => a !== null); return xs.length ? Math.max(...xs) - Math.min(...xs) : 0; });
        return rows.map((_, i) => {
            let s = 0, w = 0;
            vars.forEach((v, j) => {
                const a = xv[j], b = val[j][i], wj = weights ? weights[j] : 1;
                if (a === null || b === null || !(wj > 0)) return;
                w += wj; if (v.type === 'nominal') s += wj * (a === b ? 0 : 1); else if (range[j] > 0) s += wj * Math.min(1, Math.abs(a - b) / range[j]);
            });
            return w ? s / w : NaN;
        });
    }
    /**
     * Classification tree (CART, Gini impurity) that predicts the cluster from the clustering variables, to give
     * short rules clinicians can apply. Numbers and grades split at a threshold (midpoint between neighbouring
     * observed values, as scikit-learn; grades at a grade); categories split one category against the rest.
     * (merge = false keeps them, to show sub-groups classified with different confidence.)
     * Missing values: a split is chosen on the patients with that value recorded (improvement weighted by the
     * recorded fraction, as C4.5), and patients without it follow the larger branch. Sibling leaves that predict
     * the same cluster are merged afterwards (predictions unchanged, rules shorter).
     * rows: patients x variables (raw values); vars: [{ name, type: 'continuous'|'ordinal'|'nominal', levels }].
     */
    function classificationTree(rows, vars, labels, { maxDepth = 3, minLeaf = 5, merge = true } = {}) {
        const k = Math.max(...labels) + 1, val = vars.map((v, j) => rows.map(r => codeValue(r[j], v)));
        const counts = idx => { const c = new Array(k).fill(0); idx.forEach(i => { c[labels[i]]++; }); return c; };
        const gini = c => { const n = c.reduce((a, b) => a + b, 0); return n ? 1 - c.reduce((a, x) => a + (x / n) ** 2, 0) : 0; };
        const argmax = c => c.reduce((b, x, i) => (x > c[b] ? i : b), 0);
        function best(idx) {
            let top = null;
            const parentN = idx.length;
            vars.forEach((v, j) => {
                const obs = idx.filter(i => val[j][i] !== null);
                if (obs.length < 2 * minLeaf) return;
                const cAll = counts(obs), gParent = gini(cAll), frac = obs.length / parentN;
                const consider = (test, desc) => {
                    const L = obs.filter(i => test(val[j][i])), R = obs.filter(i => !test(val[j][i]));
                    if (L.length < minLeaf || R.length < minLeaf) return;
                    const gain = frac * (gParent - (L.length * gini(counts(L)) + R.length * gini(counts(R))) / obs.length);
                    if (gain > 1e-12 && (!top || gain > top.gain + 1e-12)) top = { gain, j, ...desc, test };
                };
                if (v.type === 'nominal') {
                    [...new Set(obs.map(i => val[j][i]))].sort().forEach(lv => consider(a => a === lv, { kind: 'is', level: lv }));
                } else {
                    // Sorted sweep for thresholds
                    const order = obs.slice().sort((a, b) => val[j][a] - val[j][b]), cL = new Array(k).fill(0), cR = cAll.slice();
                    const nO = order.length;
                    for (let t = 0; t < nO - 1; t++) {
                        cL[labels[order[t]]]++; cR[labels[order[t]]]--;
                        const a = val[j][order[t]], b = val[j][order[t + 1]];
                        if (a === b || t + 1 < minLeaf || nO - t - 1 < minLeaf) continue;
                        const gain = frac * (gParent - ((t + 1) * gini(cL) + (nO - t - 1) * gini(cR)) / nO);
                        if (gain > 1e-12 && (!top || gain > top.gain + 1e-12)) {
                            const thr = v.type === 'ordinal' ? a : (a + b) / 2;
                            top = { gain, j, kind: 'le', threshold: thr, test: x => x <= thr };
                        }
                    }
                }
            });
            return top;
        }
        function grow(idx, depth) {
            const c = counts(idx), node = { n: idx.length, counts: c, predict: argmax(c) };
            if (depth >= maxDepth || gini(c) < 1e-12 || idx.length < 2 * minLeaf) return node;
            const sp = best(idx);
            if (!sp) return node;
            const j = sp.j, obs = idx.filter(i => val[j][i] !== null), miss = idx.filter(i => val[j][i] === null);
            const L = obs.filter(i => sp.test(val[j][i])), R = obs.filter(i => !sp.test(val[j][i]));
            const missLeft = L.length >= R.length;
            Object.assign(node, { variable: j, name: vars[j].name, type: vars[j].type, kind: sp.kind, threshold: sp.threshold, level: sp.level, missLeft, gain: sp.gain });
            node.left = grow(missLeft ? L.concat(miss) : L, depth + 1);
            node.right = grow(missLeft ? R : R.concat(miss), depth + 1);
            // Merge siblings that are leaves with the same prediction
            if (merge && !node.left.left && !node.right.left && node.left.predict === node.right.predict) return { n: node.n, counts: c, predict: node.left.predict };
            return node;
        }
        const all = rows.map((_, i) => i);
        return { root: grow(all, 0), vars, k };
    }
    function goesLeft(node, x, vars) {
        const a = codeValue(x[node.variable], vars[node.variable]);
        if (a === null) return node.missLeft;
        return node.kind === 'is' ? a === node.level : a <= node.threshold;
    }
    function predictTree(tree, x) { let node = tree.root; while (node.left) node = goesLeft(node, x, tree.vars) ? node.left : node.right; return node.predict; }
    /** Path of the new patient through the tree: [{ node, left }] */
    function treePath(tree, x) { const out = []; let node = tree.root; while (node.left) { const l = goesLeft(node, x, tree.vars); out.push({ node, left: l }); node = l ? node.left : node.right; } return { steps: out, leaf: node }; }
    /** Rules: one per leaf, as a list of conditions { node, left } plus the leaf */
    function treeRules(tree) {
        const out = [];
        (function walk(node, conds) { if (!node.left) { out.push({ conds, leaf: node }); return; } walk(node.left, conds.concat([{ node, left: true }])); walk(node.right, conds.concat([{ node, left: false }])); })(tree.root, []);
        return out;
    }
    function cohenKappa(a, b) {
        const n = a.length, k = Math.max(...a, ...b) + 1, pa = new Array(k).fill(0), pb = new Array(k).fill(0);
        let agree = 0; for (let i = 0; i < n; i++) { if (a[i] === b[i]) agree++; pa[a[i]]++; pb[b[i]]++; }
        const po = agree / n, pe = pa.reduce((s, x, c) => s + (x / n) * (pb[c] / n), 0);
        return pe < 1 ? (po - pe) / (1 - pe) : 1;
    }

    /** Balanced Gower weights without the n x n matrix: the mean contribution of each variable over all pairs, from
     *  sorted values for numbers and grades (sum over pairs of |a - b| = sum of x(k) (2k - n - 1)) and from the category
     *  shares for categories (chance two patients differ = 1 - sum of squared shares); equals gowerMatrix's weights */
    function gowerBalancedWeights(rows, vars) {
        return vars.map((v, j) => {
            const val = rows.map(r => codeValue(r[j], v)).filter(x => x !== null), m = val.length;
            if (m < 2) return 0;
            const pairs = (m * (m - 1)) / 2;
            if (v.type === 'nominal') { const cnt = new Map(); val.forEach(x => cnt.set(x, (cnt.get(x) || 0) + 1)); let same = 0; cnt.forEach(c => { same += (c * (c - 1)) / 2; }); const diff = (pairs - same) / pairs; return diff > 0 ? 1 / diff : 0; }
            const s = val.slice().sort((a, b) => a - b), range = s[m - 1] - s[0];
            if (!(range > 0)) return 0;
            let tot = 0; s.forEach((x, k) => { tot += x * (2 * k - m + 1); });
            const mean = tot / range / pairs;
            return mean > 0 ? 1 / mean : 0;
        });
    }
    /** Gower dissimilarity of two patients (rows of raw values), with given weights and the data's ranges */
    function gowerPairFn(rows, vars, weights) {
        const val = vars.map((v, j) => rows.map(r => codeValue(r[j], v)));
        const range = vars.map((v, j) => { if (v.type === 'nominal') return 1; let lo = Infinity, hi = -Infinity; val[j].forEach(x => { if (x !== null) { if (x < lo) lo = x; if (x > hi) hi = x; } }); return hi > lo ? hi - lo : 0; });
        const p = vars.length;
        return (a, b) => { let s = 0, w = 0; for (let j = 0; j < p; j++) { const x = val[j][a], y = val[j][b], wj = weights ? weights[j] : 1; if (x === null || y === null || !(wj > 0)) continue; w += wj; if (vars[j].type === 'nominal') s += x === y ? 0 : wj; else if (range[j] > 0) s += (wj * Math.abs(x - y)) / range[j]; } return w ? s / w : 1; };
    }

    /** Benjamini-Hochberg adjusted p-values (q-values, false discovery rate), in the input order; nulls kept */
    function benjaminiHochberg(ps) {
        const idx = ps.map((p, i) => [p, i]).filter(([p]) => p !== null && Number.isFinite(p)).sort((a, b) => a[0] - b[0]), m = idx.length, q = ps.map(() => null);
        let run = 1;
        for (let r = m - 1; r >= 0; r--) { run = Math.min(run, (idx[r][0] * m) / (r + 1)); q[idx[r][1]] = Math.min(1, run); }
        return q;
    }
    /** Cramer's V from a chi-square statistic (n observations, r categories, c groups) */
    const cramersV = (x2, n, r, c) => (n > 0 && Math.min(r, c) > 1 ? Math.sqrt(x2 / (n * (Math.min(r, c) - 1))) : null);
    /** Epsilon squared for the Kruskal-Wallis H (Tomczak & Tomczak 2014): H / (n - 1) */
    const epsilonSquared = (H, n) => (n > 1 ? H / (n - 1) : null);

    const ClusterEngine = { createPRNG, standardize, euclideanMatrix, gowerMatrix, lloyd, kmeansPlusPlus, kmeans, pam, hierarchical, cutTree, gmm, gmmFromParams, mStep,
        silhouette, calinskiHarabasz, daviesBouldin, withinSS, adjustedRand, symmetricEigen, pca, gapStatistic, bootstrapStability, nullClusterTest, copulaNullTest, normInv, normCdf, chiSquareTest, kruskalWallis, gammaQ, relabelBySize,
        benjaminiHochberg, cramersV, epsilonSquared, codeValue, gowerDistances, gowerBalancedWeights, gowerPairFn, lca, lcaPosterior, classificationTree, predictTree, treePath, treeRules, cohenKappa };
    root.ClusterEngine = ClusterEngine;
})(typeof window !== 'undefined' ? window : globalThis);
