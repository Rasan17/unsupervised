// anomaly-engine.js: unusual patients and likely data errors. Robust Mahalanobis distance from the minimum
// covariance determinant (as scikit-learn's MinCovDet), isolation forest (Liu, Ting & Zhou 2008, as scikit-learn),
// local outlier factor (Breunig et al. 2000, as scikit-learn, from any dissimilarity), robust z-scores, and checks
// for data-entry errors. Plain script (window.AnomalyEngine); uses ClusterEngine for gammaQ and random numbers.
(function (root) {
    'use strict';
    const CE = () => root.ClusterEngine;

    // ------------------------------------------------------------------------------------------- chi-square
    const chi2Sf = (x, df) => (x <= 0 ? 1 : CE().gammaQ(df / 2, x / 2));
    const chi2Cdf = (x, df) => 1 - chi2Sf(x, df);
    function chi2Ppf(p, df) {
        let lo = 0, hi = Math.max(10, df * 10);
        while (chi2Cdf(hi, df) < p) hi *= 2;
        for (let it = 0; it < 200; it++) { const mid = (lo + hi) / 2; if (chi2Cdf(mid, df) < p) lo = mid; else hi = mid; if (hi - lo < 1e-13 * Math.max(1, hi)) break; }
        return (lo + hi) / 2;
    }

    // ------------------------------------------------------------------------------------------- small linear algebra
    const mean = (X, idx) => { const p = X[0].length, m = new Array(p).fill(0); idx.forEach(i => { for (let j = 0; j < p; j++) m[j] += X[i][j]; }); return m.map(v => v / idx.length); };
    function cov(X, idx, m) {   // maximum-likelihood covariance (divisor n), as scikit-learn's empirical_covariance
        const p = m.length, C = Array.from({ length: p }, () => new Array(p).fill(0));
        idx.forEach(i => { for (let a = 0; a < p; a++) { const da = X[i][a] - m[a]; for (let b = a; b < p; b++) C[a][b] += da * (X[i][b] - m[b]); } });
        for (let a = 0; a < p; a++) for (let b = a; b < p; b++) { C[a][b] /= idx.length; C[b][a] = C[a][b]; }
        return C;
    }
    // Inverse and log-determinant through the symmetric eigendecomposition (pseudo-inverse for tiny eigenvalues)
    function invLogdet(C) {
        const { values, vectors } = CE().symmetricEigen(C), p = C.length, tol = 1e-12 * Math.max(...values.map(Math.abs));
        const inv = Array.from({ length: p }, () => new Array(p).fill(0)); let logdet = 0;
        values.forEach((lam, q) => { if (lam <= tol) { logdet = -Infinity; return; } logdet += Math.log(lam); const v = vectors[q]; for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) inv[a][b] += (v[a] * v[b]) / lam; });
        return { inv, logdet };
    }
    function mahal2(X, m, inv) { const p = m.length; return X.map(x => { let s = 0; for (let a = 0; a < p; a++) { let t = 0; for (let b = 0; b < p; b++) t += inv[a][b] * (x[b] - m[b]); s += (x[a] - m[a]) * t; } return s; }); }
    const smallest = (d, h) => d.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, h).map(x => x[1]);

    // ------------------------------------------------------------------------------------------- MCD
    function cStep(X, h, start, maxIter) {
        let idx = start, m = mean(X, idx), C = cov(X, idx, m), { inv, logdet } = invLogdet(C), prev = Infinity, it = maxIter;
        let best = { m, C, logdet, idx, inv };
        while (logdet < prev && it > 0 && isFinite(logdet)) {
            prev = logdet; best = { m, C, logdet, idx, inv };
            idx = smallest(mahal2(X, m, inv), h);
            m = mean(X, idx); C = cov(X, idx, m); ({ inv, logdet } = invLogdet(C));
            it--;
        }
        if (logdet <= prev || !isFinite(prev)) best = { m, C, logdet, idx, inv };
        return best;
    }
    /**
     * Minimum covariance determinant (FAST-MCD of Rousseeuw & Van Driessen 1999, as scikit-learn MinCovDet):
     * 100 random h-subsets (scikit-learn: 30) refined by two concentration steps, the 20 best (10) refined to
     * convergence, which finds the smallest determinant more reliably; raw estimate
     * with the consistency factor; reweighted with the patients below the 97.5% chi-square quantile.
     * X: complete numbers (patients x variables). Returns robust location, covariance, squared robust distances,
     * the raw support and the support after reweighting.
     */
    function mcd(X, { seed = 2024, nTrials = 100, nBest = 20 } = {}) {
        const n = X.length, p = X[0].length;
        if (n < p + 2) throw new Error('Too few patients for a robust covariance.');
        const h = Math.min(Math.ceil(0.5 * (n + p + 1)), n), rng = CE().createPRNG(seed);
        const perm = () => { const a = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a.slice(0, h); };
        let cands = Array.from({ length: nTrials }, () => cStep(X, h, perm(), 2)).sort((a, b) => a.logdet - b.logdet).slice(0, nBest);
        cands = cands.map(c => cStep(X, h, smallest(mahal2(X, c.m, c.inv), h), 30)).sort((a, b) => a.logdet - b.logdet);
        const raw = cands[0];
        const consistency = (q, alpha) => alpha / chi2Cdf(chi2Ppf(alpha, q), q + 2);
        let dist = mahal2(X, raw.m, raw.inv).map(d => d / consistency(p, h / n));
        const cut = chi2Ppf(0.975, p), keep = dist.map((d, i) => (d < cut ? i : -1)).filter(i => i >= 0);
        const m = mean(X, keep), C = cov(X, keep, m).map(r => r.map(v => v * consistency(p, 0.975)));
        dist = mahal2(X, m, invLogdet(C).inv);
        const support = new Array(n).fill(false); keep.forEach(i => { support[i] = true; });
        const rawSupport = new Array(n).fill(false); raw.idx.forEach(i => { rawSupport[i] = true; });
        return { location: m, covariance: C, dist, rawLocation: raw.m, rawSupport, support, h };
    }

    // ------------------------------------------------------------------------------------------- isolation forest
    const EULER = 0.5772156649015329;
    /** Average path length of an unsuccessful search in a binary tree of n points (scikit-learn _average_path_length) */
    function avgPath(n) { if (n <= 1) return 0; if (n === 2) return 1; return 2 * (Math.log(n - 1) + EULER) - (2 * (n - 1)) / n; }
    /**
     * Isolation forest: 100 trees, each grown on a random sample of min(256, n) patients without replacement to depth
     * ceil(log2(sample size)), splitting a random variable (among those that vary) at a random value. Anomaly score
     * 2^(-mean path length / c(sample size)): near 1 isolated quickly (unusual), about 0.5 or below ordinary.
     * X: complete numbers. Returns scores.
     */
    function isolationForest(X, { nTrees = 100, maxSamples = 256, seed = 2024 } = {}) {
        const n = X.length, p = X[0].length, psi = Math.min(maxSamples, n), maxDepth = Math.ceil(Math.log2(Math.max(2, psi))), rng = CE().createPRNG(seed);
        const trees = [];
        for (let t = 0; t < nTrees; t++) {
            const all = Array.from({ length: n }, (_, i) => i);
            for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [all[i], all[j]] = [all[j], all[i]]; }
            const grow = (idx, depth) => {
                if (depth >= maxDepth || idx.length <= 1) return { size: idx.length };
                const varying = [];
                for (let j = 0; j < p; j++) { let lo = Infinity, hi = -Infinity; idx.forEach(i => { const v = X[i][j]; if (v < lo) lo = v; if (v > hi) hi = v; }); if (hi > lo) varying.push([j, lo, hi]); }
                if (!varying.length) return { size: idx.length };
                const [j, lo, hi] = varying[Math.floor(rng() * varying.length)], thr = lo + rng() * (hi - lo);
                const L = idx.filter(i => X[i][j] < thr), R = idx.filter(i => X[i][j] >= thr);
                return { j, thr, left: grow(L, depth + 1), right: grow(R, depth + 1) };
            };
            trees.push(grow(all.slice(0, psi), 0));
        }
        const cn = avgPath(psi);
        return X.map(x => {
            let total = 0;
            trees.forEach(tr => { let node = tr, d = 0; while (node.left) { node = x[node.j] < node.thr ? node.left : node.right; d++; } total += d + avgPath(node.size); });
            return Math.pow(2, -(total / nTrees) / cn);
        });
    }

    // ------------------------------------------------------------------------------------------- local outlier factor
    /** Local outlier factor from a dissimilarity matrix { n, D } with k neighbours (scikit-learn LocalOutlierFactor):
     *  about 1 = as dense as its neighbours; well above 1 = in a sparser region than its neighbours. */
    function lof({ n, D }, k = 20) {
        k = Math.min(k, n - 1);
        const nb = [], kd = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            const idx = []; for (let j = 0; j < n; j++) if (j !== i) idx.push(j);
            idx.sort((a, b) => D[i * n + a] - D[i * n + b] || a - b);
            nb.push(idx.slice(0, k)); kd[i] = D[i * n + idx[k - 1]];
        }
        const lrd = new Float64Array(n);
        for (let i = 0; i < n; i++) { let s = 0; nb[i].forEach(j => { s += Math.max(kd[j], D[i * n + j]); }); lrd[i] = 1 / (s / k + 1e-10); }
        return Array.from({ length: n }, (_, i) => nb[i].reduce((a, j) => a + lrd[j], 0) / k / lrd[i]);
    }

    /** Local outlier factor for large datasets: dissimilarities computed one record at a time from dist(i, j), keeping
     *  only each record's k nearest neighbours (memory n x k instead of n x n); identical results to lof(). */
    function* lofStreamSteps(n, dist, k = 20) {
        k = Math.min(k, n - 1);
        const nb = new Array(n), nd = new Array(n), kd = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            const bi = [], bd = [];
            for (let j = 0; j < n; j++) {
                if (j === i) continue;
                const d = dist(i, j);
                if (bi.length < k || d < bd[k - 1] || (d === bd[k - 1] && j < bi[k - 1])) {
                    let q = bi.length < k ? bi.length : k - 1;
                    while (q > 0 && (bd[q - 1] > d || (bd[q - 1] === d && bi[q - 1] > j))) { if (q < k) { bd[q] = bd[q - 1]; bi[q] = bi[q - 1]; } q--; }
                    bd[q] = d; bi[q] = j; if (bd.length > k) { bd.length = k; bi.length = k; }
                }
            }
            nb[i] = bi; nd[i] = bd; kd[i] = bd[k - 1];
            yield i + 1;
        }
        const lrd = new Float64Array(n);
        for (let i = 0; i < n; i++) { let s = 0; for (let q = 0; q < k; q++) s += Math.max(kd[nb[i][q]], nd[i][q]); lrd[i] = 1 / (s / k + 1e-10); }
        return Array.from({ length: n }, (_, i) => nb[i].reduce((a, j) => a + lrd[j], 0) / k / lrd[i]);
    }

    function lofStream(n, dist, k = 20) { const g = lofStreamSteps(n, dist, k); for (;;) { const r = g.next(); if (r.done) return r.value; } }
    /** As lofStream, returning a promise: hands control back to the page about every 100 ms (onProgress(done, n)) */
    async function lofStreamAsync(n, dist, k = 20, onProgress = null) {
        const g = lofStreamSteps(n, dist, k); let last = Date.now();
        for (;;) { const r = g.next(); if (r.done) return r.value; if (Date.now() - last > 100) { if (onProgress) onProgress(r.value, n); await new Promise(res => setTimeout(res, 0)); last = Date.now(); } }
    }

    // ------------------------------------------------------------------------------------------- robust z and data checks
    function median(xs) { const s = xs.slice().sort((a, b) => a - b), h = (s.length - 1) / 2; return (s[Math.floor(h)] + s[Math.ceil(h)]) / 2; }
    /** Robust z-scores: (x - median) / (1.4826 x median absolute deviation); falls back to the interquartile range
     *  (/ 1.349) and then the standard deviation when the MAD is 0 */
    function robustZ(xs) {
        const obs = xs.filter(Number.isFinite);
        if (obs.length < 3) return { med: NaN, scale: NaN, z: xs.map(() => NaN) };
        const med = median(obs);
        let scale = (1 / 0.6744897501960817) * median(obs.map(v => Math.abs(v - med)));
        if (!(scale > 0)) { const s = obs.slice().sort((a, b) => a - b), q = f => { const h = (s.length - 1) * f, l = Math.floor(h); return s[l] + (s[Math.ceil(h)] - s[l]) * (h - l); }; scale = (q(0.75) - q(0.25)) / 1.349; }
        if (!(scale > 0)) { const m = obs.reduce((a, b) => a + b, 0) / obs.length; scale = Math.sqrt(obs.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, obs.length - 1)); }
        return { med, scale, z: xs.map(v => (Number.isFinite(v) && scale > 0 ? (v - med) / scale : NaN)) };
    }
    /** True for a clearly right-skewed measurement with no negative values (Kelly's decile skewness above 0.2): such
     *  variables are judged on the log(1 + x) scale, so a long natural tail is not mistaken for errors or outliers */
    function skewedNonNegative(xs) {
        const v = xs.filter(Number.isFinite);
        if (v.length < 20 || !v.every(x => x >= 0)) return false;
        const t = v.slice().sort((a, b) => a - b), q = f => { const h = (t.length - 1) * f, l = Math.floor(h); return t[l] + (t[Math.ceil(h)] - t[l]) * (h - l); };
        const lo = q(0.1), md = q(0.5), hi = q(0.9);
        return hi > lo && (hi + lo - 2 * md) / (hi - lo) > 0.2;
    }
    // Typical codes used for "not recorded" in clinical spreadsheets
    const CODE_VALUES = [-9999, -999, -99, -9, 99, 999, 9999, 88, 888, 8888, 77, 777, 7777];
    /** Possible missing-value codes in a column of numbers: a typical code (e.g. -99, 999) recorded for at least 3
     *  patients and lying outside the range of the other values. Returns [{ value, count }]. */
    function missingCodes(xs) {
        const v = xs.filter(Number.isFinite);
        return CODE_VALUES.map(code => {
            const count = v.filter(x => x === code).length;
            if (count < 3) return null;
            const others = v.filter(x => x !== code);
            if (others.length < 10) return null;
            const lo = Math.min(...others), hi = Math.max(...others);
            return code < lo || code > hi ? { value: code, count } : null;
        }).filter(Boolean);
    }
    const missing = v => v === null || v === undefined || String(v).trim() === '' || /^(na|n\/a|nan|null|missing|\.)$/i.test(String(v).trim());
    /**
     * Checks for likely data-entry errors, column by column. columns: [{ name, kind: 'number' | 'category' | 'id' }];
     * rows: objects keyed by column name. Returns [{ row, column, value, kind, message }] (row: 0-based index).
     */
    function dataChecks(rows, columns, { zLimit = 5 } = {}) {
        const out = [];
        columns.forEach(({ name, kind }) => {
            const raw = rows.map(r => r[name]);
            if (kind === 'number') {
                const parsed = raw.map(v => (missing(v) ? null : Number(String(v).trim())));
                raw.forEach((v, i) => { if (!missing(v) && !Number.isFinite(parsed[i])) out.push({ row: i, column: name, value: String(v), kind: 'text', message: /^-?\d+,\d+$/.test(String(v).trim()) ? 'a decimal comma instead of a decimal point' : 'text in a column of numbers' }); });
                const codes = missingCodes(parsed.map(v => (Number.isFinite(v) ? v : NaN))), codeSet = new Set(codes.map(c => c.value));
                parsed.forEach((v, i) => { if (codeSet.has(v)) out.push({ row: i, column: name, value: String(raw[i]), kind: 'code', message: `possibly a code for "not recorded" (${v} is used for ${codes.find(c => c.value === v).count} patients, outside the range of the other values): if so, leave these cells empty` }); });
                const xs = parsed.map(v => (Number.isFinite(v) && !codeSet.has(v) ? v : NaN));
                // Clearly right-skewed positive measurements (e.g. CRP) are judged on the log scale, so a genuine long
                // tail is not mistaken for errors: right-skewed (Kelly's decile skewness above 0.2) with no negative values, judged on
                // the log(1 + x) scale (which also handles zeros, as in counts and waiting times)
                const pos = xs.filter(Number.isFinite);
                const skewed = skewedNonNegative(pos);
                const scaleXs = skewed ? xs.map(v => (Number.isFinite(v) && v >= 0 ? Math.log1p(v) : NaN)) : xs;
                const { scale, z } = robustZ(scaleXs), med = robustZ(xs).med;
                const sorted = scaleXs.filter(Number.isFinite).sort((a, b) => a - b);
                // Gap to the nearest other recorded value (an isolated value is suspicious even at a moderate z)
                const nearest = v => sorted.reduce((b, w) => (w !== v && Math.abs(w - v) < Math.abs(b - v) ? w : b), Infinity);
                const gapOf = v => { const k = sorted.indexOf(v), lo = k > 0 ? v - sorted[k - 1] : Infinity, hi = k < sorted.length - 1 && sorted[k + 1] !== v ? sorted[k + 1] - v : Infinity; const lastSame = sorted.lastIndexOf(v); const hi2 = lastSame < sorted.length - 1 ? sorted[lastSame + 1] - v : Infinity; return lastSame > k ? 0 : Math.min(lo, Math.min(hi, hi2)); };
                const nonNeg = xs.filter(Number.isFinite), negCount = nonNeg.filter(v => v < 0).length;
                xs.forEach((v, i) => {
                    if (!Number.isFinite(v)) return;
                    if (v < 0 && negCount <= Math.max(1, 0.01 * nonNeg.length)) { out.push({ row: i, column: name, value: String(raw[i]), kind: 'negative', message: 'negative, while (almost) all other values are positive' }); return; }
                    const isolated = Math.abs(z[i]) >= 3.5 && gapOf(scaleXs[i]) >= 1.5 * scale;
                    if (!(Math.abs(z[i]) >= zLimit) && !isolated) return;
                    const ratio = med !== 0 ? v / med : NaN, l10 = Math.log10(Math.abs(ratio)), r10 = Math.round(l10);
                    const decimal = Number.isFinite(l10) && r10 !== 0 && Math.abs(l10 - r10) < 0.15 && ratio > 0;
                    out.push({ row: i, column: name, value: String(raw[i]), kind: decimal ? 'decimal' : 'extreme', z: z[i],
                        message: decimal ? `about ${r10 > 0 ? Math.pow(10, r10) + ' times' : '1/' + Math.pow(10, -r10) + ' of'} the typical value (${fmt(med)}): possibly a decimal-point or unit error` : Math.abs(z[i]) >= zLimit ? `far from the other values${skewed ? ', even allowing for the skewed distribution' : ''} (typical ${fmt(med)}, robust z ${z[i].toFixed(1)}${skewed ? ' on the log scale' : ''})` : `isolated from the other values (nearest other value ${fmt(skewed ? Math.expm1(nearest(scaleXs[i])) : nearest(v))}; typical ${fmt(med)})` });
                });
            } else if (kind === 'category') {
                const vals = raw.filter(v => !missing(v)).map(v => String(v));
                const counts = new Map(); vals.forEach(v => counts.set(v, (counts.get(v) || 0) + 1));
                const norm = v => v.trim().toLowerCase().replace(/\s+/g, ' ');
                const byNorm = new Map(); counts.forEach((c, v) => { const k = norm(v); if (!byNorm.has(k)) byNorm.set(k, []); byNorm.get(k).push([v, c]); });
                byNorm.forEach(list => {
                    if (list.length < 2) return;
                    const main = list.slice().sort((a, b) => b[1] - a[1])[0][0];
                    list.forEach(([v]) => { if (v === main) return; raw.forEach((x, i) => { if (!missing(x) && String(x) === v) out.push({ row: i, column: name, value: v, kind: 'variant', message: `looks like a spelling variant of "${main}"` }); }); });
                });
                if (counts.size <= 12 && vals.length >= 30) counts.forEach((c, v) => { if (c === 1 && byNorm.get(norm(v)).length === 1) raw.forEach((x, i) => { if (!missing(x) && String(x) === v) out.push({ row: i, column: name, value: v, kind: 'rare', message: 'a category recorded for only this patient: check the spelling or coding' }); }); });
            }
        });
        // Duplicate rows (all columns except identifiers equal) and duplicate identifiers
        const idCols = columns.filter(c => c.kind === 'id').map(c => c.name), other = columns.filter(c => c.kind !== 'id').map(c => c.name);
        // Identical rows are reported only when so many variables match that coincidence is implausible: expected
        // number of chance pairs = pairs x product over variables of the chance two patients share a value
        const nR = rows.length;
        let pMatch = 1;
        other.forEach(c => { const cnt = new Map(); rows.forEach(r => { const v = String(r[c] ?? '').trim(); cnt.set(v, (cnt.get(v) || 0) + 1); }); let s2 = 0; cnt.forEach(k => { s2 += (k / nR) * (k / nR); }); pMatch *= s2; });
        const chancePairs = (nR * (nR - 1)) / 2 * pMatch;
        const seen = new Map();
        if (chancePairs < 0.05) rows.forEach((r, i) => { const key = other.map(c => String(r[c] ?? '').trim()).join('\u0001'); if (seen.has(key)) out.push({ row: i, column: '(whole row)', value: '', kind: 'duplicate', message: `identical to row ${seen.get(key) + 1} in every variable` }); else seen.set(key, i); });
        idCols.forEach(c => { const first = new Map(); rows.forEach((r, i) => { const v = String(r[c] ?? '').trim(); if (!v) return; if (first.has(v)) out.push({ row: i, column: c, value: v, kind: 'duplicate-id', message: `same identifier as row ${first.get(v) + 1} (fine if a patient has several records)` }); else first.set(v, i); }); });
        return out;
    }
    function fmt(v) { return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : Number(v.toPrecision(3)).toString(); }

    root.AnomalyEngine = { missingCodes, skewedNonNegative, chi2Sf, chi2Cdf, chi2Ppf, mcd, isolationForest, avgPath, lof, lofStream, lofStreamAsync, robustZ, dataChecks, median };
})(typeof window !== 'undefined' ? window : globalThis);
