// Statistical behaviour of the engine on simulated data (the counterpart of the predictor's simulation checks):
// does each procedure find structure when it exists, and decline to when it does not?
// Usage: node tests/behaviour_check.js
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');
global.window = globalThis;
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'data-engine.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'cluster-engine.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'embed-engine.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'anomaly-engine.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'nn-engine.js'), 'utf8'));
const DE = globalThis.DataEngine, CE = globalThis.ClusterEngine, EE = globalThis.EmbedEngine, AE = globalThis.AnomalyEngine, NN = globalThis.NNEngine;
let passed = 0, failed = 0;
function check(name, fn) {
    const t0 = Date.now();
    try { const msg = fn(); passed++; console.log(`PASS  ${name}  (${msg}; ${((Date.now() - t0) / 1000).toFixed(1)} s)`); }
    catch (e) { failed++; console.log(`FAIL  ${name}: ${e.message}`); }
}
const assert = (c, m) => { if (!c) throw new Error(m); };
function sim(seed, n, k, sep, p = 4) {
    const rng = CE.createPRNG(seed), g = () => { const u = rng() || 1e-12, v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const centres = Array.from({ length: k }, (_, c) => Array.from({ length: p }, (_, j) => (j === c % p ? sep : 0) + (j === (c + 1) % p && c >= p ? sep : 0)));
    const X = [], truth = [];
    for (let i = 0; i < n; i++) { const c = i % k; truth.push(c); X.push(centres[c].map(m => m + g())); }
    return { X, truth };
}

check('Gap statistic finds 3 well-separated clusters', () => {
    const { X } = sim(1, 150, 3, 5);
    const r = CE.gapStatistic(X, 6, { B: 20, seed: 2 });
    assert(r.kBest === 3, `chose k = ${r.kBest}`);
    return 'k = 3';
});
check('Gap statistic chooses 1 cluster for a single Gaussian cloud', () => {
    const { X } = sim(3, 150, 1, 0);
    const r = CE.gapStatistic(X, 6, { B: 20, seed: 4 });
    assert(r.kBest === 1, `chose k = ${r.kBest}`);
    return 'k = 1';
});
check('"No real clusters" test is calibrated: about 5% false positives and uniform p-values without clusters (150 datasets, incl. correlated)', () => {
    let rej = 0, below50 = 0; const sims = 150;
    for (let s = 0; s < sims; s++) {
        const { X } = sim(100 + s, 100, 1, 0, 4);
        const Xc = s % 2 ? X.map(r => r.map((v, j) => v * (j === 0 ? 3 : 1) + (j ? 0.6 * r[0] : 0))) : X; // half with correlated variables
        const pv = CE.nullClusterTest(Xc, 2, { B: 99, seed: s, nInit: 3 }).pValue;
        if (pv <= 0.05) rej++; if (pv <= 0.5) below50++;
    }
    assert(rej / sims >= 0.01 && rej / sims <= 0.10, `false-positive rate ${(100 * rej / sims).toFixed(1)}%`);
    assert(below50 / sims >= 0.38 && below50 / sims <= 0.62, `${(100 * below50 / sims).toFixed(0)}% of p-values below 0.5`);
    return `false positives ${(100 * rej / sims).toFixed(1)}%, ${(100 * below50 / sims).toFixed(0)}% of p-values below 0.5`;
});
check('"No real clusters" test detects 2 moderately separated clusters (20 datasets)', () => {
    let rej = 0; const sims = 20;
    for (let s = 0; s < sims; s++) { const { X } = sim(300 + s, 80, 2, 3, 3); if (CE.nullClusterTest(X, 2, { B: 39, seed: s, nInit: 3 }).pValue <= 0.05) rej++; }
    assert(rej / sims >= 0.8, `detected ${rej} of ${sims}`);
    return `${rej} of ${sims} detected`;
});
check('Bootstrap stability: real clusters are stable, clusters forced on noise are not', () => {
    const real = sim(5, 120, 3, 5), noise = sim(6, 120, 1, 0);
    const fn = X => idx => CE.kmeans(idx.map(i => X[i]), 3, { nInit: 3, seed: idx.length }).labels;
    const s1 = CE.bootstrapStability(120, CE.kmeans(real.X, 3, { seed: 1 }).labels, fn(real.X), { B: 40, seed: 7 });
    const s2 = CE.bootstrapStability(120, CE.kmeans(noise.X, 3, { seed: 1 }).labels, fn(noise.X), { B: 40, seed: 7 });
    const minReal = Math.min(...s1.perCluster.map(c => c.meanJaccard)), meanNoise = s2.perCluster.reduce((a, c) => a + c.meanJaccard, 0) / 3;
    assert(minReal >= 0.85, `real clusters: lowest mean Jaccard ${minReal.toFixed(2)}`);
    assert(meanNoise < minReal - 0.2, `noise clusters: mean Jaccard ${meanNoise.toFixed(2)}`);
    return `real clusters >= ${minReal.toFixed(2)}, clusters forced on noise ${meanNoise.toFixed(2)}`;
});
check('All four methods recover well-separated clusters (adjusted Rand index >= 0.95)', () => {
    const { X, truth } = sim(8, 150, 3, 6);
    const E = CE.euclideanMatrix(X);
    const out = { kmeans: CE.kmeans(X, 3).labels, pam: CE.pam(E, 3).labels, ward: CE.cutTree(CE.hierarchical(E, 'ward'), X.length, 3), gmm: CE.gmm(X, 3).labels };
    const aris = Object.entries(out).map(([m, l]) => [m, CE.adjustedRand(truth, l)]);
    aris.forEach(([m, a]) => assert(a >= 0.95, `${m}: ARI ${a.toFixed(3)}`));
    return aris.map(([m, a]) => `${m} ${a.toFixed(3)}`).join(', ');
});
check('BIC of Gaussian mixtures is lowest at the true number of clusters', () => {
    const { X } = sim(9, 200, 3, 5, 3);
    const bics = [1, 2, 3, 4, 5].map(k => CE.gmm(X, k, { covType: 'diag', nInit: 3 }).bic);
    const best = bics.indexOf(Math.min(...bics)) + 1;
    assert(best === 3, `lowest BIC at k = ${best}`);
    return 'k = 3';
});
check('Gower dissimilarity with whole groups of missing values is refused with a clear message', () => {
    try { CE.gowerMatrix([[1, null], [null, 'a']], [{ type: 'continuous' }, { type: 'nominal' }]); } catch (e) { assert(/share no recorded variable/.test(e.message), e.message); return 'refused'; }
    throw new Error('not refused');
});
check('Results are reproducible: same seed, same clusters', () => {
    const { X } = sim(10, 100, 3, 3);
    const a = CE.kmeans(X, 3, { seed: 42 }).labels, b = CE.kmeans(X, 3, { seed: 42 }).labels;
    assert(a.join() === b.join(), 'differ');
    return 'identical';
});
check('Mixed-data test (copula reference) is valid: at most 5% false positives for one population with yes/no, graded and skewed variables (100 datasets)', () => {
    const mixed = (seed, n) => { const rng = CE.createPRNG(seed), g = () => { const u = rng() || 1e-12, v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
        return Array.from({ length: n }, () => { const z = g(), l = [z + 0.5 * g(), 0.6 * z + g(), -0.5 * z + g(), g(), 0.4 * z + g(), g()];
            return [l[0], Math.exp(0.6 * l[1]), l[2] > 0.3 ? 1 : 0, l[3] > -0.2 ? 1 : 0, Math.min(3, Math.max(0, Math.round(l[4] + 1.5))), Math.min(4, Math.max(0, Math.round(l[5] + 2)))]; }); };
    let rej = 0; const sims = 100;
    for (let s = 0; s < sims; s++) if (CE.copulaNullTest(mixed(700 + s, 150), 3, { B: 49, seed: s, nInit: 2 }).pValue <= 0.05) rej++;
    assert(rej / sims <= 0.06, `false-positive rate ${(100 * rej / sims).toFixed(1)}%`);
    return `false positives ${(100 * rej / sims).toFixed(1)}% (conservative)`;
});
check('Demo cohorts: sub-types detected where they exist (copula reference), and not where there are none', () => {
    const DEp = require('path'), D = DEp.join(__dirname, '..');
    vm.runInThisContext(fs.readFileSync(DEp.join(D, 'data-engine.js'), 'utf8'));
    const vars = [['Age', 'c'], ['Sex', 'b'], ['Hypertension', 'b'], ['White_Matter_Fazekas', 'o'], ['MMSE', 'c'], ['Gait_Score', 'o'], ['Evans_Index', 'c'], ['Callosal_Angle_Deg', 'c'], ['DESH_Sign', 'b'], ['Rout_mmHg_min_ml', 'c'], ['Tap_Test_Gait_Gain_Pct', 'c']];
    const ps = ['demo_inph_phenotypes.csv', 'demo_no_subtypes.csv'].map(f => {
        const p = globalThis.DataEngine.parseCSV(fs.readFileSync(DEp.join(D, f), 'utf8'));
        const X = p.rows.filter(r => vars.every(([v]) => !globalThis.DataEngine.isMissing(r[v]))).map(r => vars.map(([v, t]) => (t === 'b' ? (['Male', 'Yes', 'Present'].includes(r[v]) ? 1 : 0) : Number(r[v]))));
        return CE.copulaNullTest(X, 3, { B: 99, seed: 1 }).pValue;
    });
    assert(ps[0] <= 0.05 && ps[1] > 0.2, `p = ${ps.join(' / ')}`);
    return `p = ${ps[0]} (sub-types) and ${ps[1]} (none)`;
});
// Latent class analysis on simulated mixed data (numbers, yes/no, a category and a grade) with 15% missing values
function lcaData(seed, n = 450) {
    const r = CE.createPRNG(seed), gz = () => { const u = r() || 1e-12, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const rows = [], truth = [];
    for (let i = 0; i < n; i++) {
        const c = i % 3; truth.push(c);
        rows.push([[50, 62, 56][c] + 5 * gz(), [10, 12, 16][c] + 2 * gz(), r() < [0.85, 0.2, 0.5][c] ? 'Yes' : 'No', ['A', 'B', 'C'][r() < 0.7 ? c : Math.floor(r() * 3)], String(Math.min(3, Math.max(0, Math.round([0.5, 1.5, 2.5][c] + 0.6 * gz()))))]);
    }
    const complete = rows.map(x => x.slice());
    for (let i = 0; i < n; i++) for (let j = 0; j < 5; j++) if (r() < 0.15) rows[i][j] = null;
    return { rows, complete, truth, vars: [{ type: 'continuous' }, { type: 'continuous' }, { type: 'nominal' }, { type: 'nominal' }, { type: 'ordinal', levels: ['0', '1', '2', '3'] }] };
}
check('Latent class analysis recovers simulated classes in mixed data; with 15% of values missing at random, class sizes and averages stay unbiased (5 datasets)', () => {
    const out = [1, 2, 3, 4, 5].map(sd => {
        const d = lcaData(100 + sd), mm = CE.lca(d.rows, d.vars, 3, { nInit: 10, seed: sd }), mc = CE.lca(d.complete, d.vars, 3, { nInit: 10, seed: sd });
        // Class averages of the first variable on its original scale, matched by sorting
        const avg = m => m.model.mu.map(r => m.model.mean[0] + m.model.sd[0] * r[0]).sort((x, y) => x - y);
        const dMean = Math.max(...avg(mm).map((v, c) => Math.abs(v - avg(mc)[c]))), dShare = Math.max(...mm.model.pi.map((v, c) => Math.abs(v - mc.model.pi[c])));
        return { ari: CE.adjustedRand(mm.labels, d.truth), dMean, dShare };
    });
    assert(out.every(o => o.ari >= 0.75 && o.dMean < 1.5 && o.dShare < 0.05), JSON.stringify(out));
    return `ARI ${out.map(o => o.ari.toFixed(2)).join(', ')}; largest difference from the complete-data fit: class average ${Math.max(...out.map(o => o.dMean)).toFixed(2)} (SD 5), class share ${Math.max(...out.map(o => o.dShare)).toFixed(3)}`;
});
check('Latent class analysis: BIC is lowest at the true number of classes (5 datasets)', () => {
    const ks = [1, 2, 3, 4, 5].map(sd => { const d = lcaData(200 + sd); const b = [1, 2, 3, 4, 5].map(k => CE.lca(d.rows, d.vars, k, { nInit: 8, seed: sd }).bic); return b.indexOf(Math.min(...b)) + 1; });
    assert(ks.every(k => k === 3), `chosen ${ks.join(', ')}`);
    return `chosen: ${ks.join(', ')}`;
});
check('Latent class analysis: membership probabilities sum to 1, a new patient is classified like the fitted ones, same seed same result', () => {
    const d = lcaData(300), m = CE.lca(d.rows, d.vars, 3, { nInit: 6, seed: 9 }), m2 = CE.lca(d.rows, d.vars, 3, { nInit: 6, seed: 9 });
    const sums = m.posterior.every(p => Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    const same = d.rows.every((x, i) => { const q = CE.lcaPosterior(m.model, x); return q.every((v, c) => Math.abs(v - m.posterior[i][c]) < 1e-9); });
    assert(sums && same && CE.adjustedRand(m.labels, m2.labels) === 1, `sums ${sums}, new-patient ${same}`);
    return 'all hold';
});
check('Maps: t-SNE and UMAP keep separated clusters apart (k-means on the map recovers them, adjusted Rand >= 0.9) and are faithful (trustworthiness >= 0.9)', () => {
    const { X, truth } = sim(41, 300, 3, 4, 6), E = CE.euclideanMatrix(X);
    const T = EE.tsne(EE.squared(E)), U = EE.umap(E, { seed: 2 });
    const ari = Y => CE.adjustedRand(CE.kmeans(Y, 3, { seed: 1 }).labels, truth), r = [ari(T.Y), ari(U.Y)], t = [EE.trustworthiness(E, T.Y, 10), EE.trustworthiness(E, U.Y, 10)];
    assert(r.every(v => v >= 0.9) && t.every(v => v >= 0.9), `ARI ${r.join(', ')}, trust ${t.join(', ')}`);
    return `ARI t-SNE ${r[0].toFixed(2)}, UMAP ${r[1].toFixed(2)}; trustworthiness ${t[0].toFixed(3)}, ${t[1].toFixed(3)}`;
});
check('Maps are reproducible: same seed, same map', () => {
    const { X } = sim(42, 120, 2, 3), E = CE.euclideanMatrix(X);
    const a = EE.umap(E, { seed: 5 }).Y, b = EE.umap(E, { seed: 5 }).Y, c = EE.tsne(EE.squared(E)).Y, d = EE.tsne(EE.squared(E)).Y;
    assert(a.every((r, i) => r[0] === b[i][0] && r[1] === b[i][1]) && c.every((r, i) => r[0] === d[i][0] && r[1] === d[i][1]), 'maps differ');
    return 'identical';
});
// Unusual patients and data errors
function corrData(seed, n, nOut, shift, scattered = false) {
    const r = CE.createPRNG(seed), gz = () => { const u = r() || 1e-12, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const X = [], y = [];
    for (let i = 0; i < n; i++) { const a = gz(), b = gz(), c = gz(), d = gz(); X.push([a, 0.8 * a + 0.6 * b, 0.5 * b + 0.8 * c, 0.6 * c + 0.8 * d]); y.push(0); }
    // Outliers: a tight group placed `shift` standard deviations out along the direction in which the clean data vary
    // least (each single value unremarkable; a group like this inflates, and so masks, the classical distance)
    if (nOut) {
        const m = X[0].map((_, j) => X.reduce((a, r) => a + r[j], 0) / X.length), C = m.map((_, a) => m.map((__, b) => X.reduce((s, r) => s + (r[a] - m[a]) * (r[b] - m[b]), 0) / X.length));
        const e = CE.symmetricEigen(C), q = e.values.length - 1, v = e.vectors[q], sd = Math.sqrt(e.values[q]);
        for (let i = 0; i < nOut; i++) {
            // scattered: a typical patient moved off the pattern, in a random direction along the least-variance axis
            const base = scattered ? X[Math.floor(r() * n)] : m, sign = scattered && r() < 0.5 ? -1 : 1;
            X.push(base.map((bj, j) => bj + sign * shift * sd * v[j] + (scattered ? 0 : 0.1 * sd * gz()))); y.push(1);
        }
    }
    return { X, y };
}
const auc = (s, y) => { let num = 0, den = 0; y.forEach((yi, i) => { if (!yi) return; y.forEach((yj, j) => { if (yj) return; den++; num += s[i] > s[j] ? 1 : s[i] === s[j] ? 0.5 : 0; }); }); return num / den; };
check('Robust distance finds a group of outliers that masks the classical Mahalanobis distance (10% contamination)', () => {
    const { X, y } = corrData(51, 270, 30, 6), cut = AE.chi2Ppf(0.999, 4), R = AE.mcd(X, { seed: 1 });
    const m = X[0].map((_, j) => X.reduce((a, r) => a + r[j], 0) / X.length), C = X[0].map((_, a) => X[0].map((__, b) => X.reduce((s, r) => s + (r[a] - m[a]) * (r[b] - m[b]), 0) / X.length));
    const Ci = (() => { const e = CE.symmetricEigen(C); return C.map((_, a) => C.map((__, b) => e.values.reduce((s, l, q) => s + e.vectors[q][a] * e.vectors[q][b] / l, 0))); })();
    const classical = X.map(x => x.reduce((s, xa, a) => s + (xa - m[a]) * x.reduce((t, xb, b) => t + Ci[a][b] * (xb - m[b]), 0), 0));
    const found = R.dist.filter((d, i) => y[i] && d > cut).length, foundClassical = classical.filter((d, i) => y[i] && d > cut).length, falseR = R.dist.filter((d, i) => !y[i] && d > cut).length;
    assert(found >= 27 && foundClassical < found && falseR <= 3, `robust ${found}/30, classical ${foundClassical}/30, false ${falseR}`);
    return `robust distance ${found} of 30, classical ${foundClassical} of 30; false alarms ${falseR} of 270`;
});
check('Robust distance: about 0.1% false alarms at the 99.9% limit in clean correlated data (20 datasets x 300 patients)', () => {
    let fa = 0, tot = 0; const cut = AE.chi2Ppf(0.999, 4);
    for (let s = 0; s < 20; s++) { const { X } = corrData(600 + s, 300, 0, 0), R = AE.mcd(X, { seed: s }); fa += R.dist.filter(d => d > cut).length; tot += X.length; }
    assert(fa / tot <= 0.005, `${fa} of ${tot}`);
    return `${fa} of ${tot} (${(100 * fa / tot).toFixed(2)}%)`;
});
check('Isolation forest and local outlier factor rank scattered outliers that break the correlations high (AUC >= 0.85 and >= 0.95)', () => {
    const { X, y } = corrData(52, 285, 15, 6, true), a1 = auc(AE.isolationForest(X, { seed: 2 }), y), a2 = auc(AE.lof(CE.euclideanMatrix(X), 20), y);
    assert(a1 >= 0.85 && a2 >= 0.95, `iso ${a1}, lof ${a2}`);
    return `AUC isolation forest ${a1.toFixed(3)}, local outlier factor ${a2.toFixed(3)}`;
});
check('Data checks raise (almost) no alarms in clean data, including skewed measurements: at most 2 in 6000 patients (rare genuine extremes)', () => {
    let alarms = 0;
    for (let s = 0; s < 20; s++) {
        const r = CE.createPRNG(700 + s), gz = () => { const u = r() || 1e-12, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
        const rows = Array.from({ length: 300 }, (_, i) => ({ id: `P${i}`, age: String(Math.round(70 + 8 * gz())), crp: (3 * Math.exp(0.9 * gz())).toFixed(1), grade: String(Math.min(4, Math.max(0, Math.round(2 + gz())))), sex: r() < 0.5 ? 'F' : 'M' }));
        alarms += AE.dataChecks(rows, [{ name: 'id', kind: 'id' }, { name: 'age', kind: 'number' }, { name: 'crp', kind: 'number' }, { name: 'grade', kind: 'number' }, { name: 'sex', kind: 'category' }]).length;
    }
    assert(alarms <= 2, `${alarms} alarms`);
    return `${alarms} alarm(s) in 6000 patients`;
});
check('Data checks find planted entry errors (decimal point, extra digit, minus sign, decimal comma, spelling, duplicate)', () => {
    const r = CE.createPRNG(800), gz = () => { const u = r() || 1e-12, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const rows = Array.from({ length: 300 }, (_, i) => ({ id: `P${i}`, evans: (0.36 + 0.02 * gz()).toFixed(2), mmse: String(Math.round(24 + 3 * gz())), angle: String(Math.round(90 + 15 * gz())), rout: (14 + 4 * gz()).toFixed(1), tap: String(Math.round(20 + 12 * gz())), sex: r() < 0.5 ? 'Female' : 'Male' }));
    rows[3].evans = '36'; rows[9].mmse = '250'; rows[17].angle = '-95'; rows[25].evans = '0,35'; rows[40].sex = 'female'; rows[60] = { ...rows[59], id: 'P999' };
    const found = new Set(AE.dataChecks(rows, [{ name: 'id', kind: 'id' }, { name: 'evans', kind: 'number' }, { name: 'mmse', kind: 'number' }, { name: 'angle', kind: 'number' }, { name: 'rout', kind: 'number' }, { name: 'tap', kind: 'number' }, { name: 'sex', kind: 'category' }]).map(x => x.row));
    const want = [3, 9, 17, 25, 40, 60], hit = want.filter(i => found.has(i)), extra = [...found].filter(i => !want.includes(i));
    assert(hit.length === want.length && extra.length === 0, `found ${hit.join(',')}, extra ${extra.join(',')}`);
    return `all ${want.length} found, no false alarms`;
});
// Autoencoder
function curveData(seed, n) {   // patients along a curved one-dimensional path in 4 variables, plus noise
    const r = CE.createPRNG(seed), gz = () => { const u = r() || 1e-12, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    return Array.from({ length: n }, () => { const t = -1.5 + 3 * r(); return [t, t * t, Math.sin(2 * t), Math.cos(1.5 * t)].map(v => v + 0.05 * gz()); });
}
check('Autoencoder captures curved structure better than PCA with the same number of dimensions (1), on new patients', () => {
    const all = curveData(61, 500), st = CE.standardize(all.slice(0, 400)), sc = x => x.map((v, j) => (v - st.mean[j]) / st.sd[j]);
    const Xtr = st.Z, Xte = all.slice(400).map(sc), ones = A => A.map(r => r.map(() => 1));
    const T = NN.train(Xtr, ones(Xtr), { hidden: [16], latent: 1, seed: 2, maxEpochs: 800 });
    const err = NN.errors(T.net, Xte, ones(Xte)).reduce((a, e) => a + e.mean, 0) / Xte.length;
    const P1 = CE.pca(Xtr, 1), v = P1.components[0], mu = P1.mean;
    const pcaErr = Xte.reduce((a, x) => { const c = x.map((xj, j) => xj - mu[j]), t = c.reduce((s2, cj, j) => s2 + cj * v[j], 0); return a + c.reduce((s2, cj, j) => s2 + (cj - t * v[j]) ** 2, 0) / x.length; }, 0) / Xte.length;
    assert(err < 0.6 * pcaErr, `autoencoder ${err}, PCA ${pcaErr}`);
    return `reconstruction error on 100 new patients ${err.toFixed(3)} vs PCA ${pcaErr.toFixed(3)} (lower is better)`;
});
check('Autoencoder reconstruction error ranks scattered outliers high (AUC >= 0.9)', () => {
    const { X, y } = corrData(53, 285, 15, 6, true), Z = CE.standardize(X).Z;
    const T = NN.train(Z, Z.map(r => r.map(() => 1)), { hidden: [8], latent: 2, seed: 4 });
    const a = auc(NN.errors(T.net, Z, Z.map(r => r.map(() => 1))).map(e => e.mean), y);
    assert(a >= 0.9, `AUC ${a}`);
    return `AUC ${a.toFixed(3)}`;
});
check('Autoencoder: the training error counts recorded values only, and the same seed gives the same network', () => {
    const Z = CE.standardize(curveData(62, 150)).Z, M = Z.map((r, i) => r.map((_, j) => ((i + j) % 9 === 0 ? 0 : 1))), Zm = Z.map((r, i) => r.map((v, j) => (M[i][j] ? v : 0)));
    const a = NN.train(Zm, M, { seed: 7, maxEpochs: 60 }), b = NN.train(Zm, M, { seed: 7, maxEpochs: 60 });
    const idx = Zm.map((_, i) => i), loss = NN.lossGrad(a.net, Zm, M, idx, 0).loss;
    const manual = Zm.reduce((s2, x, i) => { const y = NN.reconstruct(a.net, x); return s2 + y.reduce((t, v, j) => t + (M[i][j] ? (v - x[j]) ** 2 : 0), 0); }, 0) / (Zm.length * Zm[0].length);
    const same = JSON.stringify(a.net.layers) === JSON.stringify(b.net.layers);
    assert(same && Math.abs(loss - manual) < 1e-12, `same ${same}, loss ${loss} vs ${manual}`);
    return 'loss equals the error over recorded values; identical networks from the same seed';
});
check('Large datasets: fast balanced Gower weights, pairwise dissimilarity and record-by-record local outlier factor equal the full-matrix versions', () => {
    const r = CE.createPRNG(5), rows = Array.from({ length: 300 }, () => [r() < 0.1 ? null : Math.round(50 + 20 * r()), ['a', 'b', 'c'][Math.floor(r() * 3)], r() < 0.05 ? null : (r() * 10).toFixed(1), ['0', '1', '2'][Math.floor(r() * 3)]]);
    const vars = [{ type: 'continuous' }, { type: 'nominal' }, { type: 'continuous' }, { type: 'ordinal', levels: ['0', '1', '2'] }];
    const G = CE.gowerMatrix(rows, vars, { weights: 'balanced' }), w = CE.gowerBalancedWeights(rows, vars), f = CE.gowerPairFn(rows, vars, w);
    let md = 0; for (let i = 0; i < 300; i++) for (let j = 0; j < 300; j++) if (i !== j) md = Math.max(md, Math.abs(f(i, j) - G.D[i * 300 + j]));
    const a = AE.lof(G, 20), b = AE.lofStream(300, (i, j) => G.D[i * 300 + j], 20), wd = Math.max(...w.map((v, j) => Math.abs(v - G.weights[j]))), ld = Math.max(...a.map((v, i) => Math.abs(v - b[i])));
    assert(wd < 1e-10 && md < 1e-12 && ld === 0, `weights ${wd}, pairs ${md}, lof ${ld}`);
    return `largest differences: weights ${wd.toExponential(1)}, dissimilarities ${md.toExponential(1)}, local outlier factor ${ld}`;
});
check('Mixed-data test (copula reference) stays valid with categories of several levels: at most 7% false positives for one population (60 datasets)', () => {
    let fp = 0, errors = 0;
    for (let s = 0; s < 60; s++) {
        const r = CE.createPRNG(900 + s), gz = () => { const u = r() || 1e-12, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
        const X = Array.from({ length: 250 }, () => {
            const a = gz(), b = 0.5 * a + 0.87 * gz(), c = 0.4 * a + 0.9 * gz();
            const cat = c < -0.7 ? 0 : c < 0 ? 1 : c < 0.7 ? 2 : 3;   // a 4-level category, correlated with a
            return [a, Math.exp(0.5 * b), b > 0.4 ? 1 : 0, ...[1, 2, 3].map(l => (cat === l ? 1 : 0))];   // reference coding
        });
        try { if (CE.copulaNullTest(X, 3, { B: 39, seed: s, groups: [[3, 4, 5]] }).pValue <= 0.05) fp++; } catch (e) { errors++; }
    }
    assert(errors === 0 && fp / 60 <= 0.07, `${fp} false positives, ${errors} errors`);
    return `${fp} of 60 false positives (${(100 * fp / 60).toFixed(1)}%); no failures`;
});
// Calibration of the tests of "no real sub-groups" under difficult distributions (one population each)
function oneSim(kind, s) {
    const r = CE.createPRNG(5000 + s), g = () => { const u = r() || 1e-12, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    return Array.from({ length: 200 }, () => {
        const a = g(), b = g(), c = g(), d = g(), base = [a, 0.6 * a + 0.8 * b, 0.5 * b + 0.86 * c, 0.4 * c + 0.9 * d];
        if (kind === 'heavy') { const w = Math.sqrt(3 / (g() ** 2 + g() ** 2 + g() ** 2)); return base.map(v => v * w); }
        return base.map(v => Math.exp(0.8 * v));   // skewed (log-normal)
    });
}
check('Tests of no real sub-groups with one population: Gaussian reference valid for heavy tails, copula reference valid for skewed measurements (at most 7% false positives, 40 datasets each)', () => {
    let fpHeavy = 0, fpSkewG = 0, fpSkewC = 0;
    for (let s = 0; s < 40; s++) {
        if (CE.nullClusterTest(CE.standardize(oneSim('heavy', s)).Z, 3, { B: 39, seed: s }).pValue <= 0.05) fpHeavy++;
        const Xs = oneSim('skewed', s);
        if (CE.nullClusterTest(CE.standardize(Xs).Z, 3, { B: 39, seed: s }).pValue <= 0.05) fpSkewG++;
        if (CE.copulaNullTest(Xs, 3, { B: 39, seed: s }).pValue <= 0.05) fpSkewC++;
    }
    // The Gaussian reference is not valid for skewed measurements: the app then uses the copula reference
    assert(fpHeavy / 40 <= 0.07 && fpSkewC / 40 <= 0.07 && fpSkewG / 40 > 0.2, `heavy ${fpHeavy}, skewed Gaussian ${fpSkewG}, skewed copula ${fpSkewC}`);
    return `false positives: heavy tails (Gaussian) ${fpHeavy}/40; skewed: Gaussian ${fpSkewG}/40 (why the app switches), copula ${fpSkewC}/40`;
});
check('Copula reference: power to detect sub-groups in mixed data (clearly separated: at least 90%; moderate overlap reported; 30 datasets each)', () => {
    const power = sep => { let hit = 0; for (let s = 0; s < 30; s++) {
        const r = CE.createPRNG(6000 + s), g = () => { const u = r() || 1e-12, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
        const X = Array.from({ length: 240 }, (_, i) => { const c = i % 3; return [g() + sep * (c === 0), g() + sep * (c === 1), Math.exp(0.5 * g() + 0.4 * sep * (c === 2)), r() < [0.2, 0.5, 0.8][c] ? 1 : 0]; });
        if (CE.copulaNullTest(X, 3, { B: 39, seed: s }).pValue <= 0.05) hit++; } return hit; };
    const clear = power(3), moderate = power(2.2);
    assert(clear >= 27, `clear ${clear}/30`);
    return `detected: clearly separated (3 SD) ${clear}/30; moderate overlap (2.2 SD) ${moderate}/30 (a conservative test: a large p does not show that sub-groups are absent)`;
});
// Self-organising map (the neural-network method for large datasets)
check('Self-organising map keeps the layout of the data: low topographic error on a two-dimensional sheet in 4 variables', () => {
    const r = CE.createPRNG(91), X = Array.from({ length: 1500 }, () => { const a = r() * 4, b = r() * 4; return [a, b, a + b, a - b].map(v => v + 0.05 * (r() - 0.5)); }), M1 = X.map(x => x.map(() => 1));
    const m = NN.som(CE.standardize(X).Z, M1, { rows: 10, cols: 10 });
    assert(m.te < 0.1, `topographic error ${m.te}`);
    return `topographic error ${(100 * m.te).toFixed(1)}%, quantisation error ${m.qe.toFixed(3)}`;
});
check('Self-organising map with grouped units recovers simulated referral-like patterns in mixed data as well as k-means (within 0.03), reproducibly', () => {
    const r = CE.createPRNG(92), gz = () => { const u = r() || 1e-12, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const pick = probs => { let u = r(), j = 0; while (j < probs.length - 1 && u > probs[j]) { u -= probs[j]; j++; } return j; };
    const pat = [{ src: [0.85, 0.05, 0.1], urg: [0.9, 0.1], days: 90, prior: 0.3 }, { src: [0.05, 0.85, 0.1], urg: [0.05, 0.95], days: 6, prior: 0.2 }, { src: [0.35, 0.05, 0.6], urg: [0.9, 0.1], days: 150, prior: 4 }];
    const rows = [], truth = [];
    for (let i = 0; i < 3000; i++) { const c = i % 3, q = pat[c]; truth.push(c); rows.push([['GP', 'ED', 'Physio'][pick(q.src)], ['Routine', 'Urgent'][pick(q.urg)], q.days * Math.exp(0.35 * gz()), Math.max(0, Math.round(q.prior + 0.8 * gz()))]); }
    const vars = [{ name: 'src', type: 'nominal' }, { name: 'urg', type: 'nominal' }, { name: 'days', type: 'continuous' }, { name: 'prior', type: 'continuous' }];
    const coded = NN.codeData(rows, vars), m = NN.som(coded.X, coded.M, {}), g = NN.weightedKmeans(m.W, m.hits, 3, { seed: 1 }), lab = m.bmu.map(u => g.labels[u]);
    const m2 = NN.som(coded.X, coded.M, {}), same = JSON.stringify(m.W) === JSON.stringify(m2.W), a = CE.adjustedRand(lab, truth);
    const km = CE.adjustedRand(CE.kmeans(coded.X, 3, { seed: 1, nInit: 10 }).labels, truth);
    assert(a >= 0.55 && a >= km - 0.03 && same, `ARI ${a} (k-means ${km}), reproducible ${same}`);
    return `adjusted Rand ${a.toFixed(3)} (k-means ${km.toFixed(3)}; ${m.rows} x ${m.cols} map); identical map when trained again`;
});
check('Multiple imputation (MICE) is reproducible and unchanged by refactoring: observed values kept, imputations drawn from observed values, output pinned', () => {
    // A regression lock: the imputations of this fixed dataset are pinned by a checksum, so a change to the
    // arithmetic or its order (for example in the flat design matrices) fails here even if other checks pass
    const r = CE.createPRNG(31), gz = () => { const u = r() || 1e-12, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const rows = Array.from({ length: 80 }, () => { const a = gz(), b = 0.6 * a + 0.8 * gz(); return { a: (50 + 10 * a).toFixed(1), b: (20 + 5 * b).toFixed(1), g: ['mild', 'moderate', 'severe'][Math.min(2, Math.floor(r() * 3))], s: r() < 0.5 + 0.2 * Math.sign(a) ? 'Yes' : 'No' }; });
    rows.forEach((o, i) => { if (i % 5 === 1) o.a = ''; if (i % 7 === 2) o.b = ''; if (i % 9 === 3) o.g = ''; if (i % 11 === 4) o.s = ''; });
    const cfg = { a: { role: 'feature', type: 'continuous' }, b: { role: 'feature', type: 'continuous' }, g: { role: 'feature', type: 'nominal' }, s: { role: 'feature', type: 'nominal' } };
    const run = () => DE.miceImpute(rows, cfg, { m: 3, iterations: 5, seed: 7 }).datasets;
    const d1 = run(), d2 = run(), cols = Object.keys(cfg);
    const kept = d1.every(d => d.every((o, i) => cols.every(c => rows[i][c] === '' || String(o[c]) === rows[i][c])));
    const donors = cols.every(c => { const obs = new Set(rows.map(o => o[c]).filter(v => v !== '').map(String)); return d1.every(d => d.every((o, i) => rows[i][c] !== '' || obs.has(String(o[c])) || obs.has(Number(o[c]).toFixed(1)))); });
    const text = JSON.stringify(d1.map(d => d.map(o => cols.map(c => o[c]))));
    let h = 2166136261; for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 16777619) >>> 0; }
    const PINNED = '36bd8107';
    assert(kept && donors && text === JSON.stringify(d2.map(d => d.map(o => cols.map(c => o[c])))) && h.toString(16) === PINNED, `kept ${kept}, donors ${donors}, checksum ${h.toString(16)}`);
    return `3 imputed datasets identical when run again; checksum ${h.toString(16)}`;
});
console.log(`\n${passed}/${passed + failed} behaviour checks passed`);
process.exit(failed ? 1 : 0);
