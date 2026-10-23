// Statistical behaviour of the engine on simulated data (the counterpart of the predictor's simulation checks):
// does each procedure find structure when it exists, and decline to when it does not?
// Usage: node tests/behaviour_check.js
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');
global.window = globalThis;
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'cluster-engine.js'), 'utf8'));
const CE = globalThis.ClusterEngine;
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
console.log(`\n${passed}/${passed + failed} behaviour checks passed`);
process.exit(failed ? 1 : 0);
