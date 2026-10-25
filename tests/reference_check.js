// Computes the engine's results on fixed synthetic data and writes them (tests/reference_values.json) together
// with the data (tests/ref_data/*.csv) for comparison with scikit-learn, SciPy (tests/reference_check.py) and R's
// cluster package (tests/reference_check.R, run by the Python script).
// Usage (from the project folder):  node tests/reference_check.js && python3 tests/reference_check.py
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');
const dir = path.join(__dirname, '..');
global.window = globalThis;
vm.runInThisContext(fs.readFileSync(path.join(dir, 'cluster-engine.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(dir, 'embed-engine.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(dir, 'anomaly-engine.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(dir, 'nn-engine.js'), 'utf8'));
const CE = globalThis.ClusterEngine, EE = globalThis.EmbedEngine, AE = globalThis.AnomalyEngine, NN = globalThis.NNEngine;
const rng = CE.createPRNG(12345);
const g = () => { const u = rng() || 1e-12, v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const out = {};
const dataDir = path.join(__dirname, 'ref_data');
fs.mkdirSync(dataDir, { recursive: true });
const writeCsv = (name, header, rows) => fs.writeFileSync(path.join(dataDir, name), [header.join(','), ...rows.map(r => r.map(v => (v === null ? 'NA' : v)).join(','))].join('\n') + '\n');

// 1. Three Gaussian clusters in 3 dimensions (unequal sizes and spreads)
const centres = [[0, 0, 0], [4, 1, -2], [1, 5, 2]], sizes = [70, 50, 30], spread = [1, 0.7, 1.3];
const X = [];
centres.forEach((c, k) => { for (let i = 0; i < sizes[k]; i++) X.push(c.map(m => Number((m + spread[k] * g()).toFixed(6)))); });
writeCsv('blobs.csv', ['x1', 'x2', 'x3'], X);
out.blobs = { n: X.length };
// k-means from fixed starting centres (rows 0, 70, 120), and best of 10 k-means++ starts
const init = [X[0], X[70], X[120]];
const L = CE.lloyd(X, init, { maxIter: 1000, tol: 0 });
out.lloyd = { init, centres: L.centres, inertia: L.inertia, labels: L.labels };
const KM = CE.kmeans(X, 3, { nInit: 10, seed: 7 });
out.kmeans = { inertia: KM.inertia, labels: KM.labels };
const E = CE.euclideanMatrix(X);
out.indices = { silhouette: CE.silhouette(E, KM.labels).mean, silhouetteValues: CE.silhouette(E, KM.labels).values.slice(0, 10), ch: CE.calinskiHarabasz(X, KM.labels), db: CE.daviesBouldin(X, KM.labels) };
const other = X.map(() => Math.floor(rng() * 4));
out.ari = { other, value: CE.adjustedRand(KM.labels, other) };
const P = CE.pca(X);
out.pca = { ratio: P.varianceRatio, variance: P.variance, components: P.components };
out.hier = {};
for (const m of ['ward', 'average', 'complete', 'single']) { const H = CE.hierarchical(E, m); out.hier[m] = { heights: H.Z.map(z => z[2]), cut3: CE.cutTree(H, X.length, 3) }; }
// Gaussian mixture from the k-means partition of the fixed start (same parameters handed to scikit-learn)
for (const cov of ['full', 'diag']) {
    const R = L.labels.map(l => [0, 1, 2].map(c => (c === l ? 1 : 0)));
    const start = CE.mStep(X, R, 3, cov, 1e-6);
    const G = CE.gmmFromParams(X, start, { covType: cov, reg: 1e-6, maxIter: 2000, tol: 1e-12 });
    const nParams = 2 + 3 * 3 + (cov === 'diag' ? 9 : 18);
    out['gmm_' + cov] = { start, means: G.means, weights: G.weights, meanLogLik: G.meanLogLik, bic: -2 * G.logLik + nParams * Math.log(X.length), converged: G.converged };
}

// 2. Mixed clinical-type data with missing values: Gower dissimilarity, PAM and silhouette on it
const sexes = ['F', 'M'], grades = ['mild', 'moderate', 'severe'], sites = ['A', 'B', 'C'];
const M = [];
for (let i = 0; i < 60; i++) {
    const grp = i % 2;
    M.push([Number((60 + 10 * grp + 8 * g()).toFixed(1)), Number((25 + 4 * g()).toFixed(1)), grades[Math.min(2, Math.max(0, grp + (rng() < 0.3 ? 1 : 0)))], sexes[rng() < 0.5 ? 0 : 1], sites[Math.floor(rng() * 3)]]);
}
[[3, 1], [10, 2], [17, 0], [25, 4], [40, 1]].forEach(([i, j]) => { M[i][j] = null; });
writeCsv('mixed.csv', ['age', 'bmi', 'grade', 'sex', 'site'], M);
const vars = [{ type: 'continuous' }, { type: 'continuous' }, { type: 'ordinal', levels: grades }, { type: 'nominal' }, { type: 'nominal' }];
const Gw = CE.gowerMatrix(M, vars);
out.gower = { firstRows: Array.from({ length: 5 }, (_, i) => Array.from(Gw.D.slice(i * 60, i * 60 + 60))) };
const GwB = CE.gowerMatrix(M, vars, { weights: 'balanced' });
fs.writeFileSync(path.join(dataDir, 'gower_weights.txt'), GwB.weights.join(',') + '\n');
out.gowerBalanced = { weights: GwB.weights, firstRows: Array.from({ length: 5 }, (_, i) => Array.from(GwB.D.slice(i * 60, i * 60 + 60))) };
const PG = CE.pam(Gw, 3);
out.pamGower = { objective: PG.objective, medoids: PG.medoids.slice().sort((a, b) => a - b), silhouette: CE.silhouette(Gw, PG.labels).mean, labels: PG.labels };
const PE = CE.pam(E, 3);
out.pamEuclid = { objective: PE.objective, buildObjective: PE.buildObjective, medoids: PE.medoids.slice().sort((a, b) => a - b) };
// Tests used to describe clusters: chi-square and Kruskal-Wallis (with ties)
const grp = M.map((_, i) => i % 3), sexv = M.map(r => r[3]), agev = M.map(r => (r[0] === null ? 50 : r[0])), gradev = M.map(r => r[2] ?? 'mild');
out.tests = { grp, sexv, agev, gradev, chi: CE.chiSquareTest(sexv, grp), chiGrade: CE.chiSquareTest(gradev, grp), kw: CE.kruskalWallis(agev, grp), kwTies: CE.kruskalWallis(gradev.map(v => grades.indexOf(v)), grp), q: [CE.gammaQ(2.5, 3.1), CE.gammaQ(10, 4), CE.gammaQ(0.5, 20)] };
// 3. Classification tree that turns clusters into rules (vs scikit-learn DecisionTreeClassifier), and Cohen's kappa
const noisy = KM.labels.map(l => (rng() < 0.2 ? Math.floor(rng() * 3) : l));
const treeOut = {};
for (const [d, leaf] of [[2, 5], [3, 5], [4, 10]]) {
    const T = CE.classificationTree(X, [0, 1, 2].map(j => ({ name: 'x' + (j + 1), type: 'continuous' })), noisy, { maxDepth: d, minLeaf: leaf });
    treeOut[`d${d}_l${leaf}`] = X.map(x => CE.predictTree(T, x));
}
out.tree = { labels: noisy, predictions: treeOut, kappa: CE.cohenKappa(noisy, KM.labels) };
// 4. Latent class analysis: categorical items with missing values (vs R poLCA) and numbers (vs R mclust "VVI")
const items = [], itemTruth = [];
const probs = [[[0.85, 0.15], [0.7, 0.2, 0.1], [0.9, 0.1], [0.6, 0.4]], [[0.2, 0.8], [0.1, 0.3, 0.6], [0.3, 0.7], [0.5, 0.5]], [[0.5, 0.5], [0.3, 0.6, 0.1], [0.1, 0.9], [0.1, 0.9]]];
for (let i = 0; i < 500; i++) {
    const c = i < 220 ? 0 : i < 380 ? 1 : 2; itemTruth.push(c);
    items.push(probs[c].map(pv => { let u = rng(), l = 0; while (l < pv.length - 1 && u > pv[l]) { u -= pv[l]; l++; } return String(l + 1); }));
}
for (let i = 0; i < 500; i++) if (rng() < 0.05) items[i][Math.floor(rng() * 4)] = null;
writeCsv('lca_items.csv', ['i1', 'i2', 'i3', 'i4'], items);
const itemVars = [2, 3, 2, 2].map(L => ({ type: 'nominal', levels: Array.from({ length: L }, (_, l) => String(l + 1)) }));
const LC = CE.lca(items, itemVars, 3, { nInit: 30, seed: 11, tol: 1e-15, maxIter: 100000 });
out.lcaItems = { logLik: LC.logLik, bic: LC.bic, nParams: LC.nParams, shares: LC.model.pi.slice().sort((a, b) => a - b) };
const LP = CE.lca(X, [0, 1, 2].map(() => ({ type: 'continuous' })), 3, { nInit: 20, seed: 5, reg: 0, tol: 1e-15, maxIter: 100000 });
out.lcaCont = { logLik: LP.logLik, bic: LP.bic, shares: LP.model.pi.slice().sort((a, b) => a - b) };
// 5. Maps: classical scaling, t-SNE (vs scikit-learn, exact method), UMAP graph and curve (vs umap-learn), trustworthiness
{
    const n = X.length, DE2 = EE.squared(E), Y0 = EE.classicalMDS(E, 2);
    const sd0 = Math.sqrt(Y0.reduce((a, r) => a + r[0] * r[0], 0) / n), init = Y0.map(r => r.map(v => v / sd0 * 1e-4));
    const P = EE.tsneP(DE2, 30), Pc = []; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) Pc.push(P[i * n + j]);
    const rY = CE.createPRNG(77), Yr = Array.from({ length: n }, () => [rY() * 4 - 2, rY() * 4 - 2]);
    const KG = EE.tsneKL(P, Yr, n, 1);
    const T = EE.tsne(DE2, { perplexity: 30, init });
    const T20 = EE.createTSNE(DE2, { perplexity: 30, init, maxIter: 20 }); T20.step(20);
    const G = EE.umapGraph(E, 15), dense = Array.from({ length: n * n }, () => 0); G.rows.forEach((r, e) => { dense[r * n + G.cols[e]] = G.vals[e]; });
    const U = EE.umap(E, { seed: 3 });
    out.embed = { mds: Y0, init, Pc, Yr, kl: KG.kl, grad: KG.grad, tsneY: T.Y, tsneKL: T.kl, tsne20: T20.Y, sigmas: Array.from(G.sigmas), rhos: Array.from(G.rhos), graph: dense, ab: EE.findAB(1, 0.1),
        trustTsne: EE.trustworthiness(E, T.Y, 5), trustUmap: EE.trustworthiness(E, U.Y, 5), umapY: U.Y, trustMds10: EE.trustworthiness(E, Y0, 10) };
}
// 6. Dimensions: FAMD on mixed data, and its special cases MCA (categories only) and PCA (numbers only), vs FactoMineR
{
    const comp = M.filter(r => r.every(v => v !== null));
    writeCsv('famd.csv', ['age', 'bmi', 'grade', 'sex', 'site'], comp.map(r => [r[0], r[1], grades.indexOf(r[2]), r[3], r[4]]));
    const fv = [{ name: 'age', type: 'continuous' }, { name: 'bmi', type: 'continuous' }, { name: 'grade', type: 'ordinal', levels: grades }, { name: 'sex', type: 'nominal' }, { name: 'site', type: 'nominal' }];
    const F = EE.famd(comp, fv, { nDim: 5 });
    const Fc = EE.famd(comp.map(r => [r[2], r[3], r[4]]), [{ name: 'grade', type: 'nominal', levels: grades }, fv[3], fv[4]], { nDim: 5 });
    const Fn = EE.famd(comp.map(r => [r[0], r[1]]), fv.slice(0, 2), { nDim: 2 });
    out.famd = { n: comp.length, eig: F.eig, scores: F.scores, varCoord: F.varCoord, contrib: F.contrib, corr: F.corr.slice(0, 3), eigCat: Fc.eig, eigNum: Fn.eig, scoresNum: Fn.scores };
}
// 7. Unusual patients: robust covariance (MCD), local outlier factor, isolation forest, chi-square, robust z
{
    const A = [], isOut = [];
    for (let i = 0; i < 200; i++) { const z1 = g(), z2 = g(), z3 = g(); A.push([10 + 2 * z1, 5 + 1.2 * z1 + 0.8 * z2, -3 + 0.5 * z2 + z3].map(v => Number(v.toFixed(6)))); isOut.push(0); }
    for (let i = 0; i < 15; i++) { A.push([10 + 3 * g(), 9 + g(), 2 + g()].map(v => Number(v.toFixed(6)))); isOut.push(1); }
    writeCsv('anomaly.csv', ['a', 'b', 'c', 'outlier'], A.map((r, i) => [...r, isOut[i]]));
    const R = AE.mcd(A, { seed: 4 });
    const DA = CE.euclideanMatrix(A);
    out.anomaly = { location: R.location, covariance: R.covariance, dist: R.dist, support: R.support.map(Number), rawSupport: R.rawSupport.map(Number), rawLocation: R.rawLocation,
        lof: AE.lof(DA, 20), iso: AE.isolationForest(A, { seed: 3 }), avgPath: [1, 2, 3, 10, 256, 1000].map(AE.avgPath),
        chi2: [[0.975, 3], [0.999, 5], [0.5, 1], [0.025, 7]].map(([q, df]) => AE.chi2Ppf(q, df)), rz: AE.robustZ(A.map(r => r[0])) };
}
// 8. Autoencoder: gradients vs finite differences; Adam steps vs an independent NumPy implementation; linear autoencoder vs PCA
{
    const Z = CE.standardize(X).Z, Mk = Z.map((r, i) => r.map((_, j) => ((i * 7 + j * 3) % 23 === 0 ? 0 : 1)));
    const Zm = Z.map((r, i) => r.map((v, j) => (Mk[i][j] ? v : 0)));
    const net = NN.createAutoencoder({ p: 3, hidden: [5], latent: 2, activation: 'tanh', seed: 9 });
    const init = NN.clone(net), all = Zm.map((_, i) => i);
    // Finite-difference check of every gradient (central differences)
    const g = NN.lossGrad(net, Zm, Mk, all, 1e-3); let worst = 0;
    net.layers.forEach((L, l) => { L.W.forEach((row, o) => row.forEach((w, k) => { const h = 1e-6; row[k] = w + h; const a1 = NN.lossGrad(net, Zm, Mk, all, 1e-3).loss; row[k] = w - h; const a2 = NN.lossGrad(net, Zm, Mk, all, 1e-3).loss; row[k] = w; worst = Math.max(worst, Math.abs((a1 - a2) / (2 * h) - g.gW[l][o][k]) / Math.max(1e-8, Math.abs(g.gW[l][o][k]))); }));
        L.b.forEach((bb, o) => { const h = 1e-6; L.b[o] = bb + h; const a1 = NN.lossGrad(net, Zm, Mk, all, 1e-3).loss; L.b[o] = bb - h; const a2 = NN.lossGrad(net, Zm, Mk, all, 1e-3).loss; L.b[o] = bb; worst = Math.max(worst, Math.abs((a1 - a2) / (2 * h) - g.gb[l][o]) / Math.max(1e-8, Math.abs(g.gb[l][o]))); }); });
    const st = {}, losses = [];
    for (let s = 0; s < 30; s++) { const gg = NN.lossGrad(net, Zm, Mk, all, 1e-3); losses.push(gg.loss); NN.adamStep(net, gg, st, { lr: 0.01 }); }
    // Linear autoencoder with a 1- and a 2-dimensional bottleneck: full batch, no weight decay, no validation
    const lin = [1, 2].map(d => { const T = NN.createTrainer(Z, Z.map(r => r.map(() => 1)), { hidden: [], latent: d, activation: 'linear', seed: 3, lr: 0.01, weightDecay: 0, batch: Z.length, maxEpochs: 4000, patience: 4000, validation: 0 }); while (!T.done) T.step(500); return NN.lossGrad(T.net, Z, Z.map(r => r.map(() => 1)), Z.map((_, i) => i), 0).loss; });
    out.nn = { Z: Zm, M: Mk, init: init.layers.map(L => ({ W: L.W, b: L.b, act: L.act })), final: net.layers.map(L => ({ W: L.W, b: L.b })), losses, gradRelErr: worst, linLoss: lin };
}
// 9. Self-organising map: with no neighbourhood it is Lloyd's k-means (vs scikit-learn); weighted k-means for the units
{
    const ones = X.map(r => r.map(() => 1)), init = [X[0], X[70], X[120]];
    const S0 = NN.som(X, ones, { rows: 1, cols: 3, epochs: 200, sigma0: 0, sigma1: 0, init });
    const wts = X.map((_, i) => 1 + (i % 4)), WK = NN.weightedKmeans(X, wts, 3, { init, maxIter: 1000 });
    // Unordered centres for comparison (weighted k-means orders its groups by size)
    out.som = { init, W: S0.W, wts, wkCentres: WK.centres };
}
// 10. Describe-only variables: Benjamini-Hochberg q-values, Cramer's V, epsilon squared
{
    const ps = [0.001, 0.04, 0.03, 0.2, 0.0005, 0.5, null, 0.049];
    const chi = out.tests.chi, sexLevels = new Set(out.tests.sexv).size, grpLevels = new Set(out.tests.grp).size;
    out.effects = { ps, q: CE.benjaminiHochberg(ps), v: CE.cramersV(chi.statistic, out.tests.sexv.length, sexLevels, grpLevels), eps: CE.epsilonSquared(out.tests.kw.statistic, out.tests.agev.length) };
}
// 11. Audit additions: bootstrap Jaccard bookkeeping, gap statistic, latent classes on mixed data, normal distribution
{
    // (a) Bootstrap stability: every resample and its clustering are recorded, so that the Jaccard values and the
    // patient-level stability can be recomputed independently (Hennig 2007 definitions)
    const lab = CE.kmeans(X, 3, { seed: 1, nInit: 5 }).labels, rec = [];
    const st = CE.bootstrapStability(X.length, lab, idx => { const l = CE.kmeans(idx.map(i => X[i]), 3, { seed: 2, nInit: 3 }).labels; rec.push({ idx, l }); return l; }, { B: 20, seed: 5 });
    out.boot = { labels: lab, resamples: rec, meanJaccard: st.perCluster.map(c => c.meanJaccard), patientStability: st.patientStability, shareRange: st.perCluster.map(c => c.shareRange) };
    // (b) Gap statistic (Tibshirani's squared distances; compared with R cluster::clusGap, d.power = 2)
    const G = CE.gapStatistic(X, 6, { B: 200, seed: 3, nInit: 10 });
    out.gap = { table: G.table.map(t => ({ logW: t.logW, gap: t.gap, se: t.se })), kBest: G.kBest };
    // (c) Latent classes on mixed data with missing values (compared with R VarSelLCM)
    const r = CE.createPRNG(77), gz = () => { const u = r() || 1e-12, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const pick = probs => { let u = r(), j = 0; while (j < probs.length - 1 && u > probs[j]) { u -= probs[j]; j++; } return j; };
    const P3 = [[0.7, 0.2, 0.1], [0.15, 0.7, 0.15], [0.1, 0.2, 0.7]], mix = [];
    for (let i = 0; i < 300; i++) { const c = i % 3; mix.push([(2 * c + gz()).toFixed(4), (c === 1 ? 3 : 0) + 0.8 * gz(), ['a', 'b', 'c'][pick(P3[c])], ['x', 'y', 'z'][pick(P3[(c + 1) % 3])]].map((v, j) => ((i * 13 + j * 7) % 19 === 0 ? null : v))); }
    writeCsv('lca_mixed.csv', ['n1', 'n2', 'c1', 'c2'], mix.map(row => row.map(v => (typeof v === 'number' ? v.toFixed(4) : v))));
    const mixRows = mix.map(row => row.map((v, j) => (v === null ? null : j < 2 ? Number(v) : v)));
    const LM = CE.lca(mixRows, [{ type: 'continuous' }, { type: 'continuous' }, { type: 'nominal', levels: ['a', 'b', 'c'] }, { type: 'nominal', levels: ['x', 'y', 'z'] }], 3, { nInit: 30, seed: 4 });
    out.lcaMixed = { logLik: LM.logLik, nParams: LM.nParams, shares: LM.model.pi.slice().sort((a, b) => a - b) };
    // (d) Normal distribution functions
    const ps = [1e-10, 1e-6, 0.001, 0.01, 0.02425, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97575, 0.99, 0.999, 1 - 1e-6], xs = [-8, -4, -2.5, -1, -0.3, 0, 0.3, 1, 2.5, 4, 8];
    out.norm = { ps, inv: ps.map(CE.normInv), xs, cdf: xs.map(CE.normCdf) };
}
fs.writeFileSync(path.join(__dirname, 'reference_values.json'), JSON.stringify(out));
console.log('Engine values written to tests/reference_values.json; now run: python3 tests/reference_check.py');
