"""Compares the engine's results (tests/reference_values.json, from reference_check.js) with scikit-learn, SciPy
and R's cluster package (tests/reference_check.R). Usage: node tests/reference_check.js && python3 tests/reference_check.py"""
import json, os, subprocess, shutil, sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster

here = os.path.dirname(os.path.abspath(__file__))
o = json.load(open(os.path.join(here, 'reference_values.json')))
X = np.loadtxt(os.path.join(here, 'ref_data', 'blobs.csv'), delimiter=',', skiprows=1)
res = []
def chk(name, app, ref, tol):
    a, r = np.atleast_1d(np.asarray(app, float)), np.atleast_1d(np.asarray(ref, float))
    d = float(np.max(np.abs(a - r))) if a.shape == r.shape else float('inf')
    res.append(('PASS' if d <= tol else 'FAIL', name, d, tol))

# k-means
km = KMeans(3, init=np.array(o['lloyd']['init']), n_init=1, algorithm='lloyd', tol=0, max_iter=1000).fit(X)
chk('k-means (Lloyd) from fixed centres: centres (vs scikit-learn)', o['lloyd']['centres'], km.cluster_centers_, 1e-9)
chk('k-means (Lloyd) from fixed centres: within-cluster sum of squares', o['lloyd']['inertia'], km.inertia_, 1e-8)
chk('k-means (Lloyd) from fixed centres: labels identical', adjusted_rand_score(o['lloyd']['labels'], km.labels_), 1.0, 1e-12)
best = min((KMeans(3, n_init=1, random_state=s).fit(X) for s in range(20)), key=lambda m: m.inertia_)
chk('k-means++ best of 10 starts reaches the best scikit-learn solution (20 starts)', o['kmeans']['inertia'], best.inertia_, 1e-8)
lab = np.array(o['kmeans']['labels'])
chk('Silhouette, mean (vs scikit-learn)', o['indices']['silhouette'], silhouette_score(X, lab), 1e-12)
chk('Silhouette, per patient', o['indices']['silhouetteValues'], silhouette_samples(X, lab)[:10], 1e-12)
chk('Calinski-Harabasz index', o['indices']['ch'], calinski_harabasz_score(X, lab), 1e-8)
chk('Davies-Bouldin index', o['indices']['db'], davies_bouldin_score(X, lab), 1e-12)
chk('Adjusted Rand index', o['ari']['value'], adjusted_rand_score(lab, o['ari']['other']), 1e-12)
# PCA
p = PCA().fit(X)
chk('PCA: variance explained (ratio)', o['pca']['ratio'], p.explained_variance_ratio_, 1e-12)
chk('PCA: component variances', o['pca']['variance'], p.explained_variance_, 1e-10)
chk('PCA: loadings (up to sign)', np.abs(o['pca']['components']), np.abs(p.components_), 1e-9)
# hierarchical
for m in ['ward', 'average', 'complete', 'single']:
    Z = linkage(X, method=m)
    chk(f'Hierarchical ({m}): merge heights (vs SciPy linkage)', o['hier'][m]['heights'], Z[:, 2], 1e-9)
    chk(f'Hierarchical ({m}): 3-cluster cut identical to SciPy fcluster', adjusted_rand_score(o['hier'][m]['cut3'], fcluster(Z, 3, 'maxclust')), 1.0, 1e-12)
# Gaussian mixtures from the same starting parameters
for cov in ['full', 'diag']:
    g = o['gmm_' + cov]; st = g['start']
    covs = np.array(st['covs'])
    prec = np.array([np.linalg.inv(c) for c in covs]) if cov == 'full' else np.array([1 / np.diag(c) for c in covs])
    gm = GaussianMixture(3, covariance_type=cov, weights_init=st['weights'], means_init=st['means'], precisions_init=prec, tol=1e-12, max_iter=2000, reg_covar=1e-6).fit(X)
    chk(f'Gaussian mixture ({cov}): means (vs scikit-learn, same start)', g['means'], gm.means_, 1e-6)
    chk(f'Gaussian mixture ({cov}): weights', g['weights'], gm.weights_, 1e-6)
    chk(f'Gaussian mixture ({cov}): mean log-likelihood', g['meanLogLik'], gm.score(X), 1e-8)
    chk(f'Gaussian mixture ({cov}): BIC', g['bic'], gm.bic(X), 1e-5)
# Tests used to describe clusters
from scipy import stats
import pandas as pd
t = o['tests']
for key, vals in [('chi', t['sexv']), ('chiGrade', t['gradev'])]:
    tab = pd.crosstab(pd.Series(vals), pd.Series(t['grp']))
    c2, pv, df, _ = stats.chi2_contingency(tab, correction=False)
    chk(f'Chi-square test ({key}): statistic and p (vs SciPy)', [t[key]['statistic'], t[key]['p']], [c2, pv], 1e-10)
grades = ['mild', 'moderate', 'severe']
for key, vals in [('kw', t['agev']), ('kwTies', [grades.index(v) for v in t['gradev']])]:
    groups = [[v for v, gg in zip(vals, t['grp']) if gg == k] for k in range(3)]
    h, pv = stats.kruskal(*groups)
    chk(f'Kruskal-Wallis test ({key}): statistic and p (vs SciPy)', [t[key]['statistic'], t[key]['p']], [h, pv], 1e-10)
from scipy.special import gammaincc
chk('Regularised upper incomplete gamma function (vs SciPy)', t['q'], [gammaincc(2.5, 3.1), gammaincc(10, 4), gammaincc(0.5, 20)], 1e-12)
# Balanced Gower weights: 1 / mean pairwise contribution of each variable (independent computation)
import csv as _csv, itertools
rowsM = list(_csv.reader(open(os.path.join(here, 'ref_data', 'mixed.csv'))))[1:]
def contrib(j):
    vals = [r[j] for r in rowsM]
    if j in (0, 1):
        xs = [float(v) for v in vals if v != 'NA']; rg = max(xs) - min(xs)
        f = lambda a, b: abs(float(a) - float(b)) / rg
    elif j == 2:
        order = ['mild', 'moderate', 'severe']; f = lambda a, b: abs(order.index(a) - order.index(b)) / 2
    else:
        f = lambda a, b: 0.0 if a == b else 1.0
    d = [f(a, b) for a, b in itertools.combinations(vals, 2) if a != 'NA' and b != 'NA']
    return len(d) / sum(d)
chk('Balanced Gower weights (1 / mean contribution of each variable)', o['gowerBalanced']['weights'], [contrib(j) for j in range(5)], 1e-10)
# Classification tree (rules from clusters) and Cohen's kappa
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import cohen_kappa_score
yt = np.array(o['tree']['labels'])
for key, pred in o['tree']['predictions'].items():
    d, leaf = int(key.split('_')[0][1:]), int(key.split('_')[1][1:])
    sk = DecisionTreeClassifier(max_depth=d, min_samples_leaf=leaf, random_state=0).fit(X, yt).predict(X)
    chk(f'Classification tree, depth {d}, at least {leaf} per leaf: same predictions as scikit-learn', float(np.mean(np.array(pred) != sk)), 0.0, 0)
chk("Cohen's kappa (vs scikit-learn)", o['tree']['kappa'], cohen_kappa_score(yt, o['kmeans']['labels']), 1e-12)
# Maps
import warnings
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE, trustworthiness
from sklearn.manifold._t_sne import _joint_probabilities, _kl_divergence
em = o['embed']; n = X.shape[0]; Dm = squareform(pdist(X))
sc = PCA(2).fit_transform(X)
chk('Classical scaling of Euclidean distances equals PCA scores (up to sign)', np.abs(em['mds']), np.abs(sc), 1e-8)
Pref = _joint_probabilities(Dm ** 2, 30, 0)
chk('t-SNE joint probabilities from the perplexity (vs scikit-learn)', em['Pc'], Pref, 1e-7)
kl, grad = _kl_divergence(np.array(em['Yr']).ravel(), np.array(em['Pc']), 1, n, 2)
chk('t-SNE divergence (Kullback-Leibler) of a given map (vs scikit-learn)', em['kl'], kl, 1e-10)
chk('t-SNE gradient of a given map (vs scikit-learn)', np.array(em['grad']).ravel(), grad, 1e-10)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ts = TSNE(2, perplexity=30, init=np.array(em['init']), method='exact', metric='precomputed', learning_rate='auto', max_iter=1000, random_state=0).fit(Dm.copy())  # squares the given distances itself, in place
from sklearn.manifold._t_sne import _gradient_descent
p20, _, _ = _gradient_descent(_kl_divergence, np.array(em['init']).ravel(), it=0, max_iter=20, n_iter_check=50, n_iter_without_progress=250, momentum=0.5, learning_rate=50.0, min_gain=0.01, min_grad_norm=1e-7, kwargs=dict(compute_error=True), args=[np.array(em['Pc']) * 12, 1, n, 2])
chk('t-SNE descent, first 20 iterations from the same start: identical coordinates (vs scikit-learn)', np.array(em['tsne20']).ravel(), p20, 1e-9 * np.abs(p20).max())
# Beyond that, rounding differences grow (t-SNE's early phase is chaotic), so the finished maps are compared by quality
tr_sk = trustworthiness(Dm, ts.embedding_, n_neighbors=5, metric='precomputed')
chk('t-SNE finished map: divergence no worse than scikit-learn (+5%) and trustworthiness within 0.01', max(0.0, em['tsneKL'] - 1.05 * ts.kl_divergence_) + max(0.0, tr_sk - 0.01 - em['trustTsne']), 0.0, 0)
chk('Trustworthiness (vs scikit-learn)', [em['trustTsne'], em['trustUmap'], em['trustMds10']], [trustworthiness(Dm, np.array(em['tsneY']), n_neighbors=5, metric='precomputed'), trustworthiness(Dm, np.array(em['umapY']), n_neighbors=5, metric='precomputed'), trustworthiness(Dm, np.array(em['mds']), n_neighbors=10, metric='precomputed')], 1e-12)
try:
    import umap
    from umap.umap_ import fuzzy_simplicial_set, find_ab_params
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g, sig, rho = fuzzy_simplicial_set(Dm, 15, np.random.RandomState(0), 'precomputed')
        chk('UMAP neighbour scales (sigma, rho) (vs umap-learn)', np.r_[em['sigmas'], em['rhos']], np.r_[sig, rho], 1e-5)
        chk('UMAP fuzzy neighbour graph weights (vs umap-learn)', np.array(em['graph']), g.toarray().ravel(), 1e-5)
        a, b = find_ab_params(1.0, 0.1)
        chk('UMAP curve parameters a, b for min_dist 0.1 (vs umap-learn)', [em['ab']['a'], em['ab']['b']], [a, b], 1e-4)
        tu = [trustworthiness(Dm, umap.UMAP(metric='precomputed', random_state=s).fit_transform(Dm), n_neighbors=5, metric='precomputed') for s in range(3)]
    chk('UMAP map at least as trustworthy as umap-learn (within 0.01 of its mean over 3 seeds)', max(0.0, np.mean(tu) - 0.01 - em['trustUmap']), 0.0, 0)
except ImportError:
    print('umap-learn not installed: UMAP checks skipped')
# Unusual patients
from sklearn.covariance import MinCovDet
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2 as _chi2, median_abs_deviation
an = o['anomaly']; AA = np.loadtxt(os.path.join(here, 'ref_data', 'anomaly.csv'), delimiter=',', skiprows=1); Xa, ya = AA[:, :3], AA[:, 3]
# FAST-MCD is a random search with near-equal local optima: ours must reach a determinant at least as small as the best
# scikit-learn finds over 10 seeds; from the same raw subset, the corrected and reweighted results must be identical
ours_sup = np.array(an['rawSupport'], bool); ld = lambda S: np.linalg.slogdet(np.cov(Xa[S].T, bias=True))[1]
fits = [MinCovDet(random_state=s).fit(Xa) for s in range(10)]
chk('Robust covariance (MCD): smallest determinant at least as small as scikit-learn MinCovDet finds (10 seeds)', max(0.0, ld(ours_sup) - min(ld(f.raw_support_) for f in fits) - 1e-12), 0.0, 0)
same = [f for f in fits if (f.raw_support_ == ours_sup).all()]
if same:
    mc = same[0]
    chk('Robust covariance (MCD), same raw subset: location and covariance after reweighting (vs scikit-learn)', np.r_[an['location'], np.ravel(an['covariance'])], np.r_[mc.location_, np.ravel(mc.covariance_)], 1e-10)
    chk('Robust covariance (MCD), same raw subset: squared robust distances of all patients', an['dist'], mc.dist_, 1e-8)
else:
    chk('Robust covariance (MCD): scikit-learn reaches the same raw subset with some seed', 1.0, 0.0, 0)
lofr = -LocalOutlierFactor(n_neighbors=20, metric='precomputed').fit(squareform(pdist(Xa))).negative_outlier_factor_
chk('Local outlier factor (vs scikit-learn)', an['lof'], lofr, 1e-9)
chk('Isolation forest: average path length c(n) (vs scikit-learn)', an['avgPath'], _average_path_length(np.array([1, 2, 3, 10, 256, 1000])), 1e-12)
auc_ours = roc_auc_score(ya, an['iso']); auc_sk = np.mean([roc_auc_score(ya, -IsolationForest(random_state=s).fit(Xa).score_samples(Xa)) for s in range(5)])
chk('Isolation forest ranks planted outliers as well as scikit-learn (AUC within 0.02 of its mean over 5 seeds)', max(0.0, auc_sk - 0.02 - auc_ours), 0.0, 0)
chk('Chi-square quantiles (vs SciPy)', an['chi2'], [_chi2.ppf(0.975, 3), _chi2.ppf(0.999, 5), _chi2.ppf(0.5, 1), _chi2.ppf(0.025, 7)], 1e-9)
chk('Robust z-scores: median and scaled MAD (vs SciPy)', [an['rz']['med'], an['rz']['scale']], [np.median(Xa[:, 0]), median_abs_deviation(Xa[:, 0], scale='normal')], 1e-4)
# Autoencoder: independent NumPy forward/backward pass and Adam, from the same starting weights
nn = o['nn']; Zm, Mk = np.array(nn['Z']), np.array(nn['M'], float)
Ws = [np.array(L['W'], float) for L in nn['init']]; bs = [np.array(L['b'], float) for L in nn['init']]; acts = [L['act'] for L in nn['init']]
mW = [np.zeros_like(W) for W in Ws]; vW = [np.zeros_like(W) for W in Ws]; mb = [np.zeros_like(b) for b in bs]; vb = [np.zeros_like(b) for b in bs]
lossesN = []; wd, lr, b1, b2, eps = 1e-3, 0.01, 0.9, 0.999, 1e-8; nP, pP = Zm.shape
for t in range(1, 31):
    A = [Zm]
    for W, b, a in zip(Ws, bs, acts):
        z = A[-1] @ W.T + b; A.append(np.tanh(z) if a == 'tanh' else z)
    Y = A[-1]; lossesN.append(float(np.sum(Mk * (Y - Zm) ** 2) / (pP * nP) + 0.5 * wd * sum(np.sum(W ** 2) for W in Ws)))
    D = Mk * 2 * (Y - Zm) / (pP * nP); gW = [None] * len(Ws); gb = [None] * len(Ws)
    for l in range(len(Ws) - 1, -1, -1):
        dz = D * ((1 - A[l + 1] ** 2) if acts[l] == 'tanh' else 1)
        gW[l] = dz.T @ A[l] + wd * Ws[l]; gb[l] = dz.sum(0); D = dz @ Ws[l]
    for l in range(len(Ws)):
        for P, G, m, v in ((Ws, gW, mW, vW), (bs, gb, mb, vb)):
            m[l] = b1 * m[l] + (1 - b1) * G[l]; v[l] = b2 * v[l] + (1 - b2) * G[l] ** 2
            P[l] = P[l] - lr * (m[l] / (1 - b1 ** t)) / (np.sqrt(v[l] / (1 - b2 ** t)) + eps)
chk('Autoencoder gradients (back-propagation) agree with finite differences (largest relative difference)', nn['gradRelErr'], 0.0, 1e-5)
chk('Autoencoder training, 30 Adam steps with weight decay and missing values: losses (vs independent NumPy implementation)', nn['losses'], lossesN, 1e-12)
chk('Autoencoder training: weights after 30 steps (vs NumPy)', np.concatenate([np.ravel(L['W']) for L in nn['final']] + [np.ravel(L['b']) for L in nn['final']]), np.concatenate([W.ravel() for W in Ws] + [b.ravel() for b in bs]), 1e-11)
ev = PCA().fit(X).explained_variance_ * (n - 1) / n; Zs = (X - X.mean(0)) / X.std(0); evz = PCA().fit(Zs).explained_variance_ * (n - 1) / n
chk('Linear autoencoder reaches the PCA reconstruction error (bottleneck 1 and 2; Baldi & Hornik 1989)', nn['linLoss'], [evz[1:].sum() / 3, evz[2:].sum() / 3], 1e-5)
# R: Gower (daisy), PAM
rs = shutil.which('Rscript')
if rs:
    subprocess.run([rs, os.path.join(here, 'reference_check.R'), os.path.join(here, 'ref_data')], check=True, capture_output=True)
    import csv
    R = {}
    for row in csv.DictReader(open(os.path.join(here, 'ref_data', 'r_results.csv'))):
        R.setdefault(row['name'], []).append(float(row['value']))
    chk('Gower dissimilarity, mixed data with missing values (vs R cluster::daisy)', np.array(o['gower']['firstRows']).ravel(), R['gower_rows'], 1e-12)
    chk('Balanced Gower (variables weighted to contribute equally) (vs R cluster::daisy with the same weights)', np.array(o['gowerBalanced']['firstRows']).ravel(), R['gower_balanced_rows'], 1e-12)
    chk('PAM on Gower dissimilarities: objective (vs R cluster::pam)', o['pamGower']['objective'], R['pam_gower_objective'], 1e-12)
    chk('PAM on Gower dissimilarities: medoids', o['pamGower']['medoids'], R['pam_gower_medoids'], 0)
    chk('Silhouette on Gower dissimilarities (vs R cluster::pam)', o['pamGower']['silhouette'], R['pam_gower_silhouette'], 1e-12)
    chk('PAM on Euclidean data: BUILD and SWAP objectives', [o['pamEuclid']['buildObjective'], o['pamEuclid']['objective']], R['pam_euclid_objective'], 1e-12)
    chk('PAM on Euclidean data: medoids', o['pamEuclid']['medoids'], R['pam_euclid_medoids'], 0)
    if 'lca_items_loglik' in R:
        chk('Latent class analysis, categorical items with missing values: log-likelihood (vs R poLCA)', o['lcaItems']['logLik'], R['lca_items_loglik'], 1e-6)
        chk('Latent class analysis: number of parameters and BIC (vs R poLCA)', [o['lcaItems']['nParams'], o['lcaItems']['bic']], [R['lca_items_npar'][0], R['lca_items_bic'][0]], 1e-5)
        chk('Latent class analysis: class shares (vs R poLCA)', o['lcaItems']['shares'], R['lca_items_shares'], 1e-5)
    else:
        print('R package poLCA not installed: latent class check skipped')
    if 'lca_cont_loglik' in R:
        chk('Latent profile analysis (numbers): log-likelihood and BIC (vs R mclust, model VVI)', [o['lcaCont']['logLik'], o['lcaCont']['bic']], [R['lca_cont_loglik'][0], R['lca_cont_bic'][0]], 1e-5)
        chk('Latent profile analysis: class shares (vs R mclust)', o['lcaCont']['shares'], R['lca_cont_shares'], 1e-6)
    else:
        print('R package mclust not installed: latent profile check skipped')
    if 'famd_eig' in R:
        fm = o['famd']; d5 = len(R['famd_eig'])
        chk('FAMD (mixed data): eigenvalues (vs R FactoMineR)', fm['eig'][:d5], R['famd_eig'], 1e-10)
        chk("FAMD: patients' coordinates on 5 dimensions (up to sign)", np.abs(np.array(fm['scores'])).ravel(), R['famd_ind'], 1e-9)
        chk('FAMD: squared correlations / correlation ratios of each variable with each dimension', np.array(fm['varCoord']).ravel(), R['famd_var_coord'], 1e-9)
        chk('FAMD: contributions of the variables (%)', np.array(fm['contrib']).ravel(), R['famd_var_contrib'], 1e-7)
        chk('FAMD: correlations of the numbers with the dimensions (up to sign)', np.abs(np.array(fm['corr'])).ravel(), R['famd_quanti_corr'], 1e-9)
        nm = len(R['mca_eig'])
        chk('FAMD with categories only = MCA (eigenvalues x number of variables) (vs FactoMineR MCA)', np.array(fm['eigCat'][:nm]) / 3, R['mca_eig'], 1e-10)
        chk('FAMD with numbers only = PCA: eigenvalues and coordinates (vs FactoMineR PCA)', np.r_[fm['eigNum'], np.abs(np.array(fm['scoresNum'])).ravel()], np.r_[R['pca_eig'], R['pca_ind']], 1e-9)
    else:
        print('R package FactoMineR not installed: FAMD checks skipped')
else:
    print('Rscript not found: Gower and PAM checks against R skipped')
w = max(len(r[1]) for r in res)
for st, nm, d, t in res: print(f'{st}  {nm:<{w}}  max|diff|={d:.2e}  (tol {t:.0e})')
print(f"\n{sum(1 for r in res if r[0] == 'PASS')}/{len(res)} checks passed")
sys.exit(0 if all(r[0] == 'PASS' for r in res) else 1)
