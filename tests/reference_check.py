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
else:
    print('Rscript not found: Gower and PAM checks against R skipped')
w = max(len(r[1]) for r in res)
for st, nm, d, t in res: print(f'{st}  {nm:<{w}}  max|diff|={d:.2e}  (tol {t:.0e})')
print(f"\n{sum(1 for r in res if r[0] == 'PASS')}/{len(res)} checks passed")
sys.exit(0 if all(r[0] == 'PASS' for r in res) else 1)
