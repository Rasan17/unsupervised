# Unsupervised Medical Explorer

Finding structure in clinical data without an outcome: patient phenotypes (clustering), maps of the data
(PCA, t-SNE, UMAP), unusual patients and data errors (anomaly detection), and autoencoders, with the same
verification standard as the Universal Medical ML/NeuralNet Predictor. Runs entirely in the browser; patient
data never leave the computer.

**Status:** in development. The statistical engine (`cluster-engine.js`) is complete for clustering and PCA and
verified. The phenotype-discovery interface (`index.html`) works end to end: load a CSV (or a demo cohort), choose
the variables to cluster on and those only to describe, choose a method, compare 2 to 10 clusters (silhouette,
stability, gap statistic or BIC), then describe the chosen clusters with a verdict on whether they are more than
chance, a silhouette plot, a PCA map and a profile table. Next: t-SNE/UMAP maps, anomaly detection, autoencoders.

After changing a script or `style.css`, raise its `?v=` number in `index.html` so browsers do not keep the old copy.

Open `index.html` directly, or serve the folder (`python3 -m http.server`). Two demo cohorts (`tools/make_demo_cohorts.py`):
simulated iNPH patients with three sub-types (the true sub-type is included as a describe-only column, to see how
well clustering recovers it) and a cohort of one population, where the app should find nothing.

## Statistical engine

- **Clustering:** k-means (k-means++ starts), k-medoids (PAM, BUILD and SWAP) on any dissimilarity, hierarchical
  clustering (Ward, average, complete, single; nearest-neighbour chain), Gaussian mixtures (full or diagonal
  covariance, EM from k-means starts, BIC).
- **Dissimilarities:** Euclidean; Gower for mixed clinical data (numbers, ordered grades, categories) with
  missing values.
- **How many clusters, and are they real?** Silhouette, Calinski–Harabasz, Davies–Bouldin, gap statistic
  (Tibshirani's 1-SE rule), BIC; bootstrap stability of each cluster (Hennig's clusterboot: mean Jaccard); a
  test of "no real clusters" against Gaussian reference data with exactly the data's covariance (exact for
  Gaussian data; the usual version, which draws from the sample covariance, is badly conservative), and for mixed
  data a Gaussian-copula reference matched to each variable's distribution (valid but conservative).
- **Suggested number of clusters:** only stable solutions count (every cluster's mean bootstrap Jaccard at least
  0.75); among them the smallest whose average silhouette is within 0.02 of the best. The other stable choices are
  listed, because average silhouette tends to favour 2 clusters.
- **PCA** with variance explained.

## Verification

```bash
node tests/reference_check.js && python3 tests/reference_check.py   # 39 checks against scikit-learn, SciPy and R (cluster::daisy, cluster::pam)
node tests/behaviour_check.js                                       # simulations: true k found, 1 cluster when there are none, test calibration, stability
node tests/ui_check.js                                              # browser (Playwright + Chrome): both demo cohorts end to end, no page errors
```

## Author

Conceived, designed and directed by Dr G Narenthiran. Implemented with AI pair programming (Anthropic Claude Code).
© 2026 Dr G Narenthiran. All rights reserved. For research and education only; not a medical device.
