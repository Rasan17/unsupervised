# Unsupervised Medical Explorer

Finding structure in clinical data without an outcome: patient phenotypes (clustering), maps of the data
(PCA, t-SNE, UMAP), unusual patients and data errors (anomaly detection), and autoencoders, with the same
verification standard as the Universal Medical ML/NeuralNet Predictor. Runs entirely in the browser; patient
data never leave the computer.

**Status:** in development. The statistical engine (`cluster-engine.js`) is complete for clustering and PCA and
verified; the interface is being built.

## Statistical engine

- **Clustering:** k-means (k-means++ starts), k-medoids (PAM, BUILD and SWAP) on any dissimilarity, hierarchical
  clustering (Ward, average, complete, single; nearest-neighbour chain), Gaussian mixtures (full or diagonal
  covariance, EM from k-means starts, BIC).
- **Dissimilarities:** Euclidean; Gower for mixed clinical data (numbers, ordered grades, categories) with
  missing values.
- **How many clusters, and are they real?** Silhouette, Calinski–Harabasz, Davies–Bouldin, gap statistic
  (Tibshirani's 1-SE rule), BIC; bootstrap stability of each cluster (Hennig's clusterboot: mean Jaccard); a
  test of "no real clusters" against Gaussian reference data with exactly the data's covariance (exact for
  Gaussian data; the usual version, which draws from the sample covariance, is badly conservative).
- **PCA** with variance explained.

## Verification

```bash
node tests/reference_check.js && python3 tests/reference_check.py   # 34 checks against scikit-learn, SciPy and R (cluster::daisy, cluster::pam)
node tests/behaviour_check.js                                       # simulations: true k found, 1 cluster when there are none, test calibration, stability
```

## Author

Conceived, designed and directed by Dr G Narenthiran. Implemented with AI pair programming (Anthropic Claude Code).
© 2026 Dr G Narenthiran. All rights reserved. For research and education only; not a medical device.
