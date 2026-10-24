# Unsupervised Medical Explorer

Finding structure in clinical data without an outcome: patient phenotypes (clustering), maps of the data
(PCA, t-SNE, UMAP), unusual patients and data errors (anomaly detection), and autoencoders, with the same
verification standard as the Universal Medical ML/NeuralNet Predictor. Runs entirely in the browser; patient
data never leave the computer.

**Status:** the five planned stages are complete and verified (statistical engine, phenotype discovery, maps,
data checks and unusual patients, autoencoders). Next: testing on real cohorts and review by a statistician.

The interface (`index.html`) works end to end in seven steps: load a CSV (or a demo cohort); choose the variables to
cluster on and those only to describe; optionally check the data for likely entry errors and unusual patients, and
leave patients out; choose a method (an advisor suggests one for the data); compare 2 to 10 clusters (silhouette,
stability, gap statistic or BIC); describe the chosen clusters with a verdict on whether they are more than chance,
a silhouette plot, maps of the patients (PCA or FAMD, t-SNE, UMAP, autoencoder), the dimensions of the data and a
profile table; and use the clusters as a classification for clinicians: draft names and descriptions of each class
(editable), short bedside rules from a classification tree with cross-validated agreement with the clusters, a
calculator for a new patient, outcomes by class, what is needed before clinical use, and a downloadable
classification sheet.

After changing a script or `style.css`, raise its `?v=` number in `index.html` so browsers do not keep the old copy.

Open `index.html` directly, or serve the folder (`python3 -m http.server`). Two demo cohorts (`tools/make_demo_cohorts.py`):
simulated iNPH patients with three sub-types (the true sub-type is included as a describe-only column, to see how
well clustering recovers it) and a cohort of one population, where the app should find nothing.

## Statistical engine

- **Clustering:** k-means (k-means++ starts), k-medoids (PAM, BUILD and SWAP) on any dissimilarity, hierarchical
  clustering (Ward, average, complete, single; nearest-neighbour chain), Gaussian mixtures (full or diagonal
  covariance, EM from k-means starts, BIC), and latent class analysis for mixed data (normal numbers, categorical
  grades and categories, local independence; missing values left out of each patient's likelihood; EM from
  several starts; BIC, relative entropy, membership probabilities; checked against R poLCA and mclust).
- **Method advisor:** step 3 suggests a method from the mix of variable types, the share of missing values and
  the number of patients, preselects it, and recommends confirming the sub-groups with a second method.
- **Dissimilarities:** Euclidean; Gower for mixed clinical data (numbers, ordered grades, categories) with
  missing values, by default with balanced weights (each variable weighted by 1 / its mean contribution, so that
  every variable contributes equally on average; Hennig & Liao 2013). Classic equal-weight Gower lets yes/no
  variables dominate: on the one-population demo it gives a "perfectly stable" 4-cluster artefact (the four
  combinations of two yes/no variables), which balanced weighting removes.
- **Default variable roles:** outcomes and demographic columns (age, sex, gender, ethnicity) start as "describe
  only", so that clusters reflect clinical features; users can change this.
- **How many clusters, and are they real?** Silhouette, Calinski–Harabasz, Davies–Bouldin, gap statistic
  (Tibshirani's 1-SE rule), BIC; bootstrap stability of each cluster (Hennig's clusterboot: mean Jaccard); a
  test of "no real clusters" against Gaussian reference data with exactly the data's covariance (exact for
  Gaussian data; the usual version, which draws from the sample covariance, is badly conservative), and for mixed
  data a Gaussian-copula reference matched to each variable's distribution (valid but conservative).
- **Missing values:** median/most frequent value, complete cases, pairwise Gower, or multiple imputation (MICE:
  5 datasets by chained equations with predictive mean matching, from the clustering variables only; each dataset
  is clustered and combined into a consensus partition; agreement across imputations reported; stability resamples
  cycle through the imputed datasets; the test of no real sub-groups is combined by the median p-value).
- **Using the clusters:** CART classification tree (Gini) on the clustering variables, missing values sent down the
  larger branch; depth chosen by stratified 10-fold cross-validation; Cohen's kappa; new patients assigned to the
  nearest medoid (Gower, with the reference data's ranges) or nearest centroid (standardised variables).
- **Suggested number of clusters:** only stable solutions count (every cluster's mean bootstrap Jaccard at least
  0.75); among them, for model-based methods (latent classes, mixtures) the one with the lowest BIC, otherwise the
  smallest whose average silhouette is within 0.02 of the best. Low entropy is flagged for latent classes. The other stable choices are
  listed, because average silhouette tends to favour 2 clusters.
- **PCA** with variance explained.
- **Data checks and unusual patients** (`anomaly-engine.js`): entry errors column by column (values far from the
  others or isolated from them, likely decimal-point or unit errors, stray minus signs, text or decimal commas in
  number columns, spelling variants and one-off categories, duplicate rows when a chance match is implausible,
  duplicate identifiers); skewed positive measurements are judged on the log scale. Unusual patients by three
  methods on the clustering variables: robust Mahalanobis distance from the minimum covariance determinant
  (FAST-MCD with reweighting, as scikit-learn; flagged above the 99.9% chi-square limit), isolation forest, and
  local outlier factor on balanced Gower dissimilarities; each flag comes with a reason (an extreme value, a rare
  category, or the most unusual pair of values). A demo cohort has 11 planted problems.
- **Autoencoder** (`nn-engine.js`, written directly so the page keeps a strict security policy): fully connected
  network (default 16-2-16, tanh, linear bottleneck and output), mean squared error over recorded values only,
  Adam with L2 weight decay, mini-batches, early stopping on 20% held-back patients; numbers standardised and
  categories coded as in FAMD. Verified: gradients against finite differences, Adam training steps against an
  independent NumPy implementation, and a linear autoencoder reaching exactly the PCA reconstruction error.
  Used for a map (with its training curve and share of variation reconstructed) and, optionally, for unusual
  patients (reconstruction error, with the variables rebuilt worst).
- **Dimensions of the data:** FAMD (factor analysis of mixed data, as R's FactoMineR; PCA for numbers only, MCA
  for categories only): variance explained by each dimension (scree chart), the contribution and correlation of
  each variable, a plain-language description of the first three dimensions, and how strongly each dimension
  separates the clusters (correlation ratio). FAMD is also the base map for mixed data.
- **Maps of the patients** (`embed-engine.js`), from the same dissimilarities as the clustering (so mixed data and
  missing values work): classical scaling, t-SNE (exact method, as scikit-learn: perplexity 30, early
  exaggeration, adaptive gains) and UMAP (as umap-learn: fuzzy neighbour graph, curve parameters fitted for
  min_dist, SGD with negative sampling), with trustworthiness to show how faithful each map is. Patients can be
  coloured by cluster or by any variable. Each map carries a caution: t-SNE and UMAP distort sizes and distances
  and can show apparent groups in data without sub-groups (as on the one-population demo). t-SNE maps at most
  2000 patients and UMAP 5000 (a random sample beyond that).

## Verification

```bash
node tests/reference_check.js && python3 tests/reference_check.py   # 80 checks against scikit-learn, SciPy, NumPy, umap-learn and R (poLCA, mclust, cluster, FactoMineR) (cluster::daisy, cluster::pam)
node tests/behaviour_check.js                                       # simulations: true k found, 1 cluster when there are none, test calibration, stability
node tests/ui_check.js                                              # browser (Playwright + Chrome): both demo cohorts end to end (incl. MICE and step 6), layout, no page errors
npx eslint@9.39.5 -c eslint.config.mjs src cluster-engine.js embed-engine.js anomaly-engine.js nn-engine.js  # undefined names and real mistakes
```

## Author

Conceived, designed and directed by Dr G Narenthiran. Implemented with AI pair programming (Anthropic Claude Code).
© 2026 Dr G Narenthiran. All rights reserved. For research and education only; not a medical device.
