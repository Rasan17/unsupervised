# Unsupervised Medical Explorer

Finding structure in clinical data without an outcome: patient phenotypes (clustering), maps of the data
(PCA, t-SNE, UMAP), unusual patients and data errors (anomaly detection), and autoencoders, with the same
verification standard as the Universal Medical ML/NeuralNet Predictor. Runs entirely in the browser; patient
data never leave the computer.

**Status:** the five planned stages are complete and verified (statistical engine, phenotype discovery, maps,
data checks and unusual patients, autoencoders). Next: testing on real cohorts and review by a statistician.

The interface (`index.html`) works end to end in eight steps: load a CSV (or a demo cohort); choose the variables to
cluster on and those only to describe; optionally check the data for likely entry errors and unusual patients, and
leave patients out; choose a method (an advisor suggests one for the data); compare 2 to 10 clusters (silhouette,
stability, gap statistic or BIC); describe the chosen clusters with a verdict on whether they are more than chance,
a silhouette plot, maps of the patients (PCA or FAMD, t-SNE, UMAP, autoencoder), the dimensions of the data and a
profile table; and use the clusters as a classification for clinicians: draft names and descriptions of each class
(editable), short bedside rules from a classification tree with cross-validated agreement with the clusters, a
calculator for a new patient, outcomes by class, what is needed before clinical use, and a downloadable
classification sheet; and see the decision tree behind the bedside rules as a diagram (questions with yes/no
branches, patients and class mix in every box, end points coloured by class), downloadable. Every chart and diagram can be downloaded as an image (SVG for papers, PNG at double resolution
for slides), with its legend, in light colours on white.

After changing a script or `style.css`, raise its `?v=` number in `index.html` so browsers do not keep the old copy.

Open `index.html` directly, or serve the folder (`python3 -m http.server`; the demo buttons need it). Four demo
datasets: simulated iNPH patients with three sub-types (the true sub-type is included as a describe-only column, to
see how well clustering recovers it), the same cohort with 11 planted data problems, a cohort of one population
where the app should find nothing (all from `tools/make_demo_cohorts.py`), and 8,000 simulated referrals with five
referral patterns (`tools/make_referral_demo.py`).

## Statistical engine

- **Clustering:** k-means (k-means++ starts), k-medoids (PAM, BUILD and SWAP) on any dissimilarity, hierarchical
  clustering (Ward, average, complete, single; nearest-neighbour chain), Gaussian mixtures (full or diagonal
  covariance, EM from k-means and random starts, best kept, BIC), and latent class analysis for mixed data (normal numbers, categorical
  grades and categories, local independence; missing values left out of each patient's likelihood; EM from
  several starts; BIC, relative entropy, membership probabilities; checked against R poLCA and mclust).
- **Neural network for large datasets: self-organising map** (`nn-engine.js`): batch SOM (Kohonen; grid of about
  5 sqrt(n) units, Gaussian neighbourhood, linear initialisation on the principal components, missing values left
  out of the distances), whose units are grouped by k-means weighted by the records they hold (Vesanto & Alhoniemi
  2000). Not offered below 1,000 records; suggested from 5,000 (the advisor states this for every dataset).
  Verified: with no neighbourhood the SOM equals k-means (vs scikit-learn), weighted k-means vs scikit-learn with
  sample weights; on the 8,000-referral demo it recovers the five patterns as well as k-means (adjusted Rand 0.80).
  The map in step 6 is interactive: drag a rectangle (or click units, Shift / Cmd / Ctrl to add, or select a
  whole cluster) and a table lists only the records on the selected units, with how they differ from all
  records, sorting, paging and a CSV download; the same records are highlighted on the map of the patients. Conversely, on the map of the patients, click a point (its unit), drag a rectangle, or hold Option (Alt) and drag with either button for a freehand outline: exactly those records are listed, and their units are outlined on the self-organising map (dashed where only some of a unit's records are selected). On either map, Shift adds and Cmd (Ctrl) removes;
  "Clear selection", Escape or a click on an empty area deselects.
  An autoencoder followed by k-means, and IDEC deep clustering, were tried first and recovered the patterns poorly
  (about 0.45), in line with the literature on deep clustering of tabular data, so they are not used.
- **Large datasets:** above 3,000 records, silhouettes and maps use a random sample of 2,000; PAM and Ward (which
  need every pairwise dissimilarity) are not offered; multiple-imputation consensus is by aligned majority vote;
  local outlier factor is computed record by record. Skewed non-negative measurements (waiting times, distances,
  counts) are judged on the log(1 + x) scale in the data checks.
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
  separates the clusters (correlation ratio). FAMD is also the base map for mixed data; under the PCA / FAMD map,
  "What the axes mean" lists the three variables contributing most to each axis, with their contribution (%) and
  direction.
- **Maps of the patients** (`embed-engine.js`), from the same dissimilarities as the clustering (so mixed data and
  missing values work): classical scaling, t-SNE (exact method, as scikit-learn: perplexity 30, early
  exaggeration, adaptive gains) and UMAP (as umap-learn: fuzzy neighbour graph, curve parameters fitted for
  min_dist, SGD with negative sampling), with trustworthiness to show how faithful each map is. Patients can be
  coloured by cluster or by any variable. Each map carries a caution: t-SNE and UMAP distort sizes and distances
  and can show apparent groups in data without sub-groups (as on the one-population demo). t-SNE maps at most
  2000 patients and UMAP 5000 (a random sample beyond that).

## Uncertainty, sensitivity and reporting

- **Uncertainty of stability:** for each cluster, the mean bootstrap Jaccard with a 95% interval of that mean,
  the range of its share of patients across resamples, and how often it dissolved; candidates whose interval
  includes 0.75 are labelled "borderline". Each patient's assignment stability (share of resamples, among those
  that include the patient, in which they stay in their cluster) is summarised and included in the download. The
  0.75 and 0.5 limits are conventions (Hennig 2007), stated as such.
- **Multiple testing and effect sizes:** comparisons of describe-only variables between clusters give p, the
  Benjamini–Hochberg q (false discovery rate) and an effect size (Cramér's V for categories, epsilon squared for
  numbers and grades). These comparisons are labelled exploratory unless specified in advance.
- **Choice of the test of "no real sub-groups":** the Gaussian test is used only when no clustering variable is
  clearly skewed; otherwise the copula test (simulations below). A stable solution with a non-significant
  copula test is reported as "mixed" (the copula test is conservative), and every verdict warns that a curved
  single population (a continuum) can also pass both tests. The copula test reports how closely its reference
  data reproduce the observed correlations (largest difference; a caution above 0.05).
- **Warnings before the analysis:** empty or constant columns, mostly missing columns, dates, likely codes for
  "not recorded" (-9, 99, 999, -999 and similar, outside the range of the other values), strongly related
  clustering variables (rank correlation, Cramér's V or correlation ratio of 0.8 or more: one construct counted
  twice) and fewer than 10 patients per coded variable. Flags from the data checks that are known to be genuine (for
  example negative values of a change score) can be marked so for a whole variable; the analysis record notes it.
- **Sensitivity analysis** (step 6, on request): the chosen number of clusters is refitted leaving out each
  clustering variable in turn, with each other suitable method, each other way of handling missing values and the
  other Gower weighting; each alternative is compared with the main solution (adjusted Rand, share of patients in
  the same cluster after matching clusters), and each patient's robustness to these choices is reported and
  downloadable.
- **Purpose and question:** step 4 records the purpose (exploratory, confirmatory, data quality, dimension
  reduction, groundwork for a prediction model) and the clinical question; confirmatory and groundwork purposes
  add the corresponding cautions.
- **Analysis record:** step 6 downloads a report (.txt) and a machine-readable record (.json): software version,
  purpose and question, data (file name, records, exclusions after the data checks), variables and their roles,
  warnings, every setting and seed, all candidates compared, the rule that suggested the number of clusters, the
  chosen solution with its stability and test, the sensitivity analysis and the classification (rules, agreement,
  and any external application).
- **A new cohort** (step 7, section 5): a second file is classified by the bedside rules and the full model without
  retraining; the page reports the class shares against the original data, agreement between rules and model,
  records outside the original range or with unseen categories, and outcomes by class with p and q, and the
  classified records can be downloaded. `demo_inph_validation.csv` (200 simulated patients from the same three
  sub-types, in other proportions) demonstrates it.
- **Status shown in the page:** a small line under the step tabs says whether the data are a simulated demonstration or the
  user's own file (also in the analysis record); step 1, the report and the record state that the tool is for
  research and education, not a medical device, not for diagnosis or decisions about individual patients, and not
  yet validated on real cohorts or reviewed independently. Warnings about the data are repeated in step 4, before
  the analysis; on the self-organising map, grey (empty) units between clusters suggest a gap, and clusters that
  touch without them may be parts of a continuum.
- **Imputed values:** with multiple imputation or filling, the clusters table shows each cluster's share of
  patients with imputed clustering values (with methods that use recorded values only, the share with missing
  values), the maps can highlight those patients, the self-organising map's selection reports their share, and the
  download marks them. A cluster made up largely of such patients may be shaped by the imputation. The download
  holds the values as recorded, not as filled in.
- **Repeat it in R:** step 6 downloads an R script (no patient data: file name, variable names and levels,
  settings, Gower weights and row numbers of records left out) that repeats the chosen clustering with
  established packages (cluster::daisy and pam, kmeans, hclust ward.D2, VarSelLCM, poLCA, mice, kohonen) and
  compares R's clusters with the app's download by adjusted Rand index. Checked on the demos: identical clusters
  (adjusted Rand 1) for PAM / Gower, k-means, Ward and latent classes; 0.99 with multiple imputation (R imputes
  differently) and for Gaussian mixtures; 0.86 for the self-organising map (kohonen trains differently). The
  browser tests run the PAM script in R on every change.
- **Evidence summary** (step 6): stability, separation, the test against a single population, the smallest
  cluster, for latent classes how many random starts reached the best solution and the membership certainty, and
  the sensitivity analysis, each with a plain reading; listed separately, what the app cannot test (clinical
  usefulness: recognisable clusters, pre-specified outcomes, replication, impact on management).
- **Missing values by cluster:** the share of missing values of each clustering variable in each cluster, with p
  and q; when missingness differs between clusters, a caution that the clusters may partly reflect why values are
  missing, which imputation cannot correct.
- **New patients and new cohorts:** besides the class, its distance and the next class (or the membership
  probabilities for latent classes), a patient further from every class than 95% of the original patients are
  from their own is reported as poorly represented (unclassified); a new cohort reports these records and those
  near a class boundary, and its outcome table gives effect sizes.
- **Reading files:** columns written consistently with decimal commas (European exports, "67,5") are read as numbers
  (in comma-separated files only when the comma cannot be a thousands separator); "None" is a value, not a missing
  value; text grades are put in clinical order (none < mild < moderate < severe, low < medium < high), shown in step 2
  and editable there, with a caution when the order could not be recognised.
- **Describing patients:** profiles, class descriptions, the decision tree, maps coloured by a variable and the
  self-organising map's record table use the values as recorded, never values filled in for the clustering.
- **Default roles:** outcomes are recognised by name only (the shared data engine treats the last column as the
  outcome, a convention of prediction models that does not apply here); empty and constant columns start as
  "Ignore"; rare categories (fewer than 5 patients) of clustering variables are flagged.
- **Wording:** the agreement of the bedside rules is "agreement with the unsupervised cluster assignments, not
  clinical accuracy"; a new patient's class is an approximation, not a diagnosis.

## Privacy

The page has a strict content security policy (scripts only from its own files; no inline scripts, no `eval`).
It makes no network requests other than loading its own files and, when a demo button is used, the demo CSV
from the same site; a loaded file is read in the browser with `FileReader` and never uploaded. Nothing is stored
in the browser (no cookies, local or session storage, IndexedDB or caches); the data are gone when the page is
closed. Downloads (CSV, report, images) are created in the page. The browser tests check this on every change:
during a full analysis with an uploaded file and a second cohort, every request is a GET of the app's own files,
and all browser storage is empty.

## Performance

Measured on an Apple M2 Max laptop in Chrome, 8,000 simulated referrals unless stated:
data checks about 10 s; self-organising map and comparison of 2 to 10 clusters about 3.5 s; describing the chosen
clusters (100 bootstrap resamples, test of no real sub-groups) 2.5 to 13 s depending on the method; latent class
analysis: comparing 2 to 10 classes about 2 minutes and describing the chosen classes about 1 minute; maps: UMAP about 5 s, autoencoder about 10 s, t-SNE about 51 s (on the 2,000-record sample).
For 360 patients every step takes a few seconds; the sensitivity analysis refits the clustering once per alternative (16 for the iNPH demo). Above
3,000 records, silhouettes and maps use a random sample of 2,000, and PAM and Ward are not offered.

At 30,000 records (measured): data checks about 55 s, self-organising map about 15 s, describing 5 clusters about
35 s; the longest pause in any step is under 4 s.

The page stays responsive during these computations: latent class fits, bootstrap resamples and the tests of no
real sub-groups hand control back to the browser about every 100 ms and the status line counts the resamples
(the longest pause measured on the referral demo is about 2 s, down from 44 s). They are written as generators,
so the asynchronous results are identical to the direct ones (checked in the simulations). Web Workers were not
used because the page must also work when opened directly from disk, where browsers do not allow them.

## Validation status and limitations

- Verified: every computational method against an established implementation (96 reference checks) and the
  behaviour of the decisions in simulations (33 checks, including a pinned regression check of the multiple imputation), with the interface tested end to end in a browser.
- Not yet done: application to real cohorts and independent review by a statistician. A protocol for the first
  is in [`docs/validation_protocol.md`](docs/validation_protocol.md). Clusters found in one dataset are
  hypotheses until they are replicated in another.
- Known limitations: both tests of "no real sub-groups" can be passed by a curved single population (a
  continuum); the copula test for mixed data is conservative (in simulations it detects well-separated groups,
  3 SD apart, in 30 of 30 datasets but groups 2.2 SD apart in 20 of 30); the stability and silhouette limits are
  conventions. In a simulation of 20% missing values (8 datasets of 240 patients, three groups), the
  multiple-imputation consensus recovered the groups about as well as simple filling when values were missing at
  random (adjusted Rand 0.68 vs 0.69) and better when missingness depended on the value itself (0.51 vs 0.40);
  neither matched complete data (0.85), so heavy missingness weakens any clustering.

## Verification

```bash
node tests/reference_check.js && python3 tests/reference_check.py   # 96 checks against scikit-learn, SciPy, NumPy, umap-learn and R (cluster, poLCA, mclust, FactoMineR, VarSelLCM)
node tests/behaviour_check.js                                       # 33 simulations: true k found, 1 cluster when there are none, calibration and power of the tests, stability
node tests/ui_check.js                                              # browser (Playwright + Chrome): every demo end to end, the audit features, edge cases, privacy, the R script, layout, no page errors
npx eslint@9.39.5 -c eslint.config.mjs src cluster-engine.js embed-engine.js anomaly-engine.js nn-engine.js  # undefined names and real mistakes
```

GitHub Actions runs all four on every push, including the Python and R reference checks (with `REQUIRE_ALL=1`,
so a missing package fails the run instead of skipping its checks).

## Author

Conceived, designed and directed by Dr G Narenthiran. Implemented with AI pair programming (Anthropic Claude Code).
© 2026 Dr G Narenthiran. All rights reserved. For research and education only; not a medical device.
