# Validation protocol for real cohorts

A template for studies that test whether sub-groups found with the Unsupervised Medical Explorer are real and
clinically meaningful. Adapt it to the study; write the parts marked "before" down before the data are analysed
(for example in a protocol registered or held by a third party).

## 1. Before the analysis

- **Purpose:** exploratory, confirmatory, data quality, dimension reduction or groundwork for a prediction model
  (as chosen in step 4). A confirmatory study states the expected number and character of the sub-groups.
- **Clinical question**, in one sentence.
- **Cohorts:** a derivation cohort and at least one independent validation cohort (another centre, another
  period, or a random split held back and not opened until the derivation analysis is final). State inclusion and
  exclusion criteria, the source of the data and the governance approval.
- **Clustering variables:** the variables that define the sub-groups, with their type (number, grade, category)
  and a reason for each. Avoid pairs that measure the same thing (the app warns at a correlation of 0.8 or more)
  and keep at least 10 patients per coded variable.
- **Describe-only variables and outcomes:** the outcomes to be compared between sub-groups, and which of them is
  primary. Demographic variables (age, sex) are usually describe-only.
- **Missing values:** the planned approach (multiple imputation, recorded values only, filling, complete cases)
  and the expected share missing. Codes for "not recorded" (for example -999) must be blank cells.
- **Method and settings:** method (PAM / Gower, latent classes, k-means, self-organising map), range of cluster
  numbers (2 to 10), Gower weighting, random seed. The rule for choosing the number of clusters (the app's rule:
  every cluster's mean bootstrap Jaccard at least 0.75, then the lowest BIC or the smallest number within 0.02 of
  the best silhouette).
- **Criteria for "real sub-groups"**, stated in advance, for example: stable (every cluster's mean Jaccard at least
  0.75 with 100 resamples); the test of no real sub-groups significant at 0.05; the sub-groups persist in the
  sensitivity analysis (adjusted Rand at least 0.6 for most alternatives); reproduced in the validation cohort
  (below).
- **Sample size:** at least 100 patients per expected sub-group is a reasonable minimum for stable clusters with a
  handful of variables; the neural network (self-organising map) needs at least 1,000 records.

## 2. Derivation cohort

1. Load the data; check the warnings before the analysis; set the variable roles as in the protocol.
2. Run the data checks; correct errors at source where possible; record every exclusion and the reason (the
   analysis record lists them).
3. Choose the purpose and method as in the protocol, compare 2 to 10 clusters and record the suggested number.
   If a different number is chosen, give the clinical reason.
4. Describe the clusters: report the verdict, each cluster's stability with its 95% interval and share range,
   patient-level stability, the test of no real sub-groups, the silhouette plot and the profile table with p, q and
   effect sizes for the describe-only variables.
5. Run the sensitivity analysis and report the agreement for each alternative and the share of patients robust to
   all of them.
6. Confirm with a second method (for example latent classes after PAM) and report the agreement.
7. In step 7, name the classes, record the bedside rules and their cross-validated agreement with the clusters,
   and download the classification sheet.
8. Download the analysis report (.txt) and record (.json), and the patients with their cluster (.csv). Keep them
   with the protocol.

## 3. Validation cohort

Two complementary checks:

- **Transportability of the classification:** in step 7, section 5, load the validation cohort with the same
  column names. Report the class shares in each cohort, agreement between the bedside rules and the full model,
  records outside the original range or with unseen categories, and the pre-specified outcomes by class (p and q).
  The primary outcome differing between classes as expected is the strongest evidence of clinical meaning.
- **Reproducibility of the structure:** load the validation cohort in step 1 and repeat the analysis with the same
  settings without looking at the derivation result first. Report whether the same number of clusters is
  suggested, whether the clusters are stable and pass the test, and how closely the cluster profiles match those
  of the derivation cohort (for example the class of each validation cluster's medoid by the derivation rules,
  and the agreement between the validation clustering and the derivation classification, adjusted Rand).

## 4. Interpretation

- Sub-groups that are stable, pass the test, persist in the sensitivity analysis, reappear in the validation cohort
  and differ in pre-specified outcomes can be proposed as a classification for further study.
- A stable result with a non-significant test for mixed data is "mixed" evidence: the copula test is
  conservative; report both.
- A smooth continuum (for example severity) can produce stable clusters and a significant test; check the maps
  and the profile table for a gradient rather than distinct groups, and say so if that is what they show.
- Clusters that depend on one variable, one method or one way of handling missing values are described as such.
- A class assignment for an individual patient is an approximation, not a diagnosis.

## 5. Review and reporting

- Independent review of the protocol and the analysis by a statistician.
- Report: the protocol, any deviations and why, the analysis records of both cohorts, all candidates compared
  (not only the chosen solution), the sensitivity analysis, the validation results, and the software version
  (in the analysis record).
- The tool is for research and education and is not a medical device; a classification intended for patient care
  needs its own clinical validation and regulatory assessment.
