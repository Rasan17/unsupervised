// r-script.js: an R script that repeats the chosen clustering with established R packages (cluster, mclust, poLCA,
// VarSelLCM, mice, kohonen) and compares R's clusters with the app's, patient by patient (adjusted Rand index).
// The script holds no patient data: only the file name, variable names and levels, settings, weights and the row
// numbers of records left out. It is written in the page and downloaded; nothing is sent anywhere.
(function (root) {
    'use strict';
    const q = s => `"${String(s).replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`;
    const vec = xs => `c(${xs.join(', ')})`;
    const num = v => (Number.isFinite(v) ? String(Number(v.toPrecision(15))) : 'NA');

    function build({ S, P, R, build: buildId, seed }) {
        const k = R.k, specs = P.specs, file = String(S.data.name), stem = file.replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_');
        const nums = specs.filter(s => s.type === 'continuous').map(s => s.col);
        const ords = specs.filter(s => s.type === 'ordinal'), cats = specs.filter(s => s.type === 'nominal');
        const excluded = [...(S.excluded || [])].sort((a, b) => a - b).map(i => i + 1);
        const allNum = !ords.length && !cats.length, allCat = !nums.length;
        const missing = P.missing, method = P.method;
        const completeCases = missing === 'complete' || (missing === 'gower' && !['pam-gower', 'lca', 'neural'].includes(method));
        const L = [];
        const add = (...xs) => L.push(...xs);
        const pkgs = ['cluster', 'mclust'];
        if (method === 'lca' || method === 'gmm') pkgs.push(method === 'lca' && allCat ? 'poLCA' : 'VarSelLCM');
        if (missing === 'mice') pkgs.push('mice');
        if (method === 'neural') pkgs.push('kohonen');
        add(`# Repeats in R the clustering chosen in the Unsupervised Medical Explorer, for independent checking.`,
            `# Generated ${new Date().toISOString().slice(0, 10)} by ${buildId}. © 2026 Dr G Narenthiran. All rights reserved.`,
            `# This script contains no patient data: only the file name, variable names and levels, settings, weights and the`,
            `# row numbers of records left out. For research and education only; not a medical device.`,
            `# Packages: install.packages(${vec(pkgs.map(q))})`, '',
            `# 1. Paths: the data file analysed in the app, and (optional) the "patients with their cluster" download from step 6`,
            `data_file <- ${q(file)}`, `app_file  <- ${q(`clusters_${stem}.csv`)}`, '',
            'suppressPackageStartupMessages(library(mclust))   # mclust needs to be attached for Mclust()',
            `seed <- ${seed}; k <- ${k}`, 'set.seed(seed)', '',
            `# 2. Read the data as the app does: text trimmed; empty, NA, NaN, NULL, None, ?, . and N/A are missing`,
            'd <- read.csv(data_file, check.names = FALSE, colClasses = "character", na.strings = character(0))',
            'd[] <- lapply(d, function(v) { v <- trimws(v); v[tolower(v) %in% c("", "na", "nan", "null", "none", "?", ".", "n/a")] <- NA; v })',
            'd$.row <- seq_len(nrow(d))',
            `excluded <- ${excluded.length ? vec(excluded) : 'integer(0)'}   # left out after the data checks (row numbers in the file)`,
            'd <- d[!(d$.row %in% excluded), ]', '',
            `# 3. Clustering variables and their types (levels in the app's order)`,
            `num_vars <- ${nums.length ? vec(nums.map(q)) : 'character(0)'}`,
            `ord_vars <- list(${ords.map(s => `${q(s.col)} = ${vec(s.levels.map(q))}`).join(', ')})`,
            `cat_vars <- list(${cats.map(s => `${q(s.col)} = ${vec(s.levels.map(q))}`).join(', ')})`,
            `vars <- ${vec(specs.map(s => q(s.col)))}`,
            'x <- d[vars]',
            'for (v in num_vars) x[[v]] <- suppressWarnings(as.numeric(x[[v]]))',
            'for (v in names(ord_vars)) x[[v]] <- factor(x[[v]], levels = ord_vars[[v]], ordered = TRUE)',
            'for (v in names(cat_vars)) x[[v]] <- factor(x[[v]], levels = cat_vars[[v]])',
            'rows <- d$.row', '');
        add('# 4. Missing values');
        if (completeCases) add('keep <- complete.cases(x); x <- x[keep, , drop = FALSE]; rows <- rows[keep]   # complete cases only');
        else if (missing === 'fill') {
            add('# filled with the median (numbers) or the most frequent value (grades, categories), as computed by the app');
            specs.forEach(s => { const v = P.fillVal ? P.fillVal[s.col] : undefined; if (v === undefined) return; add(s.type === 'continuous' ? `x[[${q(s.col)}]][is.na(x[[${q(s.col)}]])] <- ${num(Number(v))}` : `x[[${q(s.col)}]][is.na(x[[${q(s.col)}]])] <- ${q(v)}`); });
        } else if (missing === 'mice') add(`# multiple imputation (5 datasets, predictive mean matching for numbers); R's imputations differ from the app's,`,
            `# so agreement is expected to be close, not exact`,
            `imp <- mice::mice(x, m = 5, maxit = 10, seed = seed, printFlag = FALSE)`, 'xs <- lapply(1:5, function(i) mice::complete(imp, i))');
        else add('# missing values are kept: they are left out of each pair\'s dissimilarity (Gower) or of each patient\'s likelihood');
        if (missing !== 'mice') add('xs <- list(x)');
        add('');
        // Coding for k-means, Ward, Gaussian mixtures and the map: as the app (grades as level number, yes/no as 0/1,
        // other categories as indicators), standardised with the population standard deviation
        add(`# 5. Coding used by k-means, Ward, Gaussian mixtures and the self-organising map (as the app): numbers as they`,
            `#    are, grades as level number, two-level categories 0/1, other categories one indicator per level; standardised`,
            'code_z <- function(x) {',
            '  cols <- list()',
            '  for (v in names(x)) { y <- x[[v]]',
            '    if (v %in% num_vars) cols[[v]] <- y',
            '    else if (v %in% names(ord_vars)) cols[[v]] <- as.integer(y) - 1',
            '    else { lv <- cat_vars[[v]]; if (length(lv) == 2) cols[[paste(v, "=", lv[2])]] <- as.numeric(y == lv[2]) else for (l in lv) cols[[paste(v, "=", l)]] <- as.numeric(y == l) } }',
            '  X <- as.matrix(as.data.frame(cols, check.names = FALSE)); m <- colMeans(X)',
            '  s <- sqrt(colMeans(sweep(X, 2, m)^2)); s[!(s > 1e-12 * pmax(1, abs(m)))] <- 1',
            '  sweep(sweep(X, 2, m), 2, s, "/")',
            '}', '');
        if (method === 'gmm' || (method === 'lca' && allNum && missing !== 'gower')) add('# Gaussian mixture with diagonal covariances (mclust model VVI), fitted with VarSelLCM, which tries many starts and keeps',
            '# the best (mclust\'s single default start can stop at a weaker solution); log-likelihood printed for comparison',
            'mix_best <- function(z) { r <- VarSelLCM::VarSelCluster(as.data.frame(z), gvals = k, vbleSelec = FALSE, nbcores = 1)',
            '  cat("Log-likelihood in R:", round(r@criteria@loglikelihood, 4), "\\n"); r@partitions@zMAP }');
        add(`# 6. The clustering: ${({ 'pam-gower': 'k-medoids (PAM) on Gower dissimilarities', kmeans: 'k-means', ward: 'Ward hierarchical clustering', gmm: 'Gaussian mixture (diagonal covariance)', lca: 'latent class analysis', neural: 'self-organising map with weighted k-means on its units' })[method]}`);
        let expect = '';
        if (method === 'pam-gower') {
            const w = P.gowerWeights === 'balanced' && P.gower && P.gower.weights ? P.gower.weights : specs.map(() => 1);
            add(`w <- ${vec(w.map(num))}   # ${P.gowerWeights === 'balanced' ? 'balanced Gower weights computed by the app (each variable contributes equally on average)' : 'equal Gower weights'}`,
                'fit_one <- function(x) { D <- cluster::daisy(x, metric = "gower", weights = w); cluster::pam(D, k, diss = TRUE)$clustering }');
            expect = 'PAM and Gower are deterministic: the clusters should be identical (adjusted Rand 1)';
        } else if (method === 'kmeans') { add('fit_one <- function(x) kmeans(code_z(x), k, nstart = 25, iter.max = 100)$cluster'); expect = 'k-means depends on its random starts: expect the same or nearly the same clusters'; }
        else if (method === 'ward') { add('fit_one <- function(x) cutree(hclust(dist(code_z(x)), method = "ward.D2"), k)'); expect = 'Ward clustering is deterministic: the clusters should be identical (adjusted Rand 1), apart from exact ties'; }
        else if (method === 'gmm') { add('fit_one <- function(x) mix_best(code_z(x))'); expect = 'the mixture depends on its starting values: expect the same or nearly the same clusters'; }
        else if (method === 'lca') {
            if (allCat) add('fit_one <- function(x) { x[] <- lapply(x, function(v) as.integer(v)); f <- as.formula(paste("cbind(", paste0("`", names(x), "`", collapse = ", "), ") ~ 1"))',
                '  poLCA::poLCA(f, data = x, nclass = k, nrep = 10, na.rm = FALSE, verbose = FALSE, maxiter = 5000)$predclass }');
            else if (allNum && missing !== 'gower') add('fit_one <- function(x) mix_best(x)   # latent profile analysis');
            else add('# mixed data: normal numbers and categorical grades and categories, local independence (as the app); grades as categories',
                'fit_one <- function(x) { x[] <- lapply(x, function(v) if (is.factor(v)) factor(v, ordered = FALSE) else as.numeric(v))',
                '  r <- VarSelLCM::VarSelCluster(x, gvals = k, vbleSelec = FALSE, nbcores = 1); r@partitions@zMAP }');
            expect = 'latent classes depend on their starting values: expect the same or nearly the same classes';
        } else if (method === 'neural') {
            const [gr, gc] = String(P.neural ? P.neural.grid : '10 x 10').split(' x ').map(Number);
            add(`# the app trains a batch map of ${gr} x ${gc} units; kohonen's batch map starts and trains differently, so expect close, not identical, results`,
                'wkmeans <- function(P, w, k, nstart = 10) { best <- NULL; for (s in 1:nstart) { C <- P[sample(which(w > 0), k), , drop = FALSE]',
                '  for (it in 1:100) { cl <- max.col(-as.matrix(dist(rbind(C, P)))[-(1:k), 1:k, drop = FALSE]); C2 <- t(sapply(1:k, function(j) if (any(cl == j & w > 0)) colSums(P[cl == j, , drop = FALSE] * w[cl == j]) / sum(w[cl == j]) else C[j, ])); if (max(abs(C2 - C)) < 1e-10) break; C <- C2 }',
                '  ss <- sum(w * rowSums((P - C[cl, , drop = FALSE])^2)); if (is.null(best) || ss < best$ss) best <- list(ss = ss, cl = cl) }; best$cl }',
                `fit_one <- function(x) { Z <- code_z(x); m <- kohonen::som(Z, grid = kohonen::somgrid(${gr}, ${gc}, "rectangular"), rlen = 100, mode = "batch")`,
                '  u <- m$unit.classif; w <- tabulate(u, nrow(m$codes[[1]])); g <- wkmeans(m$codes[[1]], w, k); g[u] }');
            expect = 'the map is trained differently in R: expect broadly similar groups';
        }
        add('fits <- lapply(xs, fit_one)');
        if (missing === 'mice') add('# consensus over the imputed datasets: k-medoids on the share of datasets in which two patients are apart (as the app)',
            'A <- Reduce(`+`, lapply(fits, function(f) outer(f, f, "!="))) / length(fits)',
            'cl <- cluster::pam(as.dist(A), k, diss = TRUE)$clustering');
        else add('cl <- fits[[1]]');
        add('', `# 7. Comparison with the app (${expect})`,
            'cat("Patients:", length(cl), " Cluster sizes in R:", table(cl), "\\n")',
            'if (file.exists(app_file)) {',
            '  a <- read.csv(app_file, check.names = FALSE)',
            '  app <- a$Cluster[match(rows, a[["Row in file"]])]',
            '  cat("Agreement with the app (adjusted Rand index):", round(mclust::adjustedRandIndex(cl, app), 4), "\\n")',
            '  print(table(R = cl, App = app))',
            '}');
        if (method === 'pam-gower' && missing !== 'mice') add('', '# 8. Optional: bootstrap stability of each cluster (Hennig\'s clusterboot; mean Jaccard, as in the app; values differ',
            '#    slightly because the resamples differ). install.packages("fpc")',
            '# D <- cluster::daisy(x, metric = "gower", weights = w)',
            '# fpc::clusterboot(D, B = 100, bootmethod = "boot", clustermethod = fpc::claraCBI, k = k, usepam = TRUE, count = FALSE, seed = seed)$bootmean');
        return L.join('\n') + '\n';
    }
    root.UMERScript = { build };
})(typeof window !== 'undefined' ? window : globalThis);
