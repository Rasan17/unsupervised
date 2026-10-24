# R reference values (cluster package) for tests/reference_check.py: Gower dissimilarity (daisy), PAM on Gower
# and Euclidean dissimilarities. Writes tests/ref_data/r_results.csv (name, index, value).
args <- commandArgs(trailingOnly = TRUE); d <- if (length(args)) args[1] else "tests/ref_data"
suppressPackageStartupMessages(library(cluster))
out <- data.frame(name = character(), index = integer(), value = numeric())
add <- function(name, v) out <<- rbind(out, data.frame(name = name, index = seq_along(v), value = as.numeric(v)))
m <- read.csv(file.path(d, "mixed.csv"), na.strings = "NA", stringsAsFactors = FALSE)
m$grade <- factor(m$grade, levels = c("mild", "moderate", "severe"), ordered = TRUE)
m$sex <- factor(m$sex); m$site <- factor(m$site)
G <- daisy(m, metric = "gower")
Gm <- as.matrix(G)
add("gower_rows", as.vector(t(Gm[1:5, ])))
# Balanced weights computed by the app (1 / mean contribution of each variable), applied by daisy
w <- as.numeric(strsplit(readLines(file.path(d, "gower_weights.txt")), ",")[[1]])
GB <- as.matrix(daisy(m, metric = "gower", weights = w))
add("gower_balanced_rows", as.vector(t(GB[1:5, ])))
pg <- pam(G, 3, diss = TRUE)
add("pam_gower_objective", pg$objective["swap"])
add("pam_gower_medoids", sort(pg$id.med) - 1)
add("pam_gower_silhouette", pg$silinfo$avg.width)
b <- read.csv(file.path(d, "blobs.csv"))
pe <- pam(b, 3)
add("pam_euclid_objective", pe$objective[c("build", "swap")])
add("pam_euclid_medoids", sort(pe$id.med) - 1)
# Latent class analysis: poLCA (categorical items, missing values kept) and mclust (latent profiles, model VVI)
if (requireNamespace("poLCA", quietly = TRUE)) {
  it <- read.csv(file.path(d, "lca_items.csv"), na.strings = "NA")
  set.seed(1)
  lc <- poLCA::poLCA(cbind(i1, i2, i3, i4) ~ 1, data = it, nclass = 3, nrep = 30, na.rm = FALSE, verbose = FALSE, maxiter = 100000, tol = 1e-14)
  add("lca_items_loglik", lc$llik); add("lca_items_bic", lc$bic); add("lca_items_npar", lc$npar); add("lca_items_shares", sort(lc$P))
}
if (requireNamespace("mclust", quietly = TRUE)) {
  suppressPackageStartupMessages(library(mclust))
  mc <- Mclust(b, G = 3, modelNames = "VVI", verbose = FALSE, control = emControl(tol = c(1e-14, sqrt(.Machine$double.eps)), itmax = c(100000L, 100000L)))
  add("lca_cont_loglik", mc$loglik); add("lca_cont_bic", -mc$bic); add("lca_cont_shares", sort(mc$parameters$pro))
}
write.csv(out, file.path(d, "r_results.csv"), row.names = FALSE)
