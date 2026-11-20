# ================================================================
# STUDY 1 — OBSERVED vs REPEATED SURROGATE MODEL 7
# ================================================================
# Input 1: study1_original_roi_labels_corrected.csv
#   sub, roi, task, task_stage, block, pac_value, ...
# Input 2: study1_surrogates_merged_validated.csv
#   sub, task, task_stage, block, condition, roi,
#   surrogate_id, pac_value, original_pac, ...
#
# Analysis:
#  (A) Fit original Model 7 once to observed raw MI.
#  (B) Fit same Model 7 separately for each surrogate_id (1..200).
#  (C) Compare model statistics and condition/stage contrasts with
#      their empirical surrogate distributions.
#  (D) Complementary zMI LMM as a sensitivity analysis.
#
# The surrogate distribution is a *phase/amplitude-shift* null,
# NOT a permutation of condition labels. It can preserve condition-
# dependent estimator bias or nuisance structure. Its contrasts
# need not be centred on zero, or be nonsignificant.
# ================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(lme4)
  library(car)
  library(emmeans)
})

# -------- USER SETTINGS: EDIT THESE PATHS ------------------------
OBSERVED_FILE <- "F:/research project/# study 1/eeg_data/set/Y group/roi_source_analysis/surrogate_analysis/study1_original_roi_labels_corrected.csv"
SURROGATE_FILE <- "F:/research project/# study 1/eeg_data/set/Y group/roi_source_analysis/surrogate_analysis/study1_surrogates_merged_validated.csv"
OUTPUT_DIR <- "F:/research project/# study 1/eeg_data/set/Y group/roi_source_analysis/surrogate_analysis/repeated_surrogate_model7"

# Use 1:5 for a quick smoke test; 1:200 for the final analysis.
SURROGATE_IDS <- 1:200

# Original analysis uses REML. Use identical model settings for all fits.
FIT_CONTROL <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
CONDITION_LEVELS <- c("_BL", "_MAIN_baseline", "_MAIN_adaptation")
STAGE_LEVELS <- c("_plan", "_go")
FORMULA <- pac_value ~ condition * roi * task_stage + (1 | sub)
ALPHA <- 0.01

# Be explicit about the identity of the target estimands.
# All condition/stage comparisons are averaged across the 24 ROIs,
# matching the emmeans comparisons in the submitted Model 7.
# We use *unadjusted* model contrasts to construct surrogate nulls,
# then apply Holm correction to empirical p-values within each
# family; the original Bonferroni parametric p-values are also saved.

# -------- FUNCTIONS ---------------------------------------------
stopifnot(file.exists(OBSERVED_FILE), file.exists(SURROGATE_FILE))
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

assert_cols <- function(dt, cols, name) {
  missing <- setdiff(cols, names(dt))
  if (length(missing)) stop(name, " missing columns: ", paste(missing, collapse = ", "))
}

make_factors <- function(d) {
  d <- copy(d)
  d[, condition := factor(as.character(condition), levels = CONDITION_LEVELS)]
  d[, task_stage := factor(as.character(task_stage), levels = STAGE_LEVELS)]
  d[, roi := factor(as.character(roi), levels = ROI_LEVELS)]
  d[, sub := factor(as.character(sub), levels = SUBJECT_LEVELS)]
  if (anyNA(d[, .(condition, task_stage, roi, sub)])) {
    stop("Unexpected/missing factor levels in model data")
  }
  d[]
}

fit_model <- function(d) {
  withCallingHandlers(
    lmer(FORMULA, data = d, REML = TRUE, control = FIT_CONTROL),
    warning = function(w) {
      # Keep warnings visible instead of silently suppressing them.
      message("LMM warning: ", conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
}

get_anova <- function(mod) {
  a <- as.data.frame(car::Anova(mod, type = 2))
  a$term <- rownames(a)
  data.table(term = a$term, chisq = a[["Chisq"]],
             df = a[["Df"]], parametric_p = a[["Pr(>Chisq)"]])
}

get_contrasts <- function(mod, adjusted = FALSE) {
  # Keep contrast definition constant across observed and surrogates.
  # emmeans order and signed estimates are retained.
  adj <- if (adjusted) "bonferroni" else "none"
  cond <- as.data.table(summary(
    contrast(emmeans(mod, ~ condition | task_stage), "pairwise", adjust = adj)
  ))
  cond[, `:=`(family = "condition_within_stage",
              context = as.character(task_stage),
              comparison = as.character(contrast))]
  stage <- as.data.table(summary(
    contrast(emmeans(mod, ~ task_stage | condition), "pairwise", adjust = adj)
  ))
  stage[, `:=`(family = "stage_within_condition",
               context = as.character(condition),
               comparison = as.character(contrast))]
  wanted <- c("family", "context", "comparison", "estimate", "SE", "p.value")
  rbindlist(list(cond[, ..wanted], stage[, ..wanted]))
}

extract_fit <- function(d, tag) {
  m <- fit_model(d)
  if (isSingular(m, tol = 1e-4)) message("Singular fit: ", tag)
  list(model = m, anova = get_anova(m),
       contrasts = get_contrasts(m),
       bonferroni = get_contrasts(m, adjusted = TRUE),
       singular = isSingular(m, tol = 1e-4))
}

# Empirical rank comparison: right/left tails, and a two-sided rank p.
# The two-sided p uses 2 * min(tail probabilities), not an assumed
# symmetric/zero-centred null. The observed estimate is *not*
# exchangeable with the shifted surrogates in every scientific null;
# regard these as surrogate-reference comparisons, not a universal
# randomisation test of experimental-condition equality.
compare_null <- function(observed, null) {
  null <- null[is.finite(null)]
  if (!length(null) || !is.finite(observed)) {
    return(data.table(n_null = length(null), null_mean = NA_real_,
                      null_sd = NA_real_, null_q025 = NA_real_,
                      null_q975 = NA_real_, p_left = NA_real_,
                      p_right = NA_real_, p_two_sided = NA_real_,
                      observed_minus_null_mean = NA_real_))
  }
  p_right <- (1 + sum(null >= observed)) / (length(null) + 1)
  p_left  <- (1 + sum(null <= observed)) / (length(null) + 1)
  data.table(
    n_null = length(null),
    null_mean = mean(null),
    null_sd = if (length(null) > 1) sd(null) else NA_real_,
    null_q025 = unname(quantile(null, .025)),
    null_q975 = unname(quantile(null, .975)),
    p_left = p_left, p_right = p_right,
    p_two_sided = min(1, 2 * min(p_left, p_right)),
    observed_minus_null_mean = observed - mean(null)
  )
}

# -------- LOAD AND VALIDATE -------------------------------------
cat("Loading CSVs...\n")
obs <- fread(OBSERVED_FILE)
sur <- fread(SURROGATE_FILE)
assert_cols(obs, c("sub", "roi", "task", "task_stage", "block", "pac_value"), "Observed CSV")
assert_cols(sur, c("sub", "roi", "condition", "task_stage", "surrogate_id", "pac_value", "original_pac"), "Surrogate CSV")

# Consistent condition naming for original file.
obs[, block := fifelse(is.na(block) | block == "", "", as.character(block))]
obs[, condition := paste0(as.character(task), block)]

key <- c("sub", "condition", "task_stage", "roi")
if (anyDuplicated(obs, by = key)) stop("Duplicate rows in observed CSV")
if (anyDuplicated(sur, by = c(key, "surrogate_id"))) stop("Duplicate surrogate rows")
if (anyNA(obs$pac_value) || anyNA(sur$pac_value)) stop("Missing PAC values")

# Verify same 3312 participant-condition-ROI combinations and source MI.
sur_unique <- unique(sur[, c(key, "original_pac"), with = FALSE])
if (anyDuplicated(sur_unique, by = key)) stop("Original PAC inconsistent across surrogate IDs")
validation <- merge(obs[, c(key, "pac_value"), with = FALSE],
                    sur_unique, by = key, all = TRUE)
if (nrow(validation) != nrow(obs) || nrow(validation) != nrow(sur_unique) ||
    anyNA(validation$pac_value) || anyNA(validation$original_pac)) {
  stop("Observed/surrogate row-key mismatch")
}
max_error <- max(abs(validation$pac_value - validation$original_pac))
max_rel_error <- max(abs(validation$pac_value - validation$original_pac) /
                     pmax(abs(validation$pac_value), 1e-15))
cat(sprintf("Observed/surrogate MI agreement: max abs = %.4g, max relative = %.5f%%\n",
            max_error, 100 * max_rel_error))
if (max_rel_error > 0.001) stop("Observed PAC mismatch exceeds 0.1%; investigate before modelling")

ids_available <- sort(unique(sur$surrogate_id))
if (!all(SURROGATE_IDS %in% ids_available)) stop("Requested surrogate IDs absent from CSV")
counts <- sur[surrogate_id %in% SURROGATE_IDS, .N, by = surrogate_id]
if (any(counts$N != nrow(obs)) || nrow(counts) != length(SURROGATE_IDS)) {
  stop("Surrogate iterations incomplete: expected exactly one row per observed observation per ID")
}
SUBJECT_LEVELS <- sort(unique(obs$sub))
ROI_LEVELS <- sort(unique(obs$roi))
cat(sprintf("Validated: %d observed rows, %d subjects, %d ROIs, %d complete surrogate iterations\n",
            nrow(obs), length(SUBJECT_LEVELS), length(ROI_LEVELS), length(SURROGATE_IDS)))

obs <- make_factors(obs)
sur <- make_factors(sur)
setorder(obs, sub, condition, task_stage, roi)
setorder(sur, surrogate_id, sub, condition, task_stage, roi)

# -------- OBSERVED MODEL ----------------------------------------
cat("\nFitting observed Model 7...\n")
observed_fit <- extract_fit(obs, "observed")
observed_anova <- copy(observed_fit$anova)[, iteration := 0L]
observed_contrasts <- copy(observed_fit$contrasts)[, iteration := 0L]
observed_bonf <- copy(observed_fit$bonferroni)
saveRDS(observed_fit$model, file.path(OUTPUT_DIR, "observed_model7.rds"))
fwrite(observed_anova, file.path(OUTPUT_DIR, "observed_anova.csv"))
fwrite(observed_bonf, file.path(OUTPUT_DIR, "observed_contrasts_bonferroni.csv"))

# -------- REPEATED SURROGATE MODELS ------------------------------
cat("Fitting ", length(SURROGATE_IDS), " surrogate Model 7 iterations...\n", sep = "")
anova_list <- vector("list", length(SURROGATE_IDS))
contrast_list <- vector("list", length(SURROGATE_IDS))
fit_status <- vector("list", length(SURROGATE_IDS))

for (j in seq_along(SURROGATE_IDS)) {
  id <- SURROGATE_IDS[j]
  d <- sur[surrogate_id == id, c(key, "pac_value"), with = FALSE]
  d <- make_factors(d)
  result <- tryCatch(extract_fit(d, paste0("surrogate ", id)), error = identity)
  if (inherits(result, "error")) {
    stop("Surrogate iteration ", id, " failed: ", conditionMessage(result),
         "\nStopped to avoid a misleading partial null distribution.")
  }
  anova_list[[j]] <- copy(result$anova)[, iteration := id]
  contrast_list[[j]] <- copy(result$contrasts)[, iteration := id]
  fit_status[[j]] <- data.table(iteration = id, singular = result$singular)
  if (j %% 10 == 0 || j == length(SURROGATE_IDS)) {
    cat(sprintf("  completed %d/%d\n", j, length(SURROGATE_IDS)))
  }
}

null_anova <- rbindlist(anova_list)
null_contrasts <- rbindlist(contrast_list)
status <- rbindlist(fit_status)
fwrite(null_anova, file.path(OUTPUT_DIR, "surrogate_anova_all_iterations.csv"))
fwrite(null_contrasts, file.path(OUTPUT_DIR, "surrogate_contrasts_all_iterations.csv"))
fwrite(status, file.path(OUTPUT_DIR, "surrogate_fit_status.csv"))

# -------- EMPIRICAL MODEL-STATISTIC COMPARISON --------------------
# Main effects are referenced by their nonnegative Wald chi-square.
# These p-values concern whether observed model statistics are
# unusually strong relative to time-shifted PAC, not a condition-label
# permutation test or direct proof of physiological coupling.
ANOVA_TERMS <- observed_anova$term
anova_summary <- rbindlist(lapply(ANOVA_TERMS, function(term_name) {
  observed_row <- observed_anova[term == term_name]
  comparison <- compare_null(observed_row$chisq, null_anova[term == term_name]$chisq)
  data.table(term = term_name, observed_chisq = observed_row$chisq,
             observed_parametric_p = observed_row$parametric_p,
             comparison)
}))
anova_summary[, p_empirical_upper := p_right]
anova_summary[, p_empirical_upper_holm := p.adjust(p_empirical_upper, "holm")]
fwrite(anova_summary, file.path(OUTPUT_DIR, "comparison_anova_empirical.csv"))

# -------- EMPIRICAL CONDITION/STAGE CONTRAST COMPARISON ----------
contrast_keys <- c("family", "context", "comparison")
contrast_summary <- rbindlist(lapply(seq_len(nrow(observed_contrasts)), function(i) {
  o <- observed_contrasts[i]
  n <- null_contrasts[
    family == o$family & context == o$context & comparison == o$comparison,
    estimate
  ]
  cbind(o[, .(family, context, comparison, observed_estimate = estimate,
              observed_se = SE, observed_parametric_p_unadjusted = p.value)],
        compare_null(o$estimate, n))
}))

# Adjust empirical tests across the three condition contrasts within
# each stage; and stage comparisons across the three conditions.
contrast_summary[, p_empirical_two_sided_holm := p.adjust(p_two_sided, "holm"),
                 by = .(family, context)]
# Additional global adjustment across all nine targeted contrasts.
contrast_summary[, p_empirical_two_sided_holm_all := p.adjust(p_two_sided, "holm")]
contrast_summary <- merge(
  contrast_summary,
  observed_bonf[, .(family, context, comparison, observed_parametric_p_bonferroni = p.value)],
  by = contrast_keys, all.x = TRUE, sort = FALSE
)
fwrite(contrast_summary, file.path(OUTPUT_DIR, "comparison_contrasts_empirical.csv"))

# -------- SURROGATE-NORMALIZED SENSITIVITY MODEL -----------------
# zMI for every subject x condition x ROI, based on 200 surrogates.
# This is DIFFERENT from the repeated-surrogate raw-MI LMM; it answers
# whether experimental conditions differ in their distance from null.
cat("\nFitting complementary surrogate-normalized zMI LMM...\n")
full_sur <- sur[surrogate_id %in% SURROGATE_IDS]
# Never use a 5-surrogate smoke test to fit this sensitivity model.
if (length(SURROGATE_IDS) == length(ids_available)) {
  zsummary <- full_sur[, .(surrogate_mean = mean(pac_value),
                            surrogate_sd = sd(pac_value),
                            observed_mi = first(original_pac)), by = key]
  if (any(!is.finite(zsummary$surrogate_sd) | zsummary$surrogate_sd <= 0)) {
    stop("Zero/nonfinite surrogate SD in zMI summary")
  }
  zsummary[, pac_value := (observed_mi - surrogate_mean) / surrogate_sd]
  zsummary <- make_factors(zsummary)
  z_fit <- extract_fit(zsummary, "surrogate zMI")
  fwrite(z_fit$anova, file.path(OUTPUT_DIR, "sensitivity_zmi_anova.csv"))
  fwrite(z_fit$bonferroni, file.path(OUTPUT_DIR, "sensitivity_zmi_contrasts_bonferroni.csv"))
  saveRDS(z_fit$model, file.path(OUTPUT_DIR, "sensitivity_zmi_model7.rds"))
} else {
  cat("Skipping zMI sensitivity analysis in reduced-surrogate smoke test.\n")
}

# -------- SIMPLE PDF and PNG DIAGNOSTIC PLOTS (BASE R ONLY) --------------
pdf(file.path(OUTPUT_DIR, "surrogate_null_distributions.pdf"), width = 9, height = 6)
for (i in seq_len(nrow(contrast_summary))) {
  row <- contrast_summary[i]
  vals <- null_contrasts[
    family == row$family & context == row$context & comparison == row$comparison,
    estimate
  ]
  hist(vals, breaks = 25, main = paste(row$family, row$context, "\n", row$comparison),
       xlab = "Condition/stage contrast in raw MI (surrogate model)")
  abline(v = row$observed_estimate, col = "red", lwd = 2)
  abline(v = mean(vals), col = "black", lty = 2, lwd = 2)
  legend("topright", c("Observed contrast", "Surrogate mean"),
         col = c("red", "black"), lty = c(1, 2), bty = "n")
}
dev.off()

plot_dir <- file.path(OUTPUT_DIR, "surrogate_null_distributions")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

for (i in seq_len(nrow(contrast_summary))) {
  
  row <- contrast_summary[i]
  
  vals <- null_contrasts[
    family == row$family &
      context == row$context &
      comparison == row$comparison,
    estimate
  ]
  
  # Create a safe filename
  plot_name <- paste(
    row$family,
    row$context,
    row$comparison,
    sep = "_"
  )
  
  plot_name <- gsub("[^[:alnum:]_-]", "_", plot_name)
  
  # Open PNG device
  png(
    filename = file.path(plot_dir, paste0(plot_name, ".png")),
    width = 2700,
    height = 1800,
    res = 300
  )
  
  hist(
    vals,
    breaks = 25,
    main = paste(row$family, row$context, "\n", row$comparison),
    xlab = "Condition/stage contrast in raw MI (surrogate model)",
    col = "lightgrey",
    border = "white"
  )
  
  abline(
    v = row$observed_estimate,
    col = "red",
    lwd = 2
  )
  
  abline(
    v = mean(vals),
    col = "black",
    lty = 2,
    lwd = 2
  )
  
  legend(
    "topright",
    legend = c("Observed contrast", "Surrogate mean"),
    col = c("red", "black"),
    lty = c(1, 2),
    lwd = 2,
    bty = "n"
  )
  
  dev.off()
}

cat("\nCOMPLETE. Outputs in: ", normalizePath(OUTPUT_DIR), "\n", sep = "")
cat("\nKey outputs:\n")
cat("  comparison_contrasts_empirical.csv\n")
cat("  comparison_anova_empirical.csv\n")
cat("  surrogate_null_distributions.pdf\n")
cat("PNG plots saved to:", plot_dir, "\n")
cat("  sensitivity_zmi_anova.csv and sensitivity_zmi_contrasts_bonferroni.csv\n")
