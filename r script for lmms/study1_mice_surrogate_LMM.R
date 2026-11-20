# ============================================================
# STUDY 1: OBSERVED AND SURROGATE LMM MARGINAL MEANS
# ============================================================

library(data.table)
library(lme4)
library(emmeans)
library(mice)
library(broom.mixed)

# ----- PATHS -----
OBSERVED_FILE <- "F:/research project/# study 1/eeg_data/set/Y group/roi_source_analysis/surrogate_analysis/study1_original_roi_labels_corrected.csv"
SURROGATE_FILE <- "F:/research project/# study 1/eeg_data/set/Y group/roi_source_analysis/surrogate_analysis/study1_surrogates_merged_validated.csv"
OUTPUT_DIR <- "F:/research project/# study 1/eeg_data/set/Y group/roi_source_analysis/surrogate_analysis/study1_mice_surrogate_outputs"
SURROGATE_IDS  <- 1:200     # pilot first; change to 1:200 after checking
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)


# ============================================================
# 2. LOAD DATA
# ============================================================

obs <- fread(OBSERVED_FILE)
sur <- fread(SURROGATE_FILE)

# Reconstruct condition labels in the original dataset
obs[is.na(block), block := ""]
obs[, condition := paste0(task, block)]

# Set identical factor levels for both datasets
condition_levels <- c(
  "_BL",
  "_MAIN_baseline",
  "_MAIN_adaptation"
)

stage_levels <- c("_plan", "_go")

roi_levels <- sort(unique(obs$roi))
sub_levels <- sort(unique(obs$sub))

prepare_data <- function(d) {
  
  d[, condition := factor(
    condition, levels = condition_levels
  )]
  
  d[, task_stage := factor(
    task_stage, levels = stage_levels
  )]
  
  d[, roi := factor(roi, levels = roi_levels)]
  d[, sub := factor(sub, levels = sub_levels)]
  
  return(d)
}

obs <- prepare_data(obs)
sur <- prepare_data(sur)


# ============================================================
# 3. VALIDATE DATA
# ============================================================

key_cols <- c("sub", "condition", "task_stage", "roi")

stopifnot(
  !anyNA(obs[, ..key_cols]),
  !anyNA(sur[, ..key_cols]),
  all(is.finite(obs$pac_value)),
  all(is.finite(sur$pac_value)),
  uniqueN(obs, by = key_cols) == nrow(obs),
  uniqueN(sur, by = c(key_cols, "surrogate_id")) == nrow(sur),
  all(SURROGATE_IDS %in% sur$surrogate_id)
)

sur_counts <- sur[
  surrogate_id %in% SURROGATE_IDS,
  .N,
  by = surrogate_id
]

stopifnot(all(sur_counts$N == nrow(obs)))

cat("Original observations:", nrow(obs), "\n")
cat("Surrogate observations:", nrow(sur), "\n")
cat("Participants:", uniqueN(obs$sub), "\n")
cat("ROIs:", uniqueN(obs$roi), "\n")
cat("Surrogate iterations:", length(SURROGATE_IDS), "\n")


# ============================================================
# 4. DEFINE MODEL 7
# ============================================================

formula_m7 <- pac_value ~ 
  condition * roi * task_stage + (1 | sub)

fit_model <- function(d) {
  
  lmer(
    formula_m7,
    data = d,
    REML = TRUE,
    control = lmerControl(
      optimizer = "bobyqa",
      optCtrl = list(maxfun = 2e5)
    )
  )
}


# ============================================================
# 5. FIT ORIGINAL MODEL
# ============================================================

model_obs <- fit_model(obs)

# Extract estimated marginal means and standard errors
obs_emm <- as.data.table(
  emmeans(
    model_obs,
    ~ condition | task_stage,
    weights = "equal",
    lmer.df = "asymptotic"
  )
)[, .(condition, task_stage, emmean, SE)]

print(obs_emm)


# ============================================================
# 6. FIT SURROGATE MODELS
# ============================================================

models_full <- vector("list", length(SURROGATE_IDS))

sur_emm_list <- vector("list", length(SURROGATE_IDS))

for (i in seq_along(SURROGATE_IDS)) {
  
  sid <- SURROGATE_IDS[i]
  
  # Select a single surrogate dataset
  d <- sur[surrogate_id == sid]
  
  # Fit Model 7
  models_full[[i]] <- fit_model(d)
  
  # Extract marginal means and standard errors
  emm <- as.data.table(
    emmeans(
      models_full[[i]],
      ~ condition | task_stage,
      weights = "equal",
      lmer.df = "asymptotic"
    )
  )[, .(condition, task_stage, emmean, SE)]
  
  emm[, surrogate_id := sid]
  
  sur_emm_list[[i]] <- emm
  
  cat(
    "Completed surrogate", i,
    "of", length(SURROGATE_IDS), "\n"
  )
}

sur_emm <- rbindlist(sur_emm_list)


# ============================================================
# 7. VALIDATE OUTPUTS
# ============================================================

stopifnot(
  nrow(obs_emm) == 6,
  nrow(sur_emm) == 6 * length(SURROGATE_IDS),
  all(is.finite(obs_emm$emmean)),
  all(is.finite(sur_emm$emmean)),
  all(is.finite(obs_emm$SE)),
  all(is.finite(sur_emm$SE))
)

print(obs_emm)
print(head(sur_emm))

cat("Original EMMs:", nrow(obs_emm), "\n")
cat("Surrogate EMMs:", nrow(sur_emm), "\n")


# ============================================================
# 8. SAVE OUTPUTS
# ============================================================

fwrite(
  obs_emm,
  file.path(OUTPUT_DIR, "observed_emm.csv")
)

fwrite(
  sur_emm,
  file.path(OUTPUT_DIR, "surrogate_emm.csv")
)

# Save fitted models for subsequent analyses
saveRDS(
  model_obs,
  file.path(OUTPUT_DIR, "model_obs.rds")
)

saveRDS(
  models_full,
  file.path(OUTPUT_DIR, "models_full.rds")
)

cat("Analysis completed.\n")
cat("Outputs saved to:", OUTPUT_DIR, "\n")


# ============================================================
# 9. POOL AND COMPARE MODEL OUTPUTS
# ============================================================

# Load previously generated EMMs
obs_emm <- fread(file.path(OUTPUT_DIR, "observed_emm.csv"))
sur_emm <- fread(file.path(OUTPUT_DIR, "surrogate_emm.csv"))

# Pool surrogate EMMs within each condition and stage
pooled_emm <- sur_emm[, {
  
  pooled <- mice::pool.scalar(
    Q = emmean,
    U = SE^2,
    rule = "reiter2003"
  )
  
  .(
    surrogate_mean = pooled$qbar,
    surrogate_SE = sqrt(pooled$ubar),
    surrogate_SD = sd(emmean),
    n_surrogates = .N
  )
  
}, by = .(condition, task_stage)]

# Rename original model columns
setnames(
  obs_emm,
  c("emmean", "SE"),
  c("observed_mean", "observed_SE")
)

# Combine original and pooled surrogate estimates
comparison <- merge(
  obs_emm,
  pooled_emm,
  by = c("condition", "task_stage"),
  sort = FALSE
)

# Descriptive difference
comparison[, MI_difference :=
             observed_mean - surrogate_mean
]

# Save outputs
fwrite(
  pooled_emm,
  file.path(OUTPUT_DIR, "pooled_surrogate_emm.csv")
)

fwrite(
  comparison,
  file.path(OUTPUT_DIR, "observed_vs_pooled_emm.csv")
)

print(comparison)
