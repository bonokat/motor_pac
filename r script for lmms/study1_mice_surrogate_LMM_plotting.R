library(data.table)
library(ggplot2)

# ============================================================
# 1. LOAD DATA
# ============================================================

OUTPUT_DIR <- "F:/research project/# study 1/eeg_data/set/Y group/roi_source_analysis/surrogate_analysis/study1_mice_surrogate_outputs"

df <- fread(file.path(OUTPUT_DIR, "observed_vs_pooled_emm.csv"))

# ============================================================
# 2. RECODE LABELS
# ============================================================

df[, condition_label := factor(
  condition,
  levels = c("_BL", "_MAIN_baseline", "_MAIN_adaptation"),
  labels = c("FTT", "De-CRAT baseline", "De-CRAT adaptation")
)]

df[, stage_label := factor(
  task_stage,
  levels = c("_plan", "_go"),
  labels = c("Planning", "Execution")
)]

# Optional: rescale for readability
# Here we express MI in x 10^-6
df[, observed_mean_plot := observed_mean * 1e6]
df[, observed_SE_plot   := observed_SE   * 1e6]
df[, surrogate_mean_plot := surrogate_mean * 1e6]
df[, surrogate_SE_plot   := surrogate_SE   * 1e6]

# ============================================================
# 3. RESHAPE TO LONG FORMAT
# ============================================================

obs_long <- df[, .(
  condition_label,
  stage_label,
  source = "Observed MI",
  mean = observed_mean_plot,
  se = observed_SE_plot
)]

sur_long <- df[, .(
  condition_label,
  stage_label,
  source = "Pooled surrogate MI",
  mean = surrogate_mean_plot,
  se = surrogate_SE_plot
)]

plot_df <- rbindlist(list(obs_long, sur_long))

plot_df[, source := factor(
  source,
  levels = c("Observed MI", "Pooled surrogate MI")
)]

# ============================================================
# 4. PLOT
# ============================================================

p <- ggplot(
  plot_df,
  aes(x = condition_label, y = mean, group = source, shape = source, linetype = source)
) +
  geom_point(
    position = position_dodge(width = 0.35),
    size = 3
  ) +
  geom_line(
    position = position_dodge(width = 0.35),
    linewidth = 0.8
  ) +
  geom_errorbar(
    aes(ymin = mean - se, ymax = mean + se),
    position = position_dodge(width = 0.35),
    width = 0.15,
    linewidth = 0.7
  ) +
  facet_wrap(~ stage_label, nrow = 1) +
  labs(
    x = NULL,
    y = expression("Theta-gamma MI ("*10^{-6}*")"),
    shape = NULL,
    linetype = NULL
  ) +
  theme_classic(base_size = 13) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(angle = 20, hjust = 1),
    legend.position = "top"
  )

print(p)

# ============================================================
# 5. SAVE
# ============================================================

ggsave(
  filename = file.path(OUTPUT_DIR, "observed_vs_pooled_surrogate_plot.png"),
  plot = p,
  width = 10,
  height = 4.8,
  dpi = 300
)

ggsave(
  filename = file.path(OUTPUT_DIR, "observed_vs_pooled_surrogate_plot.pdf"),
  plot = p,
  width = 10,
  height = 4.8
)