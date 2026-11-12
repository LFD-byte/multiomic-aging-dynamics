#!/usr/bin/env Rscript
# Reconstruct the logic of Shen et al. Fig. 4a-c from saved DEswan results.
# Raw count curves are never smoothed or replaced by breakpoint densities.
# Window labels denote buckets.size (half-width), not total window width.

options(stringsAsFactors = FALSE, warn = 1)
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
file_arg <- grep("^--file=", commandArgs(), value = TRUE)
base_dir <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/"))
} else if (file.exists("01-de_swan_omics_one.R")) {
  normalizePath(".", winslash = "/")
} else {
  normalizePath("02-papers/Result-02", winslash = "/")
}
data_dir <- file.path(base_dir, "outputs", "deswan-window-sensitivity")
figure_dir <- file.path(base_dir, "figures", "deswan-window-sensitivity")
invisible(lapply(c(data_dir, figure_dir), dir.create,
                 recursive = TRUE, showWarnings = FALSE))

WINDOWS <- c(5, 7.5, 10, 12.5, 15)
SELECTED_WINDOW <- 7.5
Q_CUTOFFS <- c(0.0001, 0.001, 0.01, 0.05)
SELECTED_Q <- 0.05
OMICS <- c("transcriptomics", "metabolomics", "lipidomics",
           "gut_microbiome", "oral_microbiome")
OMICS_LABELS <- c(transcriptomics = "Transcriptomics",
                  metabolomics = "Metabolomics", lipidomics = "Lipidomics",
                  gut_microbiome = "Gut microbiome",
                  oral_microbiome = "Oral microbiome", All = "All five omics")
window_tag <- function(w) paste0("window_", gsub("\\.", "p", as.character(w)))
assert <- function(ok, message) if (!isTRUE(ok)) stop(message, call. = FALSE)
assert(SELECTED_WINDOW %in% WINDOWS, "The primary window must be among the sensitivity windows.")

# Load only the peak detector from the postprocessing stage; never source the pipeline.
analysis_file <- file.path(base_dir, "01-plot_deswan_omics_one.R")
expressions <- parse(analysis_file, encoding = "UTF-8")
is_detector <- vapply(expressions, function(x) {
  is.call(x) && identical(x[[1]], as.name("<-")) &&
    identical(x[[2]], as.name("detect_crests"))
}, logical(1))
assert(sum(is_detector) == 1, "Expected exactly one detect_crests definition.")
eval(expressions[[which(is_detector)]])

build_data <- function() {
  assert(requireNamespace("DEswan", quietly = TRUE), "DEswan is required.")
  count_parts <- list()
  audit_parts <- list()
  source_files <- character()
  feature_sets <- list()
  common_ages <- NULL
  i <- 0L
  for (w in WINDOWS) for (om in OMICS) {
    i <- i + 1L
    prefix <- file.path(base_dir, "outputs", paste0("deswan-window_", w))
    rds_file <- file.path(prefix, paste0(om, "_deswan_breakpoint_", window_tag(w), ".rds"))
    csv_file <- file.path(prefix, paste0(om, "_window_counts_", window_tag(w), ".csv"))
    assert(file.exists(rds_file) && file.exists(csv_file), paste("Missing input:", om, w))
    source_files <- c(source_files, rds_file, csv_file)
    res <- readRDS(rds_file)
    assert(all(c("p", "coeff") %in% names(res)), "Unexpected RDS structure.")
    raw <- res$p
    assert(all(c("variable", "window.center", "factor", "pvalue") %in% names(raw)),
           "Missing p-value columns.")
    age_rows <- raw[raw$factor == "qt", , drop = FALSE]
    assert(!anyDuplicated(age_rows[c("variable", "window.center")]),
           "Duplicate age-feature tests would inflate counts.")
    assert(all(is.na(age_rows$pvalue) | (is.finite(age_rows$pvalue) &
                   age_rows$pvalue >= 0 & age_rows$pvalue <= 1)), "Invalid p values.")
    p_wide <- DEswan::reshape.DEswan(res, parameter = 1, factor = "qt")
    q_wide <- DEswan::q.DEswan(p_wide, method = "BH")
    ages <- as.numeric(sub("^X", "", names(q_wide)[-1]))
    order_age <- order(ages)
    ages <- ages[order_age]
    q_mat <- as.matrix(q_wide[, -1, drop = FALSE])[, order_age, drop = FALSE]
    storage.mode(q_mat) <- "numeric"
    assert(all(is.finite(ages)) && !anyDuplicated(ages), "Invalid age grid.")
    assert(all(diff(ages) == 1), "Missing age centers must not be bridged silently.")
    if (is.null(common_ages)) common_ages <- ages
    assert(identical(ages, common_ages), "Age grids differ across inputs; inspect before pooling.")
    feature_key <- sort(as.character(q_wide$variable))
    if (is.null(feature_sets[[om]])) feature_sets[[om]] <- feature_key
    assert(identical(feature_key, feature_sets[[om]]), "Feature universe changed across windows.")
    n_tested <- colSums(!is.na(q_mat))
    counts <- rbindlist(lapply(Q_CUTOFFS, function(cutoff) data.table(
      omics = om, window = w, full_window_years = 2 * w,
      q_cutoff = cutoff, age = ages,
      n_significant = colSums(!is.na(q_mat) & q_mat < cutoff),
      n_tested = n_tested
    )))
    saved <- fread(csv_file)
    setorder(saved, age)
    calculated <- counts[q_cutoff == SELECTED_Q]
    assert(identical(as.numeric(saved$age), ages), "Saved CSV has a different age grid.")
    difference <- calculated$n_significant - saved$n_significant
    assert(all(difference == 0), paste("Recomputed counts disagree with CSV:", om, w))
    assert(all(calculated$n_tested == saved$n_tested), "Tested-feature counts disagree.")
    assert(all(counts$n_significant <= counts$n_tested), "Significant count exceeds denominator.")
    nested <- counts[, .(ok = all(diff(n_significant[order(q_cutoff)]) >= 0)), by = age]
    assert(all(nested$ok), "Counts must be nested as the q cutoff increases.")
    audit_parts[[i]] <- data.table(
      omics = om, window = w, full_window_years = 2 * w,
      raw_p_rows = nrow(raw), age_test_rows = nrow(age_rows),
      excluded_non_age_rows = sum(raw$factor != "qt"),
      factors_present = paste(sort(unique(raw$factor)), collapse = ";"),
      n_features = nrow(q_mat), n_age_centers = length(ages),
      age_min = min(ages), age_max = max(ages),
      n_tested_min = min(n_tested), n_tested_max = max(n_tested),
      missing_q_values = sum(is.na(q_mat)),
      max_abs_count_difference = max(abs(difference)), csv_match = TRUE
    )
    count_parts[[i]] <- counts
    message(sprintf("Verified %s, window %s: %d features x %d ages; CSV agrees.",
                    om, w, nrow(q_mat), length(ages)))
  }
  individual <- rbindlist(count_parts)
  pooled <- individual[, .(n_significant = sum(n_significant),
                           n_tested = sum(n_tested)),
                       by = .(window, full_window_years, q_cutoff, age)]
  pooled[, omics := "All"]
  counts <- rbindlist(list(individual, pooled), use.names = TRUE)
  counts[, proportion_significant := n_significant / n_tested]
  setorder(counts, omics, window, q_cutoff, age)
  fwrite(counts, file.path(data_dir, "source_counts_all.csv"))
  fwrite(rbindlist(audit_parts), file.path(data_dir, "input_validation.csv"))
  hashes <- data.table(file = substring(source_files, nchar(base_dir) + 2),
                      md5 = unname(tools::md5sum(source_files)))
  fwrite(hashes, file.path(data_dir, "input_checksums.csv"))
  counts
}

counts <- if ("--plots-only" %in% args) {
  fread(file.path(data_dir, "source_counts_all.csv"))
} else build_data()

# Save directly consumable Source Data for the three panels.
panel_a <- counts[window == SELECTED_WINDOW & q_cutoff == SELECTED_Q]
panel_b <- counts[window == SELECTED_WINDOW]
panel_c <- counts[q_cutoff == SELECTED_Q]
fwrite(panel_a, file.path(data_dir, "source_panel_a.csv"))
fwrite(panel_b, file.path(data_dir, "source_panel_b.csv"))
fwrite(panel_c, file.path(data_dir, "source_panel_c.csv"))
assert(identical(panel_a$n_significant,
                 panel_b[q_cutoff == SELECTED_Q]$n_significant), "a/b baseline differs.")
assert(identical(panel_a$n_significant,
                 panel_c[window == SELECTED_WINDOW]$n_significant), "a/c baseline differs.")

peak_parts <- list()
curve_parts <- list()
candidate_parts <- list()
separation_parts <- list()
groups <- unique(counts[, .(omics, window, q_cutoff)])
for (i in seq_len(nrow(groups))) {
  key <- groups[i]
  d <- counts[key, on = .(omics, window, q_cutoff)]
  setorder(d, age)
  peaks <- detect_crests(d$age, d$n_significant,
                         min_peak_relative_height = 0.15,
                         min_peak_distance = 4, max_valley_ratio = 0.85)
  with_key <- function(x) if (nrow(x)) cbind(key[rep(1L, nrow(x))], as.data.table(x))
  peak_parts[[i]] <- with_key(peaks$crests)
  curve_parts[[i]] <- if (!is.null(peaks$detection_curve)) with_key(peaks$detection_curve)
  candidate_parts[[i]] <- with_key(peaks$candidates)
  separation_parts[[i]] <- with_key(peaks$separations)
}
peak_table <- rbindlist(peak_parts, fill = TRUE)
fwrite(peak_table, file.path(data_dir, "detected_crests.csv"))
fwrite(rbindlist(curve_parts, fill = TRUE), file.path(data_dir, "peak_detection_curves.csv"))
fwrite(rbindlist(candidate_parts, fill = TRUE), file.path(data_dir, "peak_candidates.csv"))
fwrite(rbindlist(separation_parts, fill = TRUE), file.path(data_dir, "peak_separations.csv"))

window_comparison <- rbindlist(lapply(c("All", OMICS), function(om) {
  reference <- counts[omics == om & window == SELECTED_WINDOW & q_cutoff == SELECTED_Q]
  rbindlist(lapply(WINDOWS, function(w) {
    d <- counts[omics == om & window == w & q_cutoff == SELECTED_Q]
    assert(identical(d$age, reference$age), "Correlation requires matched ages.")
    p <- peak_table[omics == om & window == w & q_cutoff == SELECTED_Q]
    data.table(omics = om, window = w, full_window_years = 2 * w,
               selected = w == SELECTED_WINDOW,
               reference_window = SELECTED_WINDOW,
               global_max_age = d$age[which.max(d$n_significant)],
               global_max_n = max(d$n_significant),
               detected_crest_ages = paste(p$peak_age, collapse = ";"),
               pearson_r_vs_reference = cor(d$n_significant, reference$n_significant),
               spearman_rho_vs_reference = cor(d$n_significant, reference$n_significant,
                                               method = "spearman"))
  }))
}))
fwrite(window_comparison, file.path(data_dir, "window_comparison.csv"))
fwrite(counts[, .(global_max_age = age[which.max(n_significant)],
                 global_max_n = max(n_significant)),
             by = .(omics, window, q_cutoff)], file.path(data_dir, "global_maxima.csv"))
manifest <- list(
  generated_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
  selected_window = SELECTED_WINDOW, window_definition = "DEswan buckets.size; half-width in years",
  selected_total_width = 2 * SELECTED_WINDOW,
  windows = WINDOWS, full_window_widths = 2 * WINDOWS, q_cutoffs = Q_CUTOFFS,
  selected_q = SELECTED_Q, omics = OMICS, ages = sort(unique(counts$age)),
  fdr_scope = "BH within each omics and age window, using DEswan::q.DEswan",
  model = "Saved DEswan Gaussian GLM/type-II analysis; qt contrast and sex covariate",
  aggregation = "Sum significant counts across omics; feature IDs remain omics-specific",
  displayed_curve = "Observed counts connected by straight line segments; no smoothing",
  peak_detection = list(basis = "raw_significant_counts", smoothing_used = FALSE,
                        min_relative_height = 0.15,
                        min_distance_years = 4, max_valley_ratio = 0.85),
  n_participants = "Not stored in result RDS; cannot infer from n_tested",
  limitations = c("Overlapping windows are not independent replicates",
                  "Raw-count pooling is dominated by the much larger transcriptomics feature set",
                  "Current input script prefilters ages 40-70, truncating edge-window support",
                  "Saved RDS files lack complete run metadata, raw sample IDs and coefficients' uncertainties",
                  paste0("Window ", SELECTED_WINDOW,
                         " is the user-selected primary reference, not an optimized or validated best window")),
  pipeline_function_md5 = unname(tools::md5sum(analysis_file)),
  source_count_rows = nrow(counts), source_files_unchanged = TRUE
)
jsonlite::write_json(manifest, file.path(data_dir, "analysis_manifest.json"),
                     auto_unbox = TRUE, pretty = TRUE, digits = NA)
writeLines(capture.output(sessionInfo()), file.path(data_dir, "sessionInfo.txt"))
print(window_comparison[omics == "All"])
if ("--data-only" %in% args) quit(save = "no", status = 0)

# Plotting: every mark and export is produced by R.
suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})
required <- c("svglite", "ragg", "scales")
assert(all(vapply(required, requireNamespace, logical(1), quietly = TRUE)),
       "Missing an R plotting/export dependency.")
FONT <- "Arial"
Q_LABELS <- c("< 0.0001", "< 0.001", "< 0.01", "< 0.05")
Q_COLORS <- setNames(c("#7F8E9B", "#CC8B5A", "#7A689C", "#00868B"), Q_LABELS)
W_LABELS <- ifelse(WINDOWS == SELECTED_WINDOW,
                   paste0(WINDOWS, " (reference)"), as.character(WINDOWS))
W_COLORS <- setNames(c("#65519A", "#CD7557", "#0077B5", "#B3903C", "#648D70"), W_LABELS)
CREST_COLORS <- c("#C64D8A", "#0077B5", "#8C6D31", "#555555")

base_theme <- theme_classic(base_size = 7, base_family = FONT) + theme(
  axis.line = element_line(linewidth = 0.3, colour = "#40464D"),
  axis.ticks = element_line(linewidth = 0.25, colour = "#40464D"),
  axis.ticks.length = grid::unit(1.0, "mm"),
  axis.text = element_text(size = 6.5, colour = "#30363C"),
  axis.title = element_text(size = 7),
  axis.title.x = element_text(margin = margin(t = 4)),
  axis.title.y = element_text(margin = margin(r = 4)),
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  plot.title = element_text(size = 7.5, face = "bold", margin = margin(b = 3)),
  plot.subtitle = element_text(size = 6.2, colour = "#59636D", margin = margin(b = 6)),
  plot.tag = element_text(size = 9, face = "bold"),
  plot.tag.position = "topleft",
  plot.background = element_blank(),
  legend.background = element_blank(), legend.box.background = element_blank(),
  legend.position = "bottom", legend.title.position = "top",
  legend.justification = "left", legend.box.just = "left",
  legend.title = element_text(size = 6.5), legend.text = element_text(size = 6.2),
  legend.key.width = grid::unit(4.3, "mm"),
  legend.key.height = grid::unit(3.2, "mm"),
  legend.spacing.x = grid::unit(1, "mm"),
  legend.margin = margin(t = 3, r = 0, b = 0, l = 0),
  plot.margin = margin(t = 4, r = 5, b = 4, l = 4)
)

make_triptych <- function(om = "All", label_offset = 0L,
                          show_legends = TRUE, compact = FALSE) {
  d_a <- copy(panel_a[omics == om])
  d_b <- copy(panel_b[omics == om])
  d_c <- copy(panel_c[omics == om])
  peaks <- copy(peak_table[omics == om & window == SELECTED_WINDOW & q_cutoff == SELECTED_Q])
  # All observed points are retained, including edge values and exact zeros.
  assert(all(is.finite(c(d_a$n_significant, d_b$n_significant, d_c$n_significant))),
         "Non-finite counts require an explicit missing-data policy.")
  peak_height <- max(c(d_a$n_significant, d_b$n_significant, d_c$n_significant))
  tick_step <- if (peak_height > 2000) 1000 else if (peak_height > 120) 50 else 25
  top <- max(tick_step, ceiling(peak_height * 1.18 / tick_step) * tick_step)
  breaks_y <- seq(0, top, tick_step)
  if (length(breaks_y) > 6) breaks_y <- breaks_y[seq(1, length(breaks_y), by = 2)]
  common <- list(
    scale_x_continuous(breaks = seq(40, 70, 5), limits = c(40, 70), expand = c(0, 0)),
    scale_y_continuous(breaks = breaks_y, limits = c(0, top),
                       labels = scales::label_comma(), expand = c(0, 0)),
    labs(x = "Age (years)", y = "Significant features"), base_theme
  )
  p_a <- ggplot(d_a, aes(age, n_significant))
  if (nrow(peaks)) {
    peaks[, crest_colour := CREST_COLORS[(seq_len(.N) - 1L) %% length(CREST_COLORS) + 1L]]
    peaks[, label_y := pmin(peak_n + top * if (compact) 0.13 else 0.12, top * 0.92)]
    peaks[, crest_label := if (compact) as.character(peak_age) else paste0("Crest ", crest_id, ": ", peak_age)]
    p_a <- p_a + geom_segment(data = peaks,
      aes(x = peak_age, xend = peak_age, y = 0, yend = peak_n),
      inherit.aes = FALSE, colour = peaks$crest_colour,
      linewidth = 0.3, linetype = "dashed")
  }
  p_a <- p_a + geom_line(linewidth = 0.5, colour = "#232B32") +
    geom_point(size = if (compact) 0.65 else 1.0, colour = "#232B32") + common
  if (nrow(peaks)) p_a <- p_a + geom_text(data = peaks,
    aes(x = peak_age, y = label_y, label = crest_label),
    inherit.aes = FALSE, family = FONT, size = if (compact) 2.1 else 2.2,
    colour = peaks$crest_colour, vjust = 0.5)

  d_b[, cutoff_label := factor(Q_LABELS[match(q_cutoff, Q_CUTOFFS)], levels = Q_LABELS)]
  d_b[, selected := q_cutoff == SELECTED_Q]
  d_c[, window_label := factor(W_LABELS[match(window, WINDOWS)], levels = W_LABELS)]
  d_c[, selected := window == SELECTED_WINDOW]
  p_b <- ggplot(d_b, aes(age, n_significant, colour = cutoff_label, group = cutoff_label)) +
    geom_line(aes(linewidth = selected)) +
    geom_point(size = if (compact) 0.45 else 0.65) + common +
    scale_colour_manual(values = Q_COLORS, drop = FALSE, name = "q cutoff") +
    scale_linewidth_manual(values = c(`FALSE` = 0.35, `TRUE` = 0.65), guide = "none") +
    guides(colour = guide_legend(nrow = 2, byrow = TRUE,
                                override.aes = list(linewidth = 0.55, size = 1)))
  # Draw the selected curve last so its geometry remains visible.
  p_c <- ggplot(d_c[selected == FALSE],
                aes(age, n_significant, colour = window_label, group = window_label)) +
    geom_line(linewidth = 0.35) + geom_point(size = if (compact) 0.45 else 0.65) +
    geom_line(data = d_c[selected == TRUE], linewidth = 0.7) +
    geom_point(data = d_c[selected == TRUE], size = if (compact) 0.65 else 0.9) + common +
    scale_colour_manual(values = W_COLORS, limits = W_LABELS, drop = FALSE,
                         name = "Window (years per side)") +
    guides(colour = guide_legend(nrow = 2, byrow = TRUE,
                                override.aes = list(linewidth = 0.55, size = 1)))
  titles <- if (compact) c(unname(OMICS_LABELS[om]), "q-value sensitivity", "Window sensitivity") else
    c("Reference profile", "q-value sensitivity", "Window sensitivity")
  subtitles <- c(paste0("Window = ", SELECTED_WINDOW, "; q < ", SELECTED_Q),
                 paste0("Window = ", SELECTED_WINDOW),
                 paste0("q < ", SELECTED_Q, "; window ", SELECTED_WINDOW, " reference"))
  plots <- list(p_a, p_b, p_c)
  for (j in seq_along(plots)) {
    plots[[j]] <- plots[[j]] + labs(title = titles[j], subtitle = subtitles[j],
                                    tag = letters[label_offset + j])
    if (!show_legends) plots[[j]] <- plots[[j]] + theme(legend.position = "none")
    if (compact) plots[[j]] <- plots[[j]] + theme(
      plot.title = element_text(size = 7), plot.subtitle = element_text(size = 6),
      axis.text = element_text(size = 6), axis.title = element_text(size = 6.5),
      plot.margin = margin(t = 4, r = 4, b = 4, l = 4))
  }
  plots
}

export_figure <- function(plot, name, width_mm, height_mm, n_panels) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4
  output <- file.path(figure_dir, name)
  grDevices::cairo_pdf(paste0(output, ".pdf"), width = w, height = h,
                       family = FONT, bg = "white")
  print(plot)
  grDevices::dev.off()
  svglite::svglite(paste0(output, ".svg"), width = w, height = h, bg = "white")
  print(plot)
  grDevices::dev.off()
  ggsave(paste0(output, ".png"), plot, width = w, height = h,
         device = ragg::agg_png, dpi = 300, bg = "white")
  ggsave(paste0(output, ".tiff"), plot, width = w, height = h,
         device = ragg::agg_tiff, dpi = 600, compression = "lzw", bg = "white")
  message("Exported ", name, ": PDF, SVG, 300 dpi PNG and 600 dpi TIFF.")
}

main_plot <- wrap_plots(make_triptych(), ncol = 3, widths = c(1, 1, 1)) +
  plot_annotation(
    title = "Age-associated waves across five omics",
    subtitle = paste0("Primary window ", SELECTED_WINDOW, ": ", SELECTED_WINDOW,
                      " years on each side of the age center (", 2 * SELECTED_WINDOW,
                      "-year total span)"),
    caption = "Counts at 29 age centers; 18,417 tested features. BH correction within each omics and age window.",
    theme = theme(plot.title = element_text(family = FONT, size = 10, face = "bold"),
                  plot.subtitle = element_text(family = FONT, size = 7, margin = margin(b = 5)),
                  plot.caption = element_text(family = FONT, size = 6.2, hjust = 0,
                                               colour = "#59636D", margin = margin(t = 5)),
                  plot.margin = margin(5, 5, 5, 5)))
export_figure(main_plot, "fig4abc_window_sensitivity", width_mm = 183, height_mm = 95, n_panels = 3)

omics_panels <- unlist(lapply(seq_along(OMICS), function(i) {
  make_triptych(OMICS[i], label_offset = 3L * (i - 1L),
                show_legends = TRUE, compact = TRUE)
}), recursive = FALSE)
omics_plot <- wrap_plots(omics_panels, ncol = 3, widths = c(1, 1, 1), guides = "collect") +
  plot_annotation(
    title = "Age-associated waves resolved by omics",
    subtitle = "Each row: reference profile | q-value sensitivity | window sensitivity. Legends apply to all rows.",
    caption = "Window = years per side. Curves connect observed counts; labels identify detected crests. Y scales vary by omics.",
    theme = theme(plot.title = element_text(family = FONT, size = 10, face = "bold"),
                  plot.subtitle = element_text(family = FONT, size = 7, margin = margin(b = 5)),
                  plot.caption = element_text(family = FONT, size = 6.2, hjust = 0,
                                               colour = "#59636D", margin = margin(t = 5)),
                  plot.margin = margin(5, 5, 5, 5))) & theme(legend.position = "bottom")
export_figure(omics_plot, "fig4abc_by_omics", width_mm = 183, height_mm = 235, n_panels = 15)
writeLines(capture.output(sessionInfo()), file.path(data_dir, "sessionInfo.txt"))
