#!/usr/bin/env Rscript
# Local lipid-set analysis at classified DE-SWAN MAIN peaks only.
# Responsibility: read DE-SWAN, save complete annotated peak statistics,
# perform local competitive Wilcoxon / directional Fisher enrichment, and
# draw the two local evidence panels. No website inputs or results are handled.
#
# Separate entry points:
#   03_prepare_lion_custom_statistics.R: prepare LION/web inputs only.
#   03_visualize_lion_results.R: visualize downloaded two-sided ranking results.
#   03_visualize_lion_target_list_results.R: visualize directional target results.
# Shared helpers provide pure data/annotation functions; entry points do not
# invoke each other. Local Wilcoxon tests are NOT the LION/web KS test.
#
# Positive beta means greater abundance in the older/right parcel at the
# exact main-peak center, not a longitudinal change or a per-year slope.
# Sum-composition annotations are not expanded into unmeasured acyl isomers.
# Lipid BH correction uses ALL tested features before any mapping/selection.
#
# Rscript 03_lipid_pathway_enrichment.R --tables_only=true
# Optional: --rds=... --crest=... --peaks=48 --output=... --figure=...
# Default RDS: outputs/deswan-window_7.5/lipidomics_deswan_breakpoint_window_7p5.rds
# Tables: outputs/lipid_pathway; local figures: <output>/figures.
# --tables_only=false (default) also renders the local evidence figure.
# --export_only is a deprecated alias for --tables_only; it never exports LION.
set.seed(42)


# =============================================================================
# 1. Configuration and command-line arguments
# =============================================================================

get_script_dir <- function() {
  source_paths <- as.character(unlist(lapply(sys.frames(), function(frame) frame$ofile)))
  source_paths <- source_paths[basename(source_paths) == "03_lipid_pathway_enrichment.R"]
  if (length(source_paths)) {
    return(dirname(normalizePath(tail(source_paths, 1L), winslash = "/", mustWork = TRUE)))
  }
  file_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_argument) > 0) {
    script_path <- sub("^--file=", "", file_argument[[1]])
    return(dirname(normalizePath(script_path, winslash = "/", mustWork = TRUE)))
  }
  candidates <- c(getwd(), file.path(getwd(), "02-papers", "Result-02"))
  candidates <- candidates[file.exists(file.path(candidates, "03_lipid_pathway_enrichment.R"))]
  if (!length(candidates)) stop("Cannot locate 03_lipid_pathway_enrichment.R.")
  normalizePath(candidates[[1L]], winslash = "/", mustWork = TRUE)
}

SCRIPT_DIR <- get_script_dir()
source(file.path(SCRIPT_DIR, "lion_custom_statistics_helpers.R"), encoding = "UTF-8")
source(file.path(SCRIPT_DIR, "lipid_annotation_helpers.R"), encoding = "UTF-8")
source(file.path(SCRIPT_DIR, "lion_target_list_helpers.R"), encoding = "UTF-8")

DEFAULT_PATHS <- lion_default_paths(SCRIPT_DIR)

parse_cli <- function(args) {
  defaults <- list(
    rds = DEFAULT_PATHS$rds,
    crest = "auto",
    peaks = "auto",
    output = "auto",
    figure = "auto",
    fdr = "0.05",
    min_set_size = "3",
    max_set_fraction = "0.80",
    tables_only = "false",
    top_sets = "12",
    top_lipids_per_direction = "3"
  )

  for (arg in args) {
    if (!grepl("^--[^=]+=", arg)) stop("Expected --key=value: ", arg)
    pieces <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
    key <- pieces[[1]]
    value <- paste(pieces[-1], collapse = "=")
    if (key == "export_only") {
      warning("--export_only now means local tables only; prefer --tables_only. LION inputs belong to 03_prepare_lion_custom_statistics.R.")
      key <- "tables_only"
    }
    if (!key %in% names(defaults)) stop("Unknown local-analysis option: --", key,
      ". LION inputs are prepared by 03_prepare_lion_custom_statistics.R.")
    defaults[[key]] <- value
  }
  defaults
}

ARGS <- parse_cli(commandArgs(trailingOnly = TRUE))
RDS_FILE <- ARGS$rds
if (is.na(RDS_FILE) || !file.exists(RDS_FILE)) {
  stop(
    "DE-SWAN RDS not found. Supply it with ",
    "--rds=/path/to/lipidomics_deswan_breakpoint_window_7p5.rds"
  )
}
PEAK_CONFIG <- resolve_crest_peaks(ARGS$peaks, ARGS$crest, RDS_FILE)
CREST_PEAKS <- PEAK_CONFIG$crests
PEAK_AGES <- PEAK_CONFIG$ages
PEAK_SOURCE <- PEAK_CONFIG$source
CREST_FILE <- PEAK_CONFIG$crest_file
OUTPUT_DIR <- if (identical(ARGS$output, "auto")) {
  lion_default_paths(SCRIPT_DIR, RDS_FILE)$output
} else ARGS$output
FIGURE_DIR <- if (identical(ARGS$figure, "auto")) file.path(OUTPUT_DIR, "figures") else ARGS$figure
FDR_THRESHOLD <- as.numeric(ARGS$fdr)
MIN_SET_SIZE <- as.integer(ARGS$min_set_size)
MAX_SET_FRACTION <- as.numeric(ARGS$max_set_fraction)
if (!tolower(ARGS$tables_only) %in% c("true", "false", "1", "0", "yes", "no")) {
  stop("--tables_only must be true or false.")
}
TABLES_ONLY <- tolower(ARGS$tables_only) %in% c("true", "1", "yes")
TOP_SETS <- as.integer(ARGS$top_sets)
TOP_LIPIDS_PER_DIRECTION <- as.integer(ARGS$top_lipids_per_direction)

if (!is.finite(FDR_THRESHOLD) || FDR_THRESHOLD <= 0 || FDR_THRESHOLD >= 1) {
  stop("--fdr must be between 0 and 1.")
}
if (!is.finite(MIN_SET_SIZE) || MIN_SET_SIZE < 2) {
  stop("--min_set_size must be at least 2.")
}
if (!is.finite(MAX_SET_FRACTION) || MAX_SET_FRACTION <= 0 || MAX_SET_FRACTION >= 1) {
  stop("--max_set_fraction must be between 0 and 1.")
}
if (!is.finite(TOP_SETS) || TOP_SETS < 4) stop("--top_sets must be at least 4.")
if (!is.finite(TOP_LIPIDS_PER_DIRECTION) || TOP_LIPIDS_PER_DIRECTION < 1) {
  stop("--top_lipids_per_direction must be at least 1.")
}

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
if (!TABLES_ONLY) dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)



# =============================================================================
# 2. General helpers
# =============================================================================

write_csv <- function(x, path) {
  write.csv(x, path, row.names = FALSE, na = "")
  message("Saved: ", path)
}

safe_numeric <- function(x) suppressWarnings(as.numeric(as.character(x)))

clean_label <- function(x) {
  x <- gsub("[^A-Za-z0-9]+", "_", x)
  gsub("(^_+|_+$)", "", x)
}

# =============================================================================
# 3. Lipid nomenclature and structural annotation
# =============================================================================

# Pure annotations are shared with the separate LION input preparer.

# =============================================================================
# 4. Read and validate DE-SWAN statistics
# =============================================================================

deswan <- readRDS(RDS_FILE)

# Keep every tested feature in the exported statistics, including missing
# coefficients/p-values. Individual analyses explicitly handle finite scores.
# BH always precedes filtering and uses the full tested family at each center.
stats_all <- lion_deswan_statistics(deswan)
if (!any(is.finite(stats_all$p_value) & is.finite(stats_all$coefficient))) {
  stop("No usable age-effect rows were found in the RDS.")
}

stats_all$signed_score <- sign(stats_all$coefficient) *
  -log10(pmax(stats_all$p_value, .Machine$double.xmin))
stats_all$signed_z_score <- lion_signed_z(stats_all$coefficient, stats_all$p_value)
stats_all$direction <- ifelse(
  !is.finite(stats_all$coefficient), "unavailable", ifelse(
  stats_all$coefficient > 0,
  "older_high",
  ifelse(stats_all$coefficient < 0, "older_low", "no_change")
))

available_centers <- sort(unique(stats_all$center_age))
if (any(!PEAK_AGES %in% available_centers)) {
  stop("Crest peak ages must match exact DE-SWAN centers. Missing: ",
       paste(setdiff(PEAK_AGES, available_centers), collapse = ", "))
}
chosen_centers <- PEAK_AGES

peak_lookup <- data.frame(
  crest_id = CREST_PEAKS$crest_id,
  requested_peak_age = PEAK_AGES,
  peak_n_reported = CREST_PEAKS$peak_n,
  center_age = chosen_centers,
  n_analysis_windows = rep(1L, length(PEAK_AGES)),
  selection_rule = rep(
    paste0("BH q < ", FDR_THRESHOLD, " at the crest peak window"),
    length(PEAK_AGES)
  ),
  peak_label = paste0("age_", clean_label(format(PEAK_AGES, trim = TRUE))),
  peak_source = rep(PEAK_SOURCE, length(PEAK_AGES)),
  crest_file = rep(
    if (is.na(CREST_FILE)) "" else CREST_FILE,
    length(PEAK_AGES)
  ),
  stringsAsFactors = FALSE
)
crest_metadata <- intersect(
  c("is_main", "peak_class", "boundary_limited", "stability_status", "is_stable_main"),
  names(CREST_PEAKS)
)
peak_lookup <- cbind(peak_lookup, CREST_PEAKS[, crest_metadata, drop = FALSE])

stats_peak <- merge(
  stats_all,
  peak_lookup[, c(
    "crest_id", "requested_peak_age", "center_age", "peak_label", "is_main", "peak_class"
  )],
  by = "center_age",
  all = FALSE
)
stats_peak <- stats_peak[
  order(stats_peak$requested_peak_age, stats_peak$q_value, stats_peak$feature),
  ,
  drop = FALSE
]

annotations <- annotate_lipids(unique(stats_peak$feature))
stats_peak <- merge(stats_peak, annotations, by = "feature", all.x = TRUE)
stats_peak$peak_window_significant <-
  is.finite(stats_peak$q_value) & stats_peak$q_value < FDR_THRESHOLD

write_csv(peak_lookup, file.path(OUTPUT_DIR, "peak_age_lookup.csv"))
write_csv(annotations, file.path(OUTPUT_DIR, "lipid_name_mapping.csv"))
write_csv(
  stats_peak[stats_peak$peak_window_significant %in% TRUE, ],
  file.path(OUTPUT_DIR, "crest_associated_lipids.csv")
)
crest_selection_summary <- do.call(
  rbind,
  lapply(split(stats_peak, stats_peak$crest_id), function(x) {
    selected <- x$peak_window_significant %in% TRUE
    data.frame(
      crest_id = x$crest_id[[1]],
      peak_age = x$requested_peak_age[[1]],
      peak_n_reported = peak_lookup$peak_n_reported[
        match(x$crest_id[[1]], peak_lookup$crest_id)
      ],
      n_analysis_windows = 1L,
      n_measured_lipids = nrow(x),
      n_selected_lipids = sum(selected),
      count_difference = sum(selected) - peak_lookup$peak_n_reported[
        match(x$crest_id[[1]], peak_lookup$crest_id)
      ],
      n_older_high = sum(selected & x$direction == "older_high"),
      n_older_low = sum(selected & x$direction == "older_low"),
      n_direction_ambiguous = sum(selected & x$direction == "no_change"),
      selection_rule = paste0(
        "BH q < ", FDR_THRESHOLD, " at peak age ",
        x$requested_peak_age[[1]]
      ),
      stringsAsFactors = FALSE
    )
  })
)
write_csv(
  crest_selection_summary,
  file.path(OUTPUT_DIR, "crest_lipid_selection_summary.csv")
)
write_csv(
  crest_selection_summary[, c(
    "crest_id", "peak_age", "peak_n_reported", "n_selected_lipids",
    "count_difference", "selection_rule"
  )],
  file.path(OUTPUT_DIR, "crest_peak_count_validation.csv")
)
if (any(crest_selection_summary$count_difference != 0)) {
  warning(
    "Recovered peak lipid counts differ from crest peak_n; inspect ",
    file.path(OUTPUT_DIR, "crest_peak_count_validation.csv")
  )
}
write_csv(stats_peak, file.path(OUTPUT_DIR, "deswan_peak_statistics.csv"))


# =============================================================================
# 5. Build lipid sets from class, category and structural attributes
# =============================================================================

add_sets_from_field <- function(annotation, field, prefix) {
  values <- unique(annotation[[field]])
  values <- values[!is.na(values) & nzchar(values)]
  sets <- lapply(values, function(value) annotation$feature[annotation[[field]] == value])
  names(sets) <- paste0(prefix, ": ", values)
  sets
}

lipid_sets <- c(
  add_sets_from_field(annotations, "category", "Category"),
  add_sets_from_field(annotations, "lipid_class", "Class"),
  add_sets_from_field(annotations, "unsaturation", "Unsaturation"),
  add_sets_from_field(annotations, "chain_length", "Chain length")
)

class_unsaturation <- annotations[
  !is.na(annotations$lipid_class) & !is.na(annotations$unsaturation),
  c("feature", "lipid_class", "unsaturation")
]
if (nrow(class_unsaturation) > 0) {
  interaction_name <- paste(
    class_unsaturation$lipid_class,
    class_unsaturation$unsaturation,
    sep = " / "
  )
  interaction_sets <- split(class_unsaturation$feature, interaction_name)
  names(interaction_sets) <- paste0("Class-unsaturation: ", names(interaction_sets))
  lipid_sets <- c(lipid_sets, interaction_sets)
}

lipid_sets <- lapply(lipid_sets, unique)
universe <- unique(annotations$feature)
max_set_size <- floor(length(universe) * MAX_SET_FRACTION)
lipid_sets <- lipid_sets[
  vapply(
    lipid_sets,
    function(x) length(intersect(x, universe)) >= MIN_SET_SIZE &&
      length(intersect(x, universe)) <= max_set_size,
    logical(1)
  )
]

set_membership <- do.call(
  rbind,
  lapply(names(lipid_sets), function(set_name) {
    data.frame(
      lipid_set = set_name,
      feature = lipid_sets[[set_name]],
      stringsAsFactors = FALSE
    )
  })
)
set_membership <- merge(set_membership, annotations, by = "feature", all.x = TRUE)
write_csv(set_membership, file.path(OUTPUT_DIR, "lipid_set_membership.csv"))


# =============================================================================
# 6. Competitive ranked enrichment using all measured lipids
# =============================================================================

ranked_set_test <- function(stat_table, sets) {
  # Use the signed normal deviate at the exact crest peak window as the
  # continuous ranking statistic, matching the window used for lipid selection.
  scores <- stat_table$signed_z_score
  names(scores) <- stat_table$feature
  scores <- scores[is.finite(scores)]

  results <- lapply(names(sets), function(set_name) {
    members <- intersect(sets[[set_name]], names(scores))
    nonmembers <- setdiff(names(scores), members)
    if (length(members) < MIN_SET_SIZE || length(nonmembers) < 2) return(NULL)

    test <- suppressWarnings(
      wilcox.test(
        scores[members],
        scores[nonmembers],
        alternative = "two.sided",
        exact = FALSE,
        conf.int = FALSE
      )
    )

    ranks <- rank(scores, ties.method = "average")
    n_total <- length(ranks)
    expected_mean_rank <- (n_total + 1) / 2
    finite_population_correction <-
      (n_total - length(members)) / (n_total - 1)
    rank_sd <- sqrt(
      (n_total^2 - 1) / (12 * length(members)) *
        finite_population_correction
    )
    rank_z <- (mean(ranks[members]) - expected_mean_rank) / rank_sd
    u_statistic <- sum(ranks[members]) -
      length(members) * (length(members) + 1) / 2
    rank_biserial <-
      2 * u_statistic / (length(members) * length(nonmembers)) - 1

    data.frame(
      lipid_set = set_name,
      set_size = length(members),
      mean_signed_score = mean(scores[members]),
      median_signed_score = median(scores[members]),
      rank_z = rank_z,
      rank_biserial = rank_biserial,
      enrichment_direction = ifelse(rank_z > 0, "older_high", "older_low"),
      p_value = test$p.value,
      leading_lipids = paste(
        names(sort(abs(scores[members]), decreasing = TRUE))[seq_len(min(8, length(members)))],
        collapse = ";"
      ),
      stringsAsFactors = FALSE
    )
  })

  results <- do.call(rbind, results)
  if (is.null(results) || nrow(results) == 0) return(data.frame())
  results$q_value <- p.adjust(results$p_value, method = "BH")
  results[order(results$q_value, -abs(results$rank_z)), ]
}

ranked_results <- do.call(
  rbind,
  lapply(seq_len(nrow(peak_lookup)), function(i) {
    age <- peak_lookup$center_age[[i]]
    requested <- peak_lookup$requested_peak_age[[i]]
    one <- stats_peak[stats_peak$center_age == age &
      stats_peak$requested_peak_age == requested, ]
    result <- ranked_set_test(one, lipid_sets)
    if (nrow(result) == 0) return(NULL)
    result$requested_peak_age <- requested
    result$center_age <- age
    result
  })
)

if (is.null(ranked_results)) ranked_results <- data.frame()
write_csv(
  ranked_results,
  file.path(OUTPUT_DIR, "ranked_lipid_set_enrichment.csv")
)


# =============================================================================
# 7. Direction-specific over-representation analysis
# =============================================================================

ora_one <- function(stat_table, sets, direction) {
  background <- unique(stat_table$feature)
  targets <- unique(
    stat_table$feature[
      stat_table$peak_window_significant %in% TRUE &
        stat_table$direction == direction
    ]
  )

  if (length(targets) == 0) return(data.frame())

  results <- lapply(names(sets), function(set_name) {
    members <- intersect(unique(sets[[set_name]]), background)
    if (length(members) < MIN_SET_SIZE) return(NULL)

    a <- length(intersect(targets, members))
    b <- length(setdiff(targets, members))
    c_value <- length(setdiff(members, targets))
    d <- length(setdiff(background, union(targets, members)))
    contingency <- matrix(c(a, b, c_value, d), nrow = 2, byrow = TRUE)
    test <- fisher.test(contingency, alternative = "greater")

    data.frame(
      lipid_set = set_name,
      direction = direction,
      target_size = length(targets),
      set_size = length(members),
      overlap_size = a,
      odds_ratio = unname(test$estimate),
      p_value = test$p.value,
      target_selection = paste0(
        "BH q < ", FDR_THRESHOLD, " at the crest peak window"
      ),
      overlap_lipids = paste(intersect(targets, members), collapse = ";"),
      stringsAsFactors = FALSE
    )
  })

  results <- do.call(rbind, results)
  if (is.null(results) || nrow(results) == 0) return(data.frame())
  results$q_value <- p.adjust(results$p_value, method = "BH")
  results[order(results$q_value, -results$odds_ratio), ]
}

ora_results <- do.call(
  rbind,
  lapply(seq_len(nrow(peak_lookup)), function(i) {
    age <- peak_lookup$center_age[[i]]
    requested <- peak_lookup$requested_peak_age[[i]]
    one <- stats_peak[stats_peak$center_age == age &
      stats_peak$requested_peak_age == requested, ]

    do.call(
      rbind,
      lapply(c("older_high", "older_low"), function(direction) {
        result <- ora_one(one, lipid_sets, direction)
        if (nrow(result) == 0) return(NULL)
        result$requested_peak_age <- requested
        result$center_age <- age
        result
      })
    )
  })
)

if (is.null(ora_results)) ora_results <- data.frame()
write_csv(
  ora_results,
  file.path(OUTPUT_DIR, "ora_lipid_set_enrichment.csv")
)


# =============================================================================
# 8. Local analysis provenance and hand-off statistics
# =============================================================================
peak_display_contract <- data.frame(
  requested_peak_age = PEAK_AGES,
  age_label = paste0(format(PEAK_AGES, trim = TRUE), " y\ncrest peak"),
  stringsAsFactors = FALSE
)
write_csv(peak_display_contract, file.path(OUTPUT_DIR, "local_figure_age_contract.csv"))
coverage <- data.frame(
  n_features = nrow(annotations),
  n_standardized = sum(annotations$mapping_status != "unsupported"),
  n_unsupported = sum(annotations$mapping_status == "unsupported"),
  n_ambiguous_subclass = sum(annotations$mapping_status == "ambiguous_subclass"),
  mapping_rate = mean(annotations$mapping_status != "unsupported")
)
write_csv(coverage, file.path(OUTPUT_DIR, "annotation_coverage.csv"))
input_manifest <- c(
  "Owner: 03_lipid_pathway_enrichment.R (local analysis only)",
  paste0("DE-SWAN RDS: ", normalizePath(RDS_FILE, winslash = "/")),
  paste0("Crest table: ", CREST_FILE),
  paste0("Main-peak ages (candidates excluded): ", paste(PEAK_AGES, collapse = ", ")),
  paste0("Lipid selection BH FDR: ", FDR_THRESHOLD),
  paste0("Output directory: ", normalizePath(OUTPUT_DIR, winslash = "/")),
  paste0("Tables only: ", TABLES_ONLY),
  "Every tested feature is retained in deswan_peak_statistics.csv, regardless of mapping, q or sign.",
  "The LION preparer reads this table and applies its own website mapping audit."
)
writeLines(input_manifest, file.path(OUTPUT_DIR, "input_manifest.txt"))
writeLines(c(input_manifest,
  paste0("Run time: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  paste0("R version: ", R.version.string),
  paste0("Local ranked enrichment rows: ", nrow(ranked_results)),
  paste0("Local ORA rows: ", nrow(ora_results)),
  "Local ranked test: two-sided competitive Wilcoxon, signed z inside versus outside each set.",
  "Local ORA: one-sided Fisher exact test of significant lipids, separately by coefficient sign.",
  "Local background: all tested features; LION name exclusions do not alter these local tests.",
  "BH correction is per local table/age/direction; pooled cross-family FDR is not claimed.",
  "These feature-level competitive tests do not model between-lipid correlation.",
  "Local Wilcoxon results are not LION/web Kolmogorov-Smirnov results.",
  "Ontology analysis is prepared separately; no LION/BioPAN uploads are written or imported here.",
  "Differences in significance across peaks are not tests of between-peak differences."
), file.path(OUTPUT_DIR, "analysis_manifest.txt"))
writeLines(capture.output(sessionInfo()), file.path(OUTPUT_DIR, "session_info.txt"))
writeLines(enc2utf8(c(
  "# Local lipid-set analysis", "",
  "The local analysis in this directory is produced by 03_lipid_pathway_enrichment.R.",
  "deswan_peak_statistics.csv stores the statistics and annotations of all lipids at the main peaks; candidate peaks are excluded.",
  "ranked_lipid_set_enrichment.csv is a local two-sided Wilcoxon set test, not the LION two-tailed KS result.",
  "ora_lipid_set_enrichment.csv is a local Fisher test computed separately for increased and decreased lipids.",
  "--tables_only=true writes the local tables only; by default two panels (local sets and member lipids) are also drawn.", "",
  "LION website inputs are generated only by 03_prepare_lion_custom_statistics.R --mode=both.",
  "It reads the complete statistics table in this directory and exports one two-tailed ranking and two directional target lists per main peak.",
  "The preparation script also accepts --rds to convert DE-SWAN statistics directly without the local enrichment tests.",
  "Website results are plotted by 03_visualize_lion_results.R and 03_visualize_lion_target_list_results.R."
)), file.path(OUTPUT_DIR, "README_local_analysis.md"), useBytes = TRUE)
if (TABLES_ONLY) {
  message("Local analysis tables complete: ", normalizePath(OUTPUT_DIR, winslash = "/"))
  message("For LION/web inputs, run 03_prepare_lion_custom_statistics.R --mode=both.")
  quit(save = "no", status = 0L)
}

# =============================================================================
# 9. Local evidence figure and source data
# =============================================================================

figure_packages <- c(
  "ggplot2", "dplyr", "tidyr", "stringr", "patchwork",
  "svglite", "ragg", "jsonlite", "scales"
)
missing_figure_packages <- figure_packages[
  !vapply(figure_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_figure_packages) > 0) {
  stop(
    "Nature-ready figure output requires: ",
    paste(missing_figure_packages, collapse = ", ")
  )
}

palette_contract <- c(
  # Result-1 visual anchors: the same muted blue/coral pair used in the
  # cohort overview, with neutral ink kept sufficiently dark for print.
  negative = "#5B7EC9",
  positive = "#D35F62",
  neutral_dark = "#272727",
  neutral_mid = "#767676",
  neutral_light = "#D8D8D8"
)

result1_palette <- c(
  blue = "#5B7EC9",
  coral = "#D35F62",
  green = "#72A95A",
  purple = "#8A70B2",
  gold = "#C8A54E",
  peach_pale = "#F3DDD0",
  blue_pale = "#D9E7F3",
  green_pale = "#DDEBD5",
  purple_pale = "#E6DEEE",
  gold_pale = "#F8E8B8",
  paper = "#FFFDFC"
)

theme_nature <- function(base_size = 6.5, base_family = "Helvetica") {
  ggplot2::theme_classic(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      text = ggplot2::element_text(colour = "black"),
      axis.line = ggplot2::element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = ggplot2::element_line(linewidth = 0.35, colour = "black"),
      axis.title = ggplot2::element_text(size = base_size),
      axis.text = ggplot2::element_text(size = base_size - 0.5, colour = "black"),
      legend.title = ggplot2::element_text(size = base_size - 0.3),
      legend.text = ggplot2::element_text(size = base_size - 0.7),
      legend.key.height = grid::unit(2.8, "mm"),
      legend.key.width = grid::unit(3.5, "mm"),
      strip.background = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(size = base_size - 0.2, face = "bold"),
      plot.title = ggplot2::element_text(size = base_size + 0.5, face = "bold"),
      panel.grid.minor = ggplot2::element_blank(),
      plot.margin = ggplot2::margin(3, 4, 3, 3)
    )
}

pretty_set_label <- function(x) {
  x <- sub("^(Category|Class|Unsaturation|Chain length): ", "", x)
  x <- gsub("Total carbon", "Total C", x, fixed = TRUE)
  stringr::str_wrap(x, width = 30)
}

age_palette <- setNames(
  rep_len(c(palette_contract[["negative"]], palette_contract[["positive"]]), length(PEAK_AGES)),
  peak_display_contract$age_label
)

# Panel a: bounded competitive enrichment effects. Composite
# class-unsaturation terms remain in full source data but are removed from the
# main panel to reduce deterministic parent/child redundancy.
ranked_plot_source <- merge(
  ranked_results,
  peak_display_contract,
  by = "requested_peak_age",
  all.x = TRUE
)
ranked_plot_source$significant <- ranked_plot_source$q_value < FDR_THRESHOLD
ranked_plot_source$main_eligible <- !grepl(
  "^Class-unsaturation:", ranked_plot_source$lipid_set
)

set_priority <- aggregate(
  q_value ~ lipid_set,
  data = ranked_plot_source[ranked_plot_source$main_eligible, ],
  FUN = min
)
set_priority <- set_priority[order(set_priority$q_value, set_priority$lipid_set), ]
selected_sets <- head(set_priority$lipid_set, TOP_SETS)
panel_a_data <- ranked_plot_source[
  ranked_plot_source$lipid_set %in% selected_sets,
]
panel_a_data$set_label <- pretty_set_label(panel_a_data$lipid_set)
panel_a_order <- pretty_set_label(rev(selected_sets))
panel_a_data$set_label <- factor(panel_a_data$set_label, levels = panel_a_order)
panel_a_data$age_label <- factor(
  panel_a_data$age_label,
  levels = peak_display_contract$age_label
)
write_csv(
  ranked_plot_source,
  file.path(OUTPUT_DIR, "figure2a_ranked_set_source_data.csv")
)

p_a <- ggplot2::ggplot(
  panel_a_data,
  ggplot2::aes(
    x = rank_biserial,
    y = set_label,
    colour = age_label,
    shape = significant
  )
) +
  ggplot2::geom_vline(
    xintercept = 0,
    linewidth = 0.35,
    colour = palette_contract[["neutral_mid"]]
  ) +
  ggplot2::geom_point(
    position = ggplot2::position_dodge(width = 0.45),
    size = 2.1,
    stroke = 0.55
  ) +
  ggplot2::scale_colour_manual(values = age_palette, drop = FALSE) +
  ggplot2::scale_shape_manual(
    values = c(`TRUE` = 16, `FALSE` = 1),
    labels = c(`TRUE` = "FDR < 0.05", `FALSE` = "FDR >= 0.05")
  ) +
  ggplot2::scale_x_continuous(
    limits = c(-1, 1),
    breaks = c(-1, -0.5, 0, 0.5, 1),
    expand = ggplot2::expansion(mult = c(0.02, 0.02))
  ) +
  ggplot2::labs(
    x = "Rank-biserial shift versus other lipids",
    y = NULL,
    colour = NULL,
    shape = NULL,
    title = "Competitive lipid-set enrichment"
  ) +
  theme_nature() +
  ggplot2::theme(
    legend.position = "top",
    legend.justification = "left",
    panel.grid.major.x = ggplot2::element_line(
      linewidth = 0.25,
      colour = "grey90"
    )
  )

# Panel b: leading individual lipids. Selection is balanced across every
# peak-by-direction stratum and is based on peak-window q then |signed z|.
mapped_stats <- stats_peak[
  is.finite(stats_peak$signed_z_score) &
    !stats_peak$mapping_status %in% c("unsupported", "ambiguous_subclass"),
]
leading_features <- unique(unlist(lapply(
  split(
    mapped_stats,
    interaction(
      mapped_stats$requested_peak_age,
      mapped_stats$direction,
      drop = TRUE
    )
  ),
  function(x) {
    x <- x[
      order(
        x$q_value,
        -abs(x$signed_z_score),
        x$standard_name
      ),
    ]
    head(x$feature, TOP_LIPIDS_PER_DIRECTION)
  }
)))
panel_b_data <- mapped_stats[mapped_stats$feature %in% leading_features, ]
age_label_lookup <- setNames(
  peak_display_contract$age_label,
  as.character(peak_display_contract$requested_peak_age)
)
panel_b_data$age_label <- unname(
  age_label_lookup[as.character(panel_b_data$requested_peak_age)]
)
lipid_priority <- aggregate(
  abs(panel_b_data$signed_z_score),
  by = list(standard_name = panel_b_data$standard_name),
  FUN = max
)
names(lipid_priority)[[2]] <- "max_abs_z"
lipid_priority <- lipid_priority[
  order(lipid_priority$max_abs_z, lipid_priority$standard_name),
]
panel_b_data$standard_name <- factor(
  panel_b_data$standard_name,
  levels = lipid_priority$standard_name
)
panel_b_data$age_label <- factor(
  panel_b_data$age_label,
  levels = peak_display_contract$age_label
)
panel_b_data$significant <- panel_b_data$peak_window_significant %in% TRUE
write_csv(
  panel_b_data,
  file.path(OUTPUT_DIR, "figure2b_leading_lipids_source_data.csv")
)

z_limit <- max(abs(panel_b_data$signed_z_score), na.rm = TRUE)
p_b <- ggplot2::ggplot(
  panel_b_data,
  ggplot2::aes(x = age_label, y = standard_name, fill = signed_z_score)
) +
  ggplot2::geom_tile(colour = "white", linewidth = 0.35) +
  ggplot2::geom_point(
    data = panel_b_data[panel_b_data$significant, ],
    shape = 21,
    size = 1.45,
    stroke = 0.4,
    fill = "white",
    colour = "black"
  ) +
  ggplot2::scale_fill_gradient2(
    low = palette_contract[["negative"]],
    mid = "white",
    high = palette_contract[["positive"]],
    midpoint = 0,
    limits = c(-z_limit, z_limit),
    name = "Signed z"
  ) +
  ggplot2::scale_x_discrete(position = "top") +
  ggplot2::labs(
    x = NULL,
    y = NULL,
    title = "Leading lipid-level signals"
  ) +
  theme_nature() +
  ggplot2::theme(
    axis.line = ggplot2::element_blank(),
    axis.ticks = ggplot2::element_blank(),
    legend.position = "right",
    panel.grid = ggplot2::element_blank()
  )

save_nature_plot <- function(plot, stem, width_mm, height_mm, dpi = 600) {
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4

  svglite::svglite(
    paste0(stem, ".svg"),
    width = width_in,
    height = height_in,
    bg = "white"
  )
  print(plot)
  grDevices::dev.off()

  # Optional Cairo copy (NATURE_EXPORT_CAIRO_PDF=1); the default PDF uses R's
  # native Helvetica device.
  if (tolower(Sys.getenv("NATURE_EXPORT_CAIRO_PDF", unset = "0")) %in%
      c("1", "true", "yes")) {
    grDevices::cairo_pdf(
      paste0(stem, ".cairo.pdf"),
      width = width_in,
      height = height_in,
      family = "Helvetica",
      bg = "white"
    )
    print(plot)
    grDevices::dev.off()
  }

  grDevices::pdf(
    paste0(stem, ".pdf"),
    width = width_in,
    height = height_in,
    family = "Helvetica",
    bg = "white",
    useDingbats = FALSE,
    compress = TRUE
  )
  print(plot)
  grDevices::dev.off()

  ragg::agg_tiff(
    paste0(stem, ".tiff"),
    width = width_in,
    height = height_in,
    units = "in",
    res = dpi,
    compression = "lzw",
    background = "white"
  )
  print(plot)
  grDevices::dev.off()

  ragg::agg_png(
    paste0(stem, ".png"),
    width = width_in,
    height = height_in,
    units = "in",
    res = dpi,
    background = "white"
  )
  print(plot)
  grDevices::dev.off()

  message("Saved figure bundle: ", stem, ".{svg,pdf,tiff,png}")
}

main_figure <- p_a / p_b +
  patchwork::plot_layout(heights = c(1.15, 1)) +
  patchwork::plot_annotation(tag_levels = "a") &
  ggplot2::theme(plot.tag = ggplot2::element_text(
    family = "Helvetica", size = 8, face = "bold"))
main_height_mm <- 150

main_stem <- file.path(FIGURE_DIR, "figure2_lipid_pathway_evidence")

save_nature_plot(main_figure, main_stem, 180, main_height_mm)
save_nature_plot(
  p_a,
  file.path(FIGURE_DIR, "ranked_lipid_set_enrichment"),
  180,
  105
)
save_nature_plot(
  p_b,
  file.path(FIGURE_DIR, "leading_lipid_signed_z_heatmap"),
  88,
  105
)

figure_legend <- c(
  "Fig. 2 | Lipid remodeling at DE-SWAN crest peak windows.",
  "a, Competitive ranked enrichment of lipid classes and structural sets using all mapped lipids. Each lipid is ranked by its signed DE-SWAN z statistic at the exact crest peak window. Negative and positive values indicate lower and higher abundance in the older parcel, respectively. Filled points denote lipid-set BH FDR q < 0.05.",
  paste0(
    "b, Peak-window signed z-scores for leading lipids selected within each ",
    "peak-by-direction stratum. White circles mark lipids with BH q < ",
    FDR_THRESHOLD, " at that exact crest peak window."
  ),
  "DE-SWAN statistics were derived from covariate-adjusted local age-window models. Crest peak ages were read from the saved DE-SWAN crest table, and molecular selection used only the exact peak-age window, matching the transcriptomic analysis."
)
figure_legend <- figure_legend[nzchar(figure_legend)]
writeLines(
  figure_legend,
  file.path(OUTPUT_DIR, "figure2_lipid_pathway_legend.txt")
)

figure_qa <- c(
  paste0("Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  "Backend: R (ggplot2/patchwork); no cross-backend rendering.",
  paste0("Final main-figure size: 180 x ", main_height_mm, " mm."),
  "Exports: editable SVG/PDF and 600-dpi TIFF/PNG.",
  "Optional Cairo PDF: set NATURE_EXPORT_CAIRO_PDF=1; disabled by default.",
  paste0("Panel a source rows: ", nrow(ranked_plot_source), "; displayed sets: ", length(selected_sets), "."),
  paste0("Panel b source rows: ", nrow(panel_b_data), "; displayed lipids: ", length(unique(panel_b_data$standard_name)), "."),
  "No observations were sampled or removed for rendering.",
  "Main-set filtering removes composite class-unsaturation labels only for display; full results remain in source data.",
  "A significant result at one peak and non-significant result at another is not a direct test of between-peak differences."
)
writeLines(
  figure_qa,
  file.path(OUTPUT_DIR, "figure2_lipid_pathway_qa_notes.txt")
)


message(strrep("=", 72))
message("Lipid pathway enrichment complete")
message("Peak-age source: ", PEAK_SOURCE)
message(
  "Exact crest peak windows used for molecular selection: ",
  paste(peak_lookup$center_age, collapse = ", ")
)
message("Mapped lipids: ", coverage$n_standardized, "/", coverage$n_features)
message("Ranked enrichment rows: ", nrow(ranked_results))
message("ORA rows: ", nrow(ora_results))
message("Outputs: ", normalizePath(OUTPUT_DIR, winslash = "/", mustWork = TRUE))
message(strrep("=", 72))
