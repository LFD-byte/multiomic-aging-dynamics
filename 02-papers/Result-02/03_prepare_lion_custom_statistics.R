#!/usr/bin/env Rscript
# LION/web INPUT PREPARATION ONLY. No local enrichment tests or plotting.
# 03_lipid_pathway_enrichment.R owns local enrichment and the complete peak CSV.
# This entry point can read that CSV OR convert the original DE-SWAN RDS.
# Only classified main peaks and their exact centers are eligible.
#
# Default --mode=both generates:
#   1. One full signed-z ranking per peak (including nonsignificant lipids).
#      LION/web: from high to low + two-tailed. No reverse-score ranking.
#   2. Two supplementary target lists: q < --fdr with beta > 0 / beta < 0,
#      sharing the full matched background. --include_combined_target=true
#      optionally exports a combined foreground for reference.
# --mode=ranking / --mode=target_list prepares only the selected web mode.
#
# Ranking TXT: headerless LIPID_NAME<TAB>SIGNED_Z, NOT an abundance matrix.
# Positive z means older/right parcel higher at that local window.
# --fdr changes target selection and audit counts, NEVER ranking membership.
#
# Rscript 03_prepare_lion_custom_statistics.R --mode=both
# Rscript 03_prepare_lion_custom_statistics.R --mode=both \
#   --rds=outputs/deswan-window_7.5/lipidomics_deswan_breakpoint_window_7p5.rds
# Default CSV: outputs/lipid_pathway/deswan_peak_statistics.csv.
# With --rds, --crest=auto locates its matching main_peaks CSV. With CSV input,
# classification comes from its columns/adjacent peak_age_lookup.csv or --crest.
# --ages=auto uses only main peaks; explicit candidate ages are rejected too.
# Defaults: CSV output beside CSV; RDS output outputs/lipid_pathway.
# Downloaded results belong to the two separate 03_visualize_lion_* scripts.
options(stringsAsFactors = FALSE)

script_argument <- grep("^--file=", commandArgs(), value = TRUE)
source_paths <- as.character(unlist(lapply(sys.frames(), function(frame) frame$ofile)))
source_paths <- source_paths[basename(source_paths) == "03_prepare_lion_custom_statistics.R"]
script_path <- if (length(source_paths)) {
  normalizePath(tail(source_paths, 1L), mustWork = TRUE)
} else if (length(script_argument) > 0L) {
  normalizePath(sub("^--file=", "", script_argument[[1L]]), mustWork = FALSE)
} else {
  candidates <- c(
    "03_prepare_lion_custom_statistics.R",
    file.path("02-papers", "Result-02", "03_prepare_lion_custom_statistics.R")
  )
  candidates <- candidates[file.exists(candidates)]
  if (!length(candidates)) stop("Cannot locate 03_prepare_lion_custom_statistics.R.")
  normalizePath(candidates[[1L]], mustWork = TRUE)
}
script_dir <- dirname(script_path)
source(file.path(script_dir, "lion_custom_statistics_helpers.R"), encoding = "UTF-8")
source(file.path(script_dir, "lipid_annotation_helpers.R"), encoding = "UTF-8")
source(file.path(script_dir, "lion_target_list_helpers.R"), encoding = "UTF-8")

parse_cli <- function(args) {
  paths <- lion_default_paths(script_dir)
  defaults <- list(
    input = paths$statistics,
    output = "auto",
    ages = "auto",
    mode = "both",
    include_combined_target = "false",
    fdr = "0.05",
    rds = "",
    crest = "auto",
    data = paths$data
  )

  for (arg in args) {
    if (!grepl("^--[^=]+=", arg)) stop("Expected --key=value: ", arg)
    pieces <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    key <- pieces[[1L]]
    value <- paste(pieces[-1L], collapse = "=")
    if (!key %in% names(defaults)) stop("Unknown LION-preparation option: --", key)
    defaults[[key]] <- value
  }
  defaults
}

args <- parse_cli(commandArgs(trailingOnly = TRUE))
if (!args$mode %in% c("both", "ranking", "target_list")) {
  stop("--mode must be both, ranking, or target_list.")
}
fdr <- suppressWarnings(as.numeric(args$fdr))
if (!is.finite(fdr) || fdr <= 0 || fdr >= 1) stop("--fdr must be between 0 and 1.")
if (!tolower(args$include_combined_target) %in% c("true", "false")) {
  stop("--include_combined_target must be true or false.")
}
include_combined_target <- tolower(args$include_combined_target) == "true"
use_rds <- nzchar(trimws(args$rds))
if (use_rds && any(grepl("^--input=", commandArgs(trailingOnly = TRUE)))) {
  stop("Choose one statistics source: --input=CSV or --rds=RDS, not both.")
}
source_file <- if (use_rds) args$rds else args$input
if (!file.exists(source_file)) stop("Input file not found: ", source_file,
  ". Run 03_lipid_pathway_enrichment.R --tables_only=true first, or supply --rds.")
if (identical(args$output, "auto")) {
  args$output <- if (use_rds) lion_default_paths(script_dir, args$rds)$output else dirname(args$input)
}
crest_file <- character()
if (use_rds) {
  peak_config <- resolve_crest_peaks("auto", args$crest, args$rds)
  peak_metadata <- peak_config$crests
  crest_file <- peak_config$crest_file
} else {
  peak_metadata <- lion_main_peak_metadata(dirname(args$input), args$crest)
  if (!is.null(peak_metadata)) {
    crest_file <- if (tolower(trimws(args$crest)) == "auto") {
      file.path(dirname(args$input), "peak_age_lookup.csv")
    } else args$crest
  }
}

stats <- if (use_rds) {
  x <- lion_deswan_statistics(readRDS(source_file))
  x$requested_peak_age <- x$center_age
  merge(x, annotate_lipids(unique(x$feature)), by = "feature", all.x = TRUE, sort = FALSE)
} else read.csv(source_file, check.names = FALSE, stringsAsFactors = FALSE)
required_columns <- if (args$mode == "target_list") c(
  "feature", "requested_peak_age", "center_age", "q_value", "coefficient"
) else c(
  "standard_name", "requested_peak_age", "coefficient", "p_value",
  "mapping_status"
)
if (args$mode == "both") required_columns <- union(required_columns,
  c("feature", "center_age", "q_value"))
missing_columns <- setdiff(required_columns, names(stats))
if (length(missing_columns) > 0L) {
  stop("Missing required columns: ", paste(missing_columns, collapse = ", "))
}

stats$requested_peak_age <- suppressWarnings(as.numeric(stats$requested_peak_age))
stats$coefficient <- suppressWarnings(as.numeric(stats$coefficient))
if ("p_value" %in% names(stats)) stats$p_value <- suppressWarnings(as.numeric(stats$p_value))
stats <- lion_main_peak_rows(stats, lookup = peak_metadata)

available_ages <- sort(unique(
  stats$requested_peak_age[is.finite(stats$requested_peak_age)]
))
if (length(available_ages) == 0L) {
  stop("No finite peak ages were found in requested_peak_age: ", source_file)
}

age_argument <- trimws(args$ages)
if (!nzchar(age_argument) || identical(tolower(age_argument), "auto")) {
  requested_ages <- available_ages
  age_source <- "classified main peaks in requested_peak_age"
} else {
  age_tokens <- trimws(strsplit(age_argument, ",", fixed = TRUE)[[1L]])
  requested_ages <- suppressWarnings(as.numeric(age_tokens))
  if (
    length(age_tokens) == 0L || any(!nzchar(age_tokens)) ||
      any(!is.finite(requested_ages))
  ) {
    stop(
      "--ages must be 'auto' or contain comma-separated numeric ages, ",
      "for example --ages=48."
    )
  }
  if (anyDuplicated(requested_ages)) stop("--ages contains duplicate ages.")
  absent_ages <- requested_ages[
    !vapply(
      requested_ages,
      function(age) any(abs(available_ages - age) < 1e-8),
      logical(1L)
    )
  ]
  if (length(absent_ages) > 0L) {
    stop(
      "Requested age(s) are not eligible main peaks: ",
      paste(absent_ages, collapse = ", "),
      ". Available main-peak age(s): ", paste(available_ages, collapse = ", "), "."
    )
  }
  age_source <- if (length(crest_file)) paste0("main peaks from: ", crest_file) else "explicit main-peak subset"
}

message(
  "Main-peak ages: ", paste(requested_ages, collapse = ", "),
  " (", age_source, ")"
)
if (args$mode %in% c("both", "target_list") && !file.exists(args$data)) {
  stop("Full measured background data not found: ", args$data)
}
dir.create(args$output, recursive = TRUE, showWarnings = FALSE)
lion_exports <- target_exports <- NULL

if (args$mode %in% c("both", "target_list")) {
  if (!file.exists(args$data)) stop("Full lipid background data not found: ", args$data)
  background <- annotate_lipids(lion_measured_features(args$data))
  # RDS mode uses exact centers and recalculates BH over all tested lipids.
  # CSV mode uses the saved q_value, not a potentially stale significance flag.
  target_stats <- stats[stats$requested_peak_age %in% requested_ages, , drop = FALSE]
  target_exports <- export_lion_target_lists(
    target_stats, background, args$output, requested_ages, fdr = fdr,
    source_files = c(source_file, args$data, crest_file),
    include_combined = include_combined_target
  )
  message("Target-list instructions: ", target_exports$readme)
}
if (args$mode %in% c("both", "ranking")) {
  lion_exports <- export_lion_custom_statistics(
    stats, output_dir = args$output, ages = requested_ages, fdr = fdr
  )
  message("Ranking instructions (high-to-low + two-tailed, once per main peak): ", lion_exports$readme)
}
source_files <- unique(c(source_file, crest_file,
  if (args$mode %in% c("both", "target_list")) args$data))
source_files <- source_files[!is.na(source_files) & nzchar(source_files)]
write.csv(data.frame(
  source_file = normalizePath(source_files, winslash = "/", mustWork = TRUE),
  source_md5 = unname(tools::md5sum(source_files)),
  statistics_source_type = if (use_rds) "RDS" else "CSV",
  preparation_mode = args$mode,
  generated_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
), file.path(args$output, "lion_preparation_sources.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(args$output, "lion_preparation_session_info.txt"))
write_lion_analysis_workflow(args$output, lion_exports$runs, target_exports$runs)
message("LION inputs prepared. Enrichment is run on LION/web using the recorded settings.")
