# Shared DE-SWAN peak selection and target-list preparation for the 03 scripts.

find_matching_crest_file <- function(rds_file) {
  if (length(rds_file) != 1L || is.na(rds_file) || !nzchar(rds_file)) {
    return(NA_character_)
  }
  rds_name <- basename(rds_file)
  if (!grepl("^lipidomics_deswan_breakpoint_.*\\.rds$", rds_name, ignore.case = TRUE)) {
    return(NA_character_)
  }
  suffix <- sub("^lipidomics_deswan_breakpoint_", "", rds_name, ignore.case = TRUE)
  suffix <- sub("\\.rds$", ".csv", suffix, ignore.case = TRUE)
  # Prefer the upstream main-peaks table. The full crest table is a fallback
  # only when its explicit main/candidate classification can be applied.
  paths <- file.path(dirname(rds_file), paste0(c("lipidomics_main_peaks_", "lipidomics_crest_"), suffix))
  paths <- paths[file.exists(paths)]
  if (length(paths)) paths[[1L]] else NA_character_
}

resolve_crest_peaks <- function(peaks_argument, crest_argument, rds_file) {
  crest_file <- as.character(crest_argument)[[1]]
  auto_crest <- is.na(crest_file) || !nzchar(trimws(crest_file)) ||
    tolower(trimws(crest_file)) == "auto"
  if (auto_crest) {
    crest_file <- find_matching_crest_file(rds_file)
  } else {
    crest_file <- trimws(crest_file)
  }

  if (is.na(crest_file) || !nzchar(crest_file) || !file.exists(crest_file)) {
    stop(
      "Matching lipidomics main-peak table not found next to the DE-SWAN RDS. ",
      "Run 01-plot_deswan_omics_one.R on the saved RDS first, or supply ",
      "--crest=/path/to/lipidomics_main_peaks_window_7p5.csv."
    )
  }

  crest_table <- read.csv(
    crest_file,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  required_crest_columns <- c("crest_id", "peak_age", "peak_n")
  missing_crest_columns <- setdiff(required_crest_columns, names(crest_table))
  if (length(missing_crest_columns) > 0) {
    stop(
      "Crest table is missing required columns: ",
      paste(missing_crest_columns, collapse = ", "), ". File: ", crest_file
    )
  }
  if (nrow(crest_table) == 0L) {
    stop("Crest table contains no peaks: ", crest_file)
  }

  for (column in required_crest_columns) {
    crest_table[[column]] <- suppressWarnings(
      as.numeric(as.character(crest_table[[column]]))
    )
  }
  valid_rows <- apply(
    crest_table[, required_crest_columns, drop = FALSE],
    1,
    function(x) all(is.finite(x))
  )
  if (!all(valid_rows)) {
    stop("Crest table contains non-numeric or missing peak values: ", crest_file)
  }
  if (anyDuplicated(crest_table$crest_id)) {
    stop("Crest table contains duplicate crest_id values.")
  }
  if (anyDuplicated(crest_table$peak_age)) {
    stop("Crest table contains duplicate peak_age values.")
  }
  crest_table <- lion_main_peak_rows(crest_table, age_column = "peak_age")

  crest_table <- crest_table[
    order(crest_table$crest_id, crest_table$peak_age),
    ,
    drop = FALSE
  ]

  peaks_argument <- trimws(peaks_argument)
  use_all_main_peaks <- !nzchar(peaks_argument) ||
    tolower(peaks_argument) == "auto"
  if (!use_all_main_peaks) {
    requested_peaks <- suppressWarnings(
      as.numeric(strsplit(peaks_argument, ",", fixed = TRUE)[[1]])
    )
    if (length(requested_peaks) == 0 || any(!is.finite(requested_peaks))) {
      stop("--peaks must be 'auto' or comma-separated main-peak ages.")
    }
    if (anyDuplicated(requested_peaks)) stop("--peaks contains duplicate ages.")
    crest_index <- match(requested_peaks, crest_table$peak_age)
    if (anyNA(crest_index)) {
      stop(
        "--peaks may only select classified main peaks; candidates are excluded. ",
        "Available main peaks: ", paste(crest_table$peak_age, collapse = ", ")
      )
    }
    crest_table <- crest_table[crest_index, , drop = FALSE]
  }

  normalized_crest <- normalizePath(
    crest_file,
    winslash = "/",
    mustWork = TRUE
  )
  list(
    crests = crest_table,
    ages = crest_table$peak_age,
    source = paste0("main peaks from: ", normalized_crest),
    crest_file = normalized_crest
  )
}

lion_resolve_parcel_width <- function(rds_file, value = "auto") {
  if (tolower(trimws(value)) == "auto") {
    pattern <- "^lipidomics_deswan_breakpoint_window_([0-9]+([p.][0-9]+)?)\\.rds$"
    if (!grepl(pattern, basename(rds_file), ignore.case = TRUE)) {
      stop("Cannot infer the parcel half-width from the RDS filename; supply --parcel_width.")
    }
    value <- gsub("p", ".", sub(pattern, "\\1", basename(rds_file), ignore.case = TRUE),
                  fixed = TRUE)
  }
  width <- suppressWarnings(as.numeric(value))
  if (length(width) != 1L || !is.finite(width) || width <= 0) {
    stop("--parcel_width must be 'auto' or a positive half-width in years.")
  }
  width
}

lion_measured_features <- function(data_file) {
  metadata <- c(
    "participant_id", "cohort", "research_stage", "array_index",
    "collection_date", "collection_timestamp", "timezone", "age", "sex",
    "bmi", "batch", "collection_datetime", "collection_time", "omics",
    "pred_age", "year_of_birth", "month_of_birth"
  )
  columns <- names(read.csv(data_file, nrows = 0L, check.names = FALSE))
  features <- setdiff(columns, metadata)
  if (!length(features) || anyDuplicated(columns)) {
    stop("Expected unique lipid columns in the quantitative data: ", data_file)
  }
  features
}

lion_deswan_statistics <- function(deswan) {
  if (!is.list(deswan) || is.null(deswan$p)) stop("Missing DE-SWAN p table.")
  p <- as.data.frame(deswan$p, stringsAsFactors = FALSE)
  b <- if (!is.null(deswan$coeff)) deswan$coeff else deswan$coefficients
  if (is.null(b)) stop("Missing DE-SWAN coefficient table.")
  b <- as.data.frame(b, stringsAsFactors = FALSE)
  if (!all(c("variable", "window.center", "factor", "pvalue") %in% names(p)) ||
      !all(c("variable", "window.center", "factor", "coefficient") %in% names(b))) {
    stop("Unexpected DE-SWAN p/coefficient schema.")
  }
  p <- p[p$factor %in% "qt", c("variable", "window.center", "pvalue"), drop = FALSE]
  b <- b[b$factor %in% "qt", c("variable", "window.center", "coefficient"), drop = FALSE]
  names(p) <- c("feature", "center_age", "p_value")
  names(b) <- c("feature", "center_age", "coefficient")
  numeric_value <- function(x) suppressWarnings(as.numeric(as.character(x)))
  p$center_age <- numeric_value(p$center_age)
  b$center_age <- numeric_value(b$center_age)
  p$p_value <- numeric_value(p$p_value)
  b$coefficient <- numeric_value(b$coefficient)
  if (!nrow(p) || any(!is.finite(p$center_age)) || any(!is.finite(b$center_age)) ||
      anyNA(p$feature) || anyNA(b$feature) ||
      any(!nzchar(p$feature)) || any(!nzchar(b$feature))) {
    stop("Invalid DE-SWAN feature or center_age.")
  }
  if (anyDuplicated(p[c("feature", "center_age")]) ||
      anyDuplicated(b[c("feature", "center_age")])) {
    stop("Duplicate DE-SWAN feature-by-center rows.")
  }
  if (any(!is.na(p$p_value) &
          (!is.finite(p$p_value) | p$p_value < 0 | p$p_value > 1))) {
    stop("p_value must be in [0, 1] or missing.")
  }
  # Correct the complete age-effect family before mapping, direction selection,
  # or dropping lipids whose coefficient could not be estimated.
  p$q_value <- ave(p$p_value, p$center_age, FUN = function(x) p.adjust(x, "BH"))
  merge(p, b, by = c("feature", "center_age"), all.x = TRUE, sort = FALSE)
}

export_lion_target_lists <- function(stat_table, annotation, output_dir, ages,
                                     fdr = 0.05, source_files = character(),
                                     include_combined = FALSE) {
  required <- c("feature", "requested_peak_age", "center_age", "q_value", "coefficient")
  if (!all(required %in% names(stat_table))) {
    stop("Target-list statistics require: ", paste(required, collapse = ", "))
  }
  if (!all(c("feature", "standard_name", "mapping_status") %in% names(annotation))) {
    stop("Background requires feature, standard_name and mapping_status.")
  }
  if (length(fdr) != 1L || !is.finite(fdr) || fdr <= 0 || fdr >= 1) {
    stop("fdr must be between 0 and 1.")
  }
  stat_table <- lion_main_peak_rows(stat_table)
  lion_require_main_ages(ages, unique(stat_table$requested_peak_age))
  stat_table <- stat_table[stat_table$requested_peak_age %in% ages, , drop = FALSE]
  for (column in c("requested_peak_age", "center_age", "q_value", "coefficient")) {
    stat_table[[column]] <- suppressWarnings(as.numeric(as.character(stat_table[[column]])))
  }
  if (any(!is.na(stat_table$q_value) &
          (!is.finite(stat_table$q_value) | stat_table$q_value < 0 | stat_table$q_value > 1))) {
    stop("q_value must be in [0, 1] or missing.")
  }
  annotation$feature <- as.character(annotation$feature)
  if (!nrow(annotation) || anyNA(annotation$feature) ||
      any(!nzchar(trimws(annotation$feature))) || anyDuplicated(annotation$feature)) {
    stop("Background requires unique, nonempty feature identifiers.")
  }
  annotation <- lion_web_mapping(annotation)
  annotation$uses_original_name <- FALSE
  annotation$submitted_name <- ifelse(annotation$lion_web_eligible, annotation$standard_name, NA_character_)
  if (any(grepl("[\t\r\n]", annotation$submitted_name))) {
    stop("Lipid identifiers cannot contain tabs or newlines.")
  }
  annotation <- annotation[order(annotation$feature, method = "radix"), , drop = FALSE]
  background <- sort(unique(annotation$submitted_name[annotation$lion_web_eligible]), method = "radix")
  if (!length(background)) stop("No eligible LION background lipids after mapping exclusions.")
  selections <- c("older_high", "older_low", if (isTRUE(include_combined)) "all_significant")

  # Validate all peaks before writing. The submitted background is the mapped
  # subset of ALL measured lipids, independent of significance or direction.
  # Keep excluded features in the audit and retain the original full-family q.
  prepared <- lapply(ages, function(age) {
    one <- stat_table[is.finite(stat_table$requested_peak_age) &
      abs(stat_table$requested_peak_age - age) < 1e-8, , drop = FALSE]
    if (!nrow(one)) stop("No peak statistics for age ", age, ".")
    centers <- unique(one$center_age)
    if (length(centers) != 1L || any(!is.finite(centers)) || abs(centers[[1L]] - age) > 1e-8) {
      stop("Expected the exact DE-SWAN center_age for peak ", age, ".")
    }
    if (anyNA(one$feature) || anyDuplicated(one$feature)) {
      stop("Missing/duplicate feature at peak ", age, ".")
    }
    outside <- setdiff(one$feature, annotation$feature)
    if (length(outside)) {
      stop("Peak features absent from full measured background: ", paste(outside, collapse = ", "))
    }
    index <- match(annotation$feature, one$feature)
    audit <- annotation
    audit$requested_peak_age <- age
    audit$is_main <- TRUE
    audit$peak_class <- "main"
    audit$center_age <- centers[[1L]]
    audit$has_peak_statistics <- !is.na(index)
    audit$q_value <- one$q_value[index]
    audit$coefficient <- one$coefficient[index]
    audit$all_significant <- is.finite(audit$q_value) & audit$q_value < fdr
    audit$older_high <- audit$all_significant &
      is.finite(audit$coefficient) & audit$coefficient > 0
    audit$older_low <- audit$all_significant &
      is.finite(audit$coefficient) & audit$coefficient < 0
    audit
  })
  if (length(source_files) && any(!file.exists(source_files))) stop("Missing provenance source file.")

  upload_dir <- file.path(output_dir, "lion_web_target_list_inputs")
  dir.create(upload_dir, recursive = TRUE, showWarnings = FALSE)
  upload_dir <- normalizePath(upload_dir, winslash = "/", mustWork = TRUE)
  background_file <- file.path(upload_dir, "lion_background_matched_lipids.txt")
  write_list <- function(x, path) writeLines(enc2utf8(x), path, useBytes = TRUE)
  write_list(background, background_file)
  write.csv(annotation, file.path(upload_dir, "background_mapping_audit.csv"), row.names = FALSE, na = "")
  runs <- list()
  for (i in seq_along(ages)) {
    age <- ages[[i]]
    label <- lion_age_label(age)
    audit <- prepared[[i]]
    for (selection in selections) {
      chosen <- audit[[selection]] & audit$lion_web_eligible
      target <- sort(unique(audit$submitted_name[chosen]), method = "radix")
      stopifnot(all(target %in% background), !anyDuplicated(target))
      input_file <- file.path(upload_dir, paste0("lion_age_", label, "_", selection, "_target.txt"))
      write_list(target, input_file)
      runs[[length(runs) + 1L]] <- data.frame(
        analysis_mode = "target-list", requested_peak_age = age,
        analysis_role = if (selection == "all_significant") "combined_reference" else "supplementary_directional_ORA",
        is_main = TRUE, peak_class = "main",
        center_age = audit$center_age[[1L]], selection = selection,
        fdr_threshold = fdr,
        selection_rule = paste0("LION eligible; full-family BH q < fdr; ",
          switch(selection, older_high = "beta > 0", older_low = "beta < 0", all_significant = "either sign")),
        measured_features = nrow(annotation), background_lipids = length(background),
        features_with_finite_q = sum(is.finite(audit$q_value)),
        significant_features = sum(chosen), target_lipids = length(target),
        significant_measured_features = sum(audit[[selection]]),
        excluded_mapping_features = sum(!audit$lion_web_eligible),
        excluded_significant_features = sum(audit[[selection]] & !audit$lion_web_eligible),
        original_name_background_features = sum(annotation$uses_original_name),
        original_name_target_features = sum(chosen & audit$uses_original_name),
        recommended_run = selection != "all_significant" && length(target) > 0L,
        run_status = if (length(target)) "ready" else "empty_target_skip",
        input_file = input_file, input_md5 = unname(tools::md5sum(input_file)),
        background_file = background_file, background_md5 = unname(tools::md5sum(background_file)),
        result_file = file.path(upload_dir, paste0("lion_target_list_age_", label, "_", selection, "_result.csv")),
        stringsAsFactors = FALSE
      )
      message("Saved target-list: age ", age, " / ", selection, " = ", length(target),
              " lipids; shared background = ", length(background))
    }
  }
  run_manifest <- do.call(rbind, runs)
  audit <- do.call(rbind, prepared)
  write.csv(run_manifest, file.path(upload_dir, "lion_target_list_run_manifest.csv"), row.names = FALSE)
  write.csv(audit, file.path(upload_dir, "peak_feature_selection_audit.csv"), row.names = FALSE, na = "")
  write.csv(stat_table, file.path(upload_dir, "deswan_target_list_statistics.csv"), row.names = FALSE, na = "")
  sources <- data.frame(
    source_file = normalizePath(source_files, winslash = "/", mustWork = TRUE),
    source_md5 = unname(tools::md5sum(source_files)), stringsAsFactors = FALSE
  )
  write.csv(sources, file.path(upload_dir, "source_files.csv"), row.names = FALSE)
  readme_file <- file.path(upload_dir, "README_LION_TARGET_LIST.md")
  primary <- run_manifest[run_manifest$selection != "all_significant", ]
  write_list(c(
    "# LION/web supplementary analysis: increased and decreased target lists with a shared matchable background", "",
    "Only main peaks flagged upstream (is_main=TRUE / peak_class=main) are analysed; candidate peaks are not read.", "",
    "1. Open https://www.lipidontology.com/ and enter target-list mode.",
    "2. Paste the entire content of older_high_target.txt of one peak into the upper target-list box.",
    "3. Paste the entire content of lion_background_matched_lipids.txt into the lower background-list box.",
    "4. Submit, check the lipid-name matching table and save the actual matching rate, unmatched names and complete enrichment results.",
    "5. Repeat with older_low_target.txt and the same background; keep website options identical for both directions.",
    "6. With several main peaks, repeat for each; analyse main peaks only and do not merge directions or ages.", "",
    "All input TXT files have no header, statistics or quotes, with one lipid identifier per line. The background includes the foreground.",
    "This mode needs no high-to-low setting; the website uses a one-sided Fisher enrichment test of foreground against background.", "",
    "| Peak age | Foreground TXT | Significant features | Deduplicated foreground | Deduplicated background |",
    "|---|---|---|---|---|",
    vapply(seq_len(nrow(primary)), function(i) paste0(
      "| ", primary$requested_peak_age[[i]], " | ", basename(primary$input_file[[i]]),
      " | ", primary$significant_features[[i]], " | ", primary$target_lipids[[i]],
      " | ", primary$background_lipids[[i]], " |"
    ), character(1L)), "",
    paste0("Foreground threshold: BH q < ", fdr,
           " at the single DE-SWAN centre of the peak; older_high also requires beta > 0 and older_low beta < 0."),
    "When reading the RDS, BH is applied to all qt tests at that centre before significant molecules are selected; no re-adjustment is made within the significant or mappable subset.",
    "When reading an exported CSV, its q_value is used directly; --fdr controls the current selection and does not depend on the legacy peak_window_significant column.",
    "The background is drawn from all lipid columns of the lipid quantification table with matchable names, including non-significant lipids; it is not filtered by direction or coefficient.",
    "Name filtering is shared with ranking: unsupported or structurally ambiguous names and the CoQ10, acca and pce names reported as unmatched by the website are excluded.",
    "Annotations and the statistical audit of all measured lipids are kept; lipid-level BH q is not recomputed among the remaining matchable lipids.",
    "Local name conversion is not actual LION matching. Unrecognized entries contribute no ontology annotation; the effective set is the one in the website matching report.",
    "Each submission name is kept once: it enters the combined foreground if any original feature is significant; increased and decreased lists can overlap through duplicated names.",
    "older_high / older_low are the recommended supplementary analyses. Empty foregrounds are flagged empty_target_skip and should not be submitted.",
    "Only older_high / older_low are exported by default; a combined foreground reference is exported only with --include_combined_target=true.",
    "The primary analysis is one two-tailed ranking of all matchable lipids (high-to-low + two-tailed); this mode adds a directional over-representation analysis of significant lipids.",
    "q values of different website runs are adjusted separately; they are not interpreted as an FDR controlled across directions or modes, and directions are not reported selectively by significance.", "",
    "Name downloaded results after result_file in lion_target_list_run_manifest.csv.",
    "These are target-list results; plot them with 03_visualize_lion_target_list_results.R, which reads ages from the manifest in this directory by default.",
    "03_visualize_lion_results.R reads ranking results; do not replace the ranking CSV with results from this directory.",
    "Lipid-level q and website ontology-term enrichment q are tests at different levels. Significant versus non-significant across peaks is not a test of a difference between peaks.", "",
    "background_mapping_audit.csv records each original lipid and the name actually submitted; peak_feature_selection_audit.csv records the basis of selection.",
    "source_files.csv records input paths and MD5; deswan_target_list_statistics.csv stores the statistics used in this run.",
    "Official description of modes and nomenclature: https://www.lipidontology.com/faq.html"
  ), readme_file)
  invisible(list(runs = run_manifest, readme = readme_file, audit = audit))
}
