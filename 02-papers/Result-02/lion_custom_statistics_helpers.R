# Shared by the 03 lipid pathway, LION export and visualization scripts.
# Primary ranking uses signed z once per peak: high-to-low, two-tailed KS.
# Only upstream main peaks are eligible. Both web modes use the same mapping
# policy; the measured background and lipid-level BH correction stay intact.

lion_default_paths <- function(script_dir, rds_file = NULL) {
  if (is.null(rds_file)) {
    rds_file <- file.path(script_dir, "outputs", "deswan-window_7.5",
                          "lipidomics_deswan_breakpoint_window_7p5.rds")
  }
  output <- file.path(script_dir, "outputs", "lipid_pathway")
  list(
    rds = rds_file,
    data = file.path(script_dir, "..", "01-interim_data", "omics_data",
                     "lipidomics_with_metadata.csv"),
    output = output,
    statistics = file.path(output, "deswan_peak_statistics.csv"),
    figure = file.path(output, "figures"),
    target_input = file.path(output, "lion_web_target_list_inputs")
  )
}

lion_main_peak_metadata <- function(output_dir, crest = "auto") {
  path <- if (tolower(trimws(crest)) == "auto") {
    file.path(output_dir, "peak_age_lookup.csv")
  } else crest
  if (!file.exists(path)) {
    if (tolower(trimws(crest)) == "auto") return(NULL)
    stop("Main-peak metadata file not found: ", path)
  }
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}

lion_main_peak_rows <- function(x, age_column = "requested_peak_age", lookup = NULL) {
  if (!age_column %in% names(x)) stop("Missing peak age column: ", age_column)
  x[[age_column]] <- suppressWarnings(as.numeric(as.character(x[[age_column]])))
  if (any(!is.finite(x[[age_column]]))) stop("Peak ages must be finite.")
  has_flag <- "is_main" %in% names(x)
  has_class <- "peak_class" %in% names(x)
  if (!has_flag && !has_class && is.null(lookup)) {
    stop("Main-peak classification is required (is_main or peak_class). ",
         "Regenerate the peak statistics/run manifest or provide classified peak metadata.")
  }
  keep <- rep(TRUE, nrow(x))
  if (has_flag) {
    keep <- keep & tolower(trimws(as.character(x$is_main))) %in% c("true", "1")
  }
  if (has_class) {
    keep <- keep & tolower(trimws(as.character(x$peak_class))) %in% "main"
  }
  if (!is.null(lookup)) {
    lookup_age <- if ("peak_age" %in% names(lookup)) "peak_age" else "requested_peak_age"
    main <- lion_main_peak_rows(lookup, age_column = lookup_age)
    keep <- keep & x[[age_column]] %in% main[[lookup_age]]
  }
  if (!any(keep)) stop("No classified main peaks remain; candidate peaks are excluded.")
  x <- x[keep, , drop = FALSE]
  x$is_main <- TRUE
  x$peak_class <- "main"
  x
}

lion_require_main_ages <- function(ages, main_ages) {
  if (!length(ages) || any(!is.finite(ages)) || anyDuplicated(ages)) {
    stop("ages must contain unique, finite main-peak ages.")
  }
  outside <- setdiff(ages, main_ages)
  if (length(outside)) {
    stop("Only classified main peaks may be selected. Not eligible: ",
         paste(outside, collapse = ", "), "; main peaks: ", paste(main_ages, collapse = ", "))
  }
  invisible(ages)
}

lion_age_label <- function(age) {
  gsub("[^A-Za-z0-9]+", "_", format(age, scientific = FALSE, trim = TRUE))
}

lion_signed_z <- function(coefficient, p_value) {
  if (any(!is.na(p_value) & (!is.finite(p_value) | p_value < 0 | p_value > 1))) {
    stop("p_value must be finite and between 0 and 1 (or missing).")
  }
  sign(coefficient) * qnorm(
    pmax(p_value, .Machine$double.xmin) / 2, lower.tail = FALSE
  )
}

# Website feedback supplied by the user on 2026-09-10: 169/177 identifiers
# matched. Preserve local annotations, but exclude these observed failures from
# BOTH web modes. This is a naming/coverage rule, never a significance filter.
lion_web_mapping <- function(x) {
  if (!all(c("standard_name", "mapping_status") %in% names(x))) {
    stop("LION mapping requires standard_name and mapping_status.")
  }
  x$standard_name <- trimws(as.character(x$standard_name))
  status <- as.character(x$mapping_status)
  reason <- rep("", nrow(x))
  reason[is.na(status) | !status %in% c("sum_composition", "molecular_species", "common_name")] <-
    "unsupported_local_mapping"
  reason[status %in% "ambiguous_subclass"] <- "ambiguous_subclass"
  reason[is.na(x$standard_name) | !nzchar(x$standard_name)] <- "missing_standard_name"
  unmatched <- tolower(c(
    "acca_c10.0", "acca_c16.0", "acca_c18.1", "coenzyme_q10", "CoQ10",
    "pce_34.0", "pce_34.1", "pce_38.4", "pce_38.5",
    "PCE(34:0)", "PCE(34:1)", "PCE(38:4)", "PCE(38:5)"
  ))
  observed_unmatched <- tolower(x$standard_name) %in% unmatched
  if ("feature" %in% names(x)) {
    observed_unmatched <- observed_unmatched | tolower(as.character(x$feature)) %in% unmatched
  }
  reason[observed_unmatched] <- "user_reported_LION_unmatched_2026-09-10"
  x$lion_web_eligible <- !nzchar(reason)
  x$lion_web_exclusion_reason <- reason
  x$lion_web_mapping_policy <- "local_standardization_plus_web_exclusions_2026-09-10"
  x
}

export_lion_custom_statistics <- function(stat_table, output_dir, ages, fdr = 0.05) {
  required <- c(
    "standard_name", "requested_peak_age", "coefficient", "p_value",
    "mapping_status"
  )
  missing_columns <- setdiff(required, names(stat_table))
  if (length(missing_columns)) {
    stop("Missing required columns: ", paste(missing_columns, collapse = ", "))
  }
  stat_table <- lion_main_peak_rows(stat_table)
  lion_require_main_ages(ages, unique(stat_table$requested_peak_age))
  if (length(fdr) != 1L || !is.finite(fdr) || fdr <= 0 || fdr >= 1) stop("Invalid fdr.")
  for (column in c("requested_peak_age", "coefficient", "p_value")) {
    stat_table[[column]] <- suppressWarnings(as.numeric(as.character(stat_table[[column]])))
  }
  stat_table <- lion_web_mapping(stat_table)
  if (!"q_value" %in% names(stat_table)) stat_table$q_value <- NA_real_
  stat_table$q_value <- suppressWarnings(as.numeric(as.character(stat_table$q_value)))
  stat_table$signed_z_score <- lion_signed_z(stat_table$coefficient, stat_table$p_value)

  # Validate every requested age before writing. Ranking keeps the complete
  # mapped background; it never filters to significant or positive lipids.
  prepared <- lapply(ages, function(age) {
    age_rows <- stat_table[
      is.finite(stat_table$requested_peak_age) &
        abs(stat_table$requested_peak_age - age) < 1e-8,
      , drop = FALSE
    ]
    center_age <- NA_real_
    if ("center_age" %in% names(age_rows)) {
      centers <- unique(suppressWarnings(as.numeric(as.character(age_rows$center_age))))
      if (length(centers) != 1L || any(!is.finite(centers)) || abs(centers[[1L]] - age) > 1e-8) {
        stop("Expected the exact DE-SWAN center_age for peak age ", age, ".")
      }
      center_age <- centers[[1L]]
    }
    age_rows$finite_ranking_statistic <- is.finite(age_rows$coefficient) &
      is.finite(age_rows$signed_z_score)
    keep <- age_rows$finite_ranking_statistic & age_rows$lion_web_eligible
    one <- age_rows[keep, , drop = FALSE]
    if (!nrow(one)) stop("No usable mapped lipids found for age ", age, ".")
    if (any(grepl("[\t\r\n]", one$standard_name))) {
      stop("Lipid names cannot contain tabs or newlines (age ", age, ").")
    }
    # Resolve duplicate submission names deterministically using largest |z|;
    # keep the selected feature and the excluded duplicates in the audit.
    one <- one[order(-abs(one$signed_z_score), one$standard_name,
                     if ("feature" %in% names(one)) one$feature else seq_len(nrow(one))), , drop = FALSE]
    one <- one[!duplicated(one$standard_name), , drop = FALSE]
    one <- one[order(-one$signed_z_score, one$standard_name), , drop = FALSE]
    age_rows$selected_for_ranking <- rownames(age_rows) %in% rownames(one)
    age_rows$ranking_exclusion_reason <- age_rows$lion_web_exclusion_reason
    age_rows$ranking_exclusion_reason[age_rows$lion_web_eligible & !age_rows$finite_ranking_statistic] <-
      "nonfinite_ranking_statistic"
    age_rows$ranking_exclusion_reason[keep & !age_rows$selected_for_ranking] <- "duplicate_standard_name"
    list(data = one, center_age = center_age, audit = age_rows)
  })

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  output_dir <- normalizePath(output_dir, winslash = "/", mustWork = TRUE)
  upload_dir <- file.path(output_dir, "lion_web_inputs")
  dir.create(upload_dir, recursive = TRUE, showWarnings = FALSE)
  write_ranking <- function(x, path) {
    write.table(
      x, path, sep = "\t", row.names = FALSE, col.names = FALSE,
      quote = FALSE, na = "", dec = ".", fileEncoding = "UTF-8"
    )
  }
  summaries <- vector("list", length(ages))
  runs <- vector("list", length(ages))
  for (i in seq_along(ages)) {
    age <- ages[[i]]
    label <- lion_age_label(age)
    one <- prepared[[i]]$data
    upload <- data.frame(standard_name = one$standard_name,
                         ranking_score = one$signed_z_score)
    input_file <- file.path(upload_dir,
      paste0("lion_age_", label, "_two_sided_high_to_low.txt"))
    write_ranking(upload, input_file)
    runs[[i]] <- data.frame(
      schema_version = 2L,
      analysis_mode = "ranking",
      analysis_role = "primary_two_sided_ranking",
      requested_peak_age = age, is_main = TRUE, peak_class = "main",
      center_age = prepared[[i]]$center_age,
      biological_direction = "two_sided",
      score_multiplier = 1,
      score_definition = "signed_z",
      web_ranking_direction = "high-to-low",
      web_alternative_hypothesis = "two-tailed",
      web_test = "Kolmogorov-Smirnov",
      multiple_testing_family = "LION_terms_within_main_peak",
      exported_lipids = nrow(upload),
      selection_rule = "all_mapped_finite_statistics_no_significance_or_sign_filter",
      lipid_fdr_for_audit_only = fdr,
      significant_input_lipids = sum(is.finite(one$q_value) & one$q_value < fdr),
      nonsignificant_input_lipids = sum(is.finite(one$q_value) & one$q_value >= fdr),
      unknown_q_input_lipids = sum(!is.finite(one$q_value)),
      excluded_mapping_features = sum(!prepared[[i]]$audit$lion_web_eligible),
      input_positive_statistics = sum(upload$ranking_score > 0),
      input_negative_statistics = sum(upload$ranking_score < 0),
      input_zero_statistics = sum(upload$ranking_score == 0),
      input_file = input_file,
      input_md5 = unname(tools::md5sum(input_file)),
      result_file = file.path(output_dir, paste0("lion_age_", label, "_two_sided_result.csv")),
      stringsAsFactors = FALSE
    )
    summaries[[i]] <- data.frame(
      requested_peak_age = age, is_main = TRUE, peak_class = "main",
      center_age = prepared[[i]]$center_age,
      exported_lipids = nrow(one),
      positive_statistics = sum(one$signed_z_score > 0),
      negative_statistics = sum(one$signed_z_score < 0),
      zero_statistics = sum(one$signed_z_score == 0),
      output_file = input_file,
      web_ranking_direction = "high-to-low",
      web_alternative_hypothesis = "two-tailed",
      stringsAsFactors = FALSE
    )
    message("Saved ", nrow(upload), " lipids (high-to-low; two-tailed KS): ", input_file)
  }
  summary_table <- do.call(rbind, summaries)
  run_manifest <- do.call(rbind, runs)
  write.csv(summary_table, file.path(output_dir, "lion_custom_statistic_summary.csv"),
            row.names = FALSE, na = "")
  write.csv(run_manifest, file.path(upload_dir, "lion_web_run_manifest.csv"),
            row.names = FALSE, na = "")
  write.csv(do.call(rbind, lapply(prepared, `[[`, "audit")),
            file.path(upload_dir, "lion_ranking_input_audit.csv"), row.names = FALSE, na = "")
  readme_file <- file.path(upload_dir, "README_LION_WEB.md")
  writeLines(enc2utf8(c(
    "# LION/web primary analysis: one two-tailed ranking per main peak", "",
    "Single entry point for input preparation: 03_prepare_lion_custom_statistics.R. Only main peaks classified upstream are processed; candidate peaks are excluded.", "",
    "1. Open ranking mode, skip (i) process input and go directly to (ii) analysis.",
    "2. Paste both columns of the TXT listed below into the custom-statistic input box.",
    "3. K-S settings: set ranking direction to from high to low.",
    "4. Set alternative hypothesis to two-tailed; the website may still default to one-tailed, so check it manually.",
    "5. Submit each main peak once, download the complete result CSV and save it in the parent directory under the name listed below.",
    "6. Plot the results with 03_visualize_lion_results.R; the local analysis script no longer imports website results.", "",
    "| Main peak | Input TXT | Lipids | Significant / non-significant | Website options | Result CSV |",
    "|---:|---|---:|---:|---|---|",
    vapply(seq_len(nrow(run_manifest)), function(k) {
      run <- run_manifest[k, ]
      paste0("| ", run$requested_peak_age, " | ", basename(run$input_file),
        " | ", run$exported_lipids, " | ", run$significant_input_lipids, " / ",
        run$nonsignificant_input_lipids, " | high-to-low + two-tailed | ",
        basename(run$result_file), " |")
    }, character(1L)), "",
    "## Statistic and interpretive limits", "",
    "z = sign(beta) * Phi^-1(1 - p/2), where p is the raw DE-SWAN P value; a positive beta means higher on the older side of the window than on the younger side.",
    "z is a direction-preserving evidence score, not an abundance effect size, and is not claimed to be the Wald z of the original model.",
    "Each TXT contains all matchable lipids with a finite statistic, including non-significant lipids and scores of both signs; nothing is truncated by q or sign.",
    "--fdr affects only the ranking audit counts, not the ranking list or scores.",
    "The two-tailed test compares the distribution of the members of an ontology term in the signed ranking with the expected distribution; each term receives one two-tailed P/q.",
    "Two-tailed significance does not imply that a whole term is increased, decreased or strongly changed; inspect member coefficients and the rank distribution.",
    "Do not rerun the two-tailed test on -z, and do not replace the signed input with |z|.",
    "LION term enrichment is not pathway activity or a direct functional measurement; different modes are not independent validation.",
    "Term q values are adjusted within the website run of each main peak; no unified FDR across peaks or modes is claimed.", "",
    "## Names, audit and legacy files", "",
    "The current rules exclude CoQ10, 3 acca and 4 pce names reported as unmatched by the website; the ether-bond structure of PCE is not guessed.",
    "lion_ranking_input_audit.csv keeps all original features with coefficients, P/q, signed z, inclusion status and reason for exclusion.",
    "For duplicated submission names, the original feature with the largest |z| is kept and the excluded duplicates are recorded.",
    "Legacy older_high/older_low ranking TXT and result.csv files are not part of this two-tailed primary analysis; do not mix them.",
    "Only the two_sided files listed in the manifest are used; legacy custom scores and raw-abundance upload matrices are no longer generated.",
    "The manifest records the recommended website settings and input MD5 and cannot replace confirmation of the settings actually submitted.", "",
    "Official implementation of the LION two-tailed option: https://github.com/martijnmolenaar/LION-web/blob/master/OntologyApp/ui.R"
  )), readme_file, useBytes = TRUE)
  invisible(list(summary = summary_table, runs = run_manifest, readme = readme_file))
}

write_lion_analysis_workflow <- function(output_dir, ranking_runs = NULL, target_runs = NULL) {
  path <- file.path(output_dir, "README_workflow.md")
  lines <- c(
    "# LION analysis of main peaks: one two-tailed ranking + two directional target lists", "",
    "Only main peaks with is_main=TRUE / peak_class=main are processed; candidate peaks are not included.", "",
    "## Script responsibilities", "",
    "| Script | Responsibility |",
    "|---|---|",
    "| 03_lipid_pathway_enrichment.R | Local lipid-set enrichment, complete main-peak statistics table, local two-panel figure |",
    "| 03_prepare_lion_custom_statistics.R | LION website inputs and submission manifest; no website enrichment or plotting |",
    "| 03_visualize_lion_results.R | Import and plot the two-tailed ranking results from the website |",
    "| 03_visualize_lion_target_list_results.R | Import and plot the increased/decreased target-list results from the website |", "",
    "The local Wilcoxon set test and the LION/web two-tailed KS test are different analyses and do not replace each other.", "",
    "## 1. Primary analysis: one two-tailed ranking of all matchable lipids", "",
    "Ranking mode → (ii) analysis; paste both columns. from high to low + two-tailed must be set."
  )
  if (!is.null(ranking_runs) && nrow(ranking_runs)) {
    lines <- c(lines, "",
      "| Main peak | All lipids | Significant | Non-significant | Input TXT | Result CSV (this directory) |",
      "|---:|---:|---:|---:|---|---|",
      vapply(seq_len(nrow(ranking_runs)), function(i) {
        x <- ranking_runs[i, ]
        paste0("| ", x$requested_peak_age, " | ", x$exported_lipids,
          " | ", x$significant_input_lipids, " | ", x$nonsignificant_input_lipids,
          " | [", basename(x$input_file), "](lion_web_inputs/", basename(x$input_file), ") | ",
          basename(x$result_file), " |")
      }, character(1L)))
  } else lines <- c(lines, "No ranking was generated with this --mode; use --mode=ranking or --mode=both.")
  lines <- c(lines, "",
    "Signed score z = sign(beta) * Phi^-1(1 - p/2), from the raw P value and the coefficient at the exact main-peak centre.",
    "Each main peak is tested once, two-tailed; the ranking is not split into increased and decreased parts and -z is not rerun.",
    "Inputs are not filtered by significance. The two-tailed q reflects only a difference in rank distribution; direction must be interpreted from member coefficients and distributions.", "",
    "## 2. Supplementary: increased and decreased target lists", "",
    "In target-list mode, submit the two foregrounds separately with the shared [matchable background](lion_web_target_list_inputs/lion_background_matched_lipids.txt).")
  if (!is.null(target_runs) && nrow(target_runs)) {
    target_runs <- target_runs[target_runs$selection %in% c("older_high", "older_low"), , drop = FALSE]
    lines <- c(lines, "",
      "| Main peak | Direction | Foreground | Background | Input TXT | Result CSV (in lion_web_target_list_inputs) |",
      "|---:|---|---:|---:|---|---|",
      vapply(seq_len(nrow(target_runs)), function(i) {
        x <- target_runs[i, ]
        paste0("| ", x$requested_peak_age, " | ", x$selection, " | ", x$target_lipids,
          " | ", x$background_lipids, " | [", basename(x$input_file), "](lion_web_target_list_inputs/",
          basename(x$input_file), ") | ", basename(x$result_file), " |")
      }, character(1L)))
  } else lines <- c(lines, "No target list was generated with this --mode; use --mode=target_list or --mode=both.")
  lines <- c(lines, "",
    "The foreground uses lipid-level BH q < --fdr and then beta > 0 or beta < 0; empty foregrounds are skipped.",
    "The background is all matchable measured lipids, including non-significant ones; significant lipids must not be used as the background.",
    "Lipid-level BH is computed over all tests at that centre and is not recomputed within the matchable subset.",
    "Only the two directional foregrounds are exported by default; --include_combined_target=true also exports a combined reference.",
    "q values of different directions or modes do not form a unified FDR; report them as supplementary results and do not choose the most significant mode as the primary analysis.", "",
    "## 3. Order of execution", "",
    "Run in an R environment from the directory containing these scripts:", "",
    "```powershell",
    "# Local analysis; omit tables_only to also plot the local results",
    "Rscript 03_lipid_pathway_enrichment.R --tables_only=true",
    "# Prepare LION inputs only",
    "Rscript 03_prepare_lion_custom_statistics.R --mode=both",
    "# Or skip the local analysis and prepare LION inputs directly from the RDS:",
    "Rscript 03_prepare_lion_custom_statistics.R --mode=both --rds=outputs/deswan-window_7.5/lipidomics_deswan_breakpoint_window_7p5.rds",
    "# After submitting on the website, downloading and naming files as in the manifest, plot:",
    "Rscript 03_visualize_lion_results.R",
    "Rscript 03_visualize_lion_target_list_results.R",
    "```", "",
    "Ranking figures are saved to figures/lipid_pathway by default; real LION results are not produced until the website downloads are complete.",
    "Input sources and MD5 are in lion_preparation_sources.csv; website settings and result file names are in the manifest of each mode."
  )
  writeLines(enc2utf8(lines), path, useBytes = TRUE)
  invisible(path)
}
