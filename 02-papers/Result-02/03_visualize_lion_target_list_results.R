#!/usr/bin/env Rscript
# Standalone LION/web TARGET-LIST visualization.
# Main: union of supplied FDR q < --fdr terms, shown at every age/direction.
# Fill = Significant / Expected; area = Significant; * = supplied q < --fdr.
# Expected is rounded by LION, so the displayed fold enrichment is approximate.
# All supplied P/q values and all returned terms are retained in Source Data.
# Figure contract and usage: README_lion_target_list_figure.md.
#
# Rscript 03_visualize_lion_target_list_results.R
# --input=... --output=... --source_output=... --ages=auto --fdr=0.05
# Default input: outputs/lipid_pathway/lion_web_target_list_inputs.
# Default --selections=older_high,older_low. Use --selections=all_significant
# explicitly only for the optional combined foreground reference.
# Auto ages include only classified main peaks in the target-list manifest.
# Explicit --ages (for example --ages=48) must also be a main-peak subset.
# Figures default to <input>/../figures/target_list.
# --width_mm=100 --dpi=600
# Height follows the number of terms; the figure has no bottom caption.
# Tests can source functions with options(lion.target.figure.functions_only=TRUE).

target_figure_dir <- function() {
  sources <- as.character(unlist(lapply(sys.frames(), function(x) x$ofile)))
  sources <- sources[basename(sources) == "03_visualize_lion_target_list_results.R"]
  if (length(sources)) return(dirname(normalizePath(tail(sources, 1L), winslash = "/")))
  arg <- grep("^--file=", commandArgs(), value = TRUE)
  if (length(arg)) return(dirname(normalizePath(sub("^--file=", "", arg[[1L]]), winslash = "/")))
  candidates <- c(getwd(), file.path(getwd(), "02-papers", "Result-02"))
  candidates <- candidates[file.exists(file.path(candidates, "03_visualize_lion_target_list_results.R"))]
  if (!length(candidates)) stop("Cannot locate the plotting script.")
  normalizePath(candidates[[1L]], winslash = "/")
}

source(file.path(target_figure_dir(), "lion_custom_statistics_helpers.R"), local = TRUE, encoding = "UTF-8")

target_figure_args <- function(argv, script_dir) {
  args <- list(
    input = lion_default_paths(script_dir)$target_input,
    output = "auto",
    source_output = "auto", ages = "auto", selections = "older_high,older_low",
    fdr = "0.05", width_mm = "100", dpi = "600"
  )
  for (arg in argv) {
    if (!grepl("^--[^=]+=", arg)) stop("Expected --key=value: ", arg)
    key <- sub("^--([^=]+)=.*$", "\\1", arg)
    if (!key %in% names(args)) stop("Unknown option: --", key)
    args[[key]] <- sub("^--[^=]+=", "", arg)
  }
  for (key in c("fdr", "width_mm", "dpi")) {
    args[[key]] <- suppressWarnings(as.numeric(args[[key]]))
    if (length(args[[key]]) != 1L || !is.finite(args[[key]])) stop("Invalid --", key)
  }
  if (tolower(trimws(args$ages)) == "auto") {
    args$ages <- NULL
  } else {
    args$ages <- suppressWarnings(as.numeric(trimws(strsplit(args$ages, ",", fixed = TRUE)[[1L]])))
    if (!length(args$ages) || any(!is.finite(args$ages)) || anyDuplicated(args$ages)) {
      stop("--ages must be 'auto' or contain unique numeric ages.")
    }
  }
  if (args$fdr <= 0 || args$fdr >= 1) stop("--fdr must be between 0 and 1.")
  args$selections <- trimws(strsplit(args$selections, ",", fixed = TRUE)[[1L]])
  if (!length(args$selections) || anyDuplicated(args$selections) ||
      any(!args$selections %in% c("older_high", "older_low", "all_significant"))) {
    stop("--selections must contain older_high, older_low and/or all_significant.")
  }
  if (args$width_mm < 100 || args$width_mm > 183) stop("--width_mm must be between 100 and 183.")
  if (args$dpi < 300) stop("--dpi must be at least 300.")
  if (identical(args$output, "auto")) args$output <- file.path(dirname(args$input), "figures", "target_list")
  if (identical(args$source_output, "auto")) args$source_output <- file.path(args$input, "visualization")
  args
}

target_load_runs <- function(input_dir, ages = NULL, selections = c("older_high", "older_low")) {
  path <- file.path(input_dir, "lion_target_list_run_manifest.csv")
  if (!file.exists(path)) {
    stop("Main-peak selection requires the target-list export manifest: ", path,
         ". Run 03_prepare_lion_custom_statistics.R first.")
  }
  manifest <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  required <- c("analysis_mode", "requested_peak_age", "selection", "target_lipids")
  if (!all(required %in% names(manifest))) stop("Incomplete target-list export manifest.")
  if (!length(selections) || anyDuplicated(selections) ||
      any(!selections %in% c("older_high", "older_low", "all_significant"))) stop("Invalid target-list selections.")
  manifest <- manifest[manifest$analysis_mode %in% "target-list", , drop = FALSE]
  manifest <- lion_main_peak_rows(manifest, lookup = lion_main_peak_metadata(dirname(input_dir)))
  main_ages <- sort(unique(manifest$requested_peak_age))
  if (is.null(ages)) ages <- main_ages else lion_require_main_ages(ages, main_ages)
  runs <- manifest[manifest$selection %in% selections & manifest$requested_peak_age %in% ages, , drop = FALSE]
  runs$target_lipids <- suppressWarnings(as.numeric(runs$target_lipids))
  key <- paste(runs$requested_peak_age, runs$selection)
  expected <- as.vector(t(outer(ages, selections, paste)))
  if (!nrow(runs) || anyDuplicated(key) || !setequal(key, expected) ||
      any(!is.finite(runs$target_lipids)) ||
      any(runs$target_lipids < 0 | runs$target_lipids != floor(runs$target_lipids))) {
    stop("Invalid ages or target counts in target-list manifest.")
  }
  runs <- runs[match(expected, key), , drop = FALSE]
  if (any(runs$target_lipids == 0)) {
    message("Skipping empty target-list runs: ",
            paste(expected[runs$target_lipids == 0], collapse = ", "))
  }
  runs <- runs[runs$target_lipids > 0, , drop = FALSE]
  if (!nrow(runs)) stop("No nonempty main-peak target lists to visualize.")
  runs$age <- runs$requested_peak_age
  runs$run_id <- paste0("age_", vapply(runs$age, target_age_label, character(1L)), "_", runs$selection)
  runs$column_index <- seq_len(nrow(runs))
  labels <- c(older_high = "Older-high", older_low = "Older-low", all_significant = "Combined")
  runs$column_label <- paste0(runs$age, " years\n", unname(labels[runs$selection]))
  runs
}

target_resolve_ages <- function(input_dir, ages = NULL, selections = c("older_high", "older_low")) {
  unique(target_load_runs(input_dir, ages, selections)$requested_peak_age)
}

target_age_label <- function(age) {
  gsub("[^A-Za-z0-9]+", "_", format(age, scientific = FALSE, trim = TRUE))
}

target_read_results <- function(input_dir, ages = NULL, selections = c("older_high", "older_low")) {
  runs <- target_load_runs(input_dir, ages, selections)
  rows <- lapply(seq_len(nrow(runs)), function(i) {
    file <- file.path(input_dir, paste0("lion_target_list_age_", target_age_label(runs$age[[i]]),
                                      "_", runs$selection[[i]], "_result.csv"))
    if (!file.exists(file)) stop("Missing target-list result: ", file)
    x <- read.csv(file, check.names = FALSE, stringsAsFactors = FALSE)
    description <- intersect(c("Discription", "Description"), names(x))
    fields <- c("Term ID", "Annotated", "Significant", "Expected", "p-value", "FDR q-value")
    if (anyDuplicated(names(x)) || !length(description) || !all(fields %in% names(x))) {
      stop("Expected LION target-list schema, including Significant and Expected: ", basename(file))
    }
    one <- data.frame(
      term_id = trimws(as.character(x[["Term ID"]])),
      term = trimws(as.character(x[[description[[1L]]]])),
      annotated = suppressWarnings(as.numeric(x[["Annotated"]])),
      hits = suppressWarnings(as.numeric(x[["Significant"]])),
      expected = suppressWarnings(as.numeric(x[["Expected"]])),
      p_value = suppressWarnings(as.numeric(x[["p-value"]])),
      q_value = suppressWarnings(as.numeric(x[["FDR q-value"]])),
      age = runs$age[[i]], selection = runs$selection[[i]], run_id = runs$run_id[[i]],
      column_index = i, column_label = runs$column_label[[i]], stringsAsFactors = FALSE
    )
    if (!nrow(one) || anyNA(one$term_id) || any(!nzchar(one$term_id)) ||
        anyDuplicated(one$term_id) || anyNA(one$term) || any(!nzchar(one$term))) {
      stop("Empty or duplicate ontology IDs/labels: ", basename(file))
    }
    if (any(!is.finite(as.matrix(one[c("annotated", "hits", "expected", "p_value", "q_value")])))) {
      stop("Missing or nonnumeric target-list result values: ", basename(file))
    }
    if (any(one$annotated < 0 | one$hits < 0 | one$hits > one$annotated) ||
        any(one$annotated != floor(one$annotated) | one$hits != floor(one$hits)) ||
        any(one$expected <= 0 | one$expected > one$annotated + 0.005) ||
        any(one$p_value < 0 | one$p_value > 1 | one$q_value < 0 | one$q_value > 1)) {
      stop("Invalid count, probability, or non-positive Expected denominator: ", basename(file))
    }
    # No recomputation of q and no division by the submitted background count.
    one$fold_enrichment <- one$hits / one$expected
    one$result_file <- normalizePath(file, winslash = "/")
    one$result_md5 <- unname(tools::md5sum(file))
    one
  })
  results <- do.call(rbind, rows)
  labels <- unique(results[c("term_id", "term")])
  if (anyDuplicated(labels$term_id)) stop("Conflicting labels for the same ontology ID across ages.")
  results
}

target_check_manifest <- function(input_dir, results, ages = NULL, selections = c("older_high", "older_low")) {
  manifest <- target_load_runs(input_dir, ages, selections)
  required <- c("analysis_mode", "requested_peak_age", "selection", "input_file", "input_md5",
                "background_file", "background_md5", "background_lipids", "target_lipids")
  if (!all(required %in% names(manifest))) stop("Incomplete target-list export manifest.")
  if (!setequal(unique(results$run_id), manifest$run_id)) stop("Results and requested target-list runs disagree.")
  for (field in c("input", "background")) {
    file_col <- paste0(field, "_file")
    hash_col <- paste0(field, "_md5")
    # Resolve beside these CSVs, so moving the output folder is supported.
    manifest[[file_col]] <- file.path(input_dir, basename(manifest[[file_col]]))
    if (any(!file.exists(manifest[[file_col]])) || anyNA(manifest[[hash_col]]) ||
        any(unname(tools::md5sum(manifest[[file_col]])) != manifest[[hash_col]])) {
      stop("Input TXT checksum differs from the export manifest: ", field)
    }
  }
  if (length(unique(manifest$background_md5)) != 1L) stop("All directions/ages must share the same submitted background.")
  for (i in seq_len(nrow(manifest))) {
    x <- results[results$run_id == manifest$run_id[[i]], ]
    if (any(x$hits > manifest$target_lipids[[i]]) ||
        any(x$annotated > manifest$background_lipids[[i]])) stop("Result counts exceed submitted list sizes.")
  }
  manifest
}

target_display_labels <- function(term_id, term) {
  # Display-only aliases; every original term and ID remains in Source Data.
  aliases <- c(
    "LION:0000002" = "Glycerolipids", "LION:0000622" = "Triacylglycerols",
    "LION:0012011" = "Lipid storage", "LION:0012084" = "Lipid droplet",
    "LION:0000094" = "Neutral headgroup charge", "LION:0000038" = "Diacyl PE",
    "LION:0012009" = "Lipid-mediated signalling", "LION:0012086" = "Endosome/lysosome"
  )
  labels <- gsub(" \\[.*\\]$", "", term)
  labels <- vapply(labels, function(x) {
    if (grepl("^[a-z]+( |$)", x)) paste0(toupper(substr(x, 1L, 1L)), substring(x, 2L)) else x
  }, character(1L))
  use_alias <- term_id %in% names(aliases)
  labels[use_alias] <- unname(aliases[term_id[use_alias]])
  labels
}

target_select_terms <- function(results, fdr) {
  terms <- do.call(rbind, lapply(split(results, results$term_id), function(x) data.frame(
    term_id = x$term_id[[1L]], term = x$term[[1L]], min_q = min(x$q_value),
    significant_runs = sum(x$q_value < fdr), returned_runs = nrow(x),
    stringsAsFactors = FALSE
  )))
  terms <- terms[order(terms$min_q, terms$term_id, method = "radix"), , drop = FALSE]
  terms$display_name <- target_display_labels(terms$term_id, terms$term)
  terms$selected <- terms$min_q < fdr
  terms$selection_rule <- ifelse(terms$selected, "q_below_threshold_in_at_least_one_run",
                                  "no_run_below_threshold")
  terms$data_order <- seq_len(nrow(terms))
  terms$wrapped_name <- vapply(terms$display_name,
    function(x) paste(strwrap(x, width = 28), collapse = "\n"), character(1L))
  lines <- lengths(strsplit(terms$wrapped_name, "\n", fixed = TRUE))
  terms$row_height_mm <- pmax(7.5, 3.0 * lines + 1.8)
  rownames(terms) <- NULL
  terms
}

target_complete_grid <- function(results, terms, ages, fdr) {
  runs <- unique(results[c("run_id", "age", "selection", "column_index", "column_label")])
  runs <- runs[runs$age %in% ages, , drop = FALSE]
  runs <- runs[order(runs$column_index), , drop = FALSE]
  grid <- expand.grid(term_id = terms$term_id, run_id = runs$run_id, stringsAsFactors = FALSE)
  grid <- merge(grid, results[setdiff(names(results), setdiff(names(runs), "run_id"))],
                by = c("term_id", "run_id"), all.x = TRUE, sort = FALSE)
  grid <- merge(grid, runs, by = "run_id", all.x = TRUE, sort = FALSE)
  grid <- merge(grid, terms[setdiff(names(terms), "term")], by = "term_id", all.x = TRUE, sort = FALSE)
  grid$term <- terms$term[match(grid$term_id, terms$term_id)]
  grid$returned <- is.finite(grid$q_value)
  grid$significant <- grid$returned & grid$q_value < fdr
  grid$status <- ifelse(!grid$returned, "not_returned",
                       ifelse(grid$hits == 0, "zero_hits", ifelse(grid$significant, "significant", "not_significant")))
  grid[order(grid$data_order, grid$column_index), , drop = FALSE]
}

target_paginate <- function(terms, row_budget_mm = 126) {
  page <- integer(nrow(terms))
  current <- 1L
  height <- 0
  for (i in seq_len(nrow(terms))) {
    if (height > 0 && height + terms$row_height_mm[[i]] > row_budget_mm) {
      current <- current + 1L
      height <- 0
    }
    page[[i]] <- current
    height <- height + terms$row_height_mm[[i]]
  }
  split(terms$term_id, page)
}

target_make_plot <- function(grid, terms, ages, term_ids, fdr, scales, main = TRUE) {
  lookup <- terms[match(term_ids, terms$term_id), , drop = FALSE]
  if (!nrow(lookup)) {
    plot <- ggplot2::ggplot() + ggplot2::theme_void(base_family = "Arial") +
      ggplot2::annotate("text", x = 0, y = 0, label = paste0("No terms have FDR q < ", fdr, "."),
                        family = "Arial", size = 8 / (72.27 / 25.4))
    return(list(plot = plot, height_mm = 85, rows = 0L))
  }
  row_height <- if (main) pmax(8, lookup$row_height_mm) else lookup$row_height_mm
  total_height <- sum(row_height)
  lookup$y <- total_height - cumsum(row_height) + row_height / 2
  data <- grid[grid$term_id %in% term_ids, , drop = FALSE]
  data$y <- lookup$y[match(data$term_id, lookup$term_id)]
  columns <- unique(grid[c("column_index", "column_label")])
  columns <- columns[order(columns$column_index), , drop = FALSE]
  plot <- ggplot2::ggplot(data, ggplot2::aes(column_index, y)) +
    ggplot2::geom_hline(yintercept = head(total_height - cumsum(row_height), -1L),
      colour = "#E8EDF1", linewidth = 0.1) +
    ggplot2::geom_vline(xintercept = columns$column_index, colour = "#E8EDF1", linewidth = 0.1) +
    ggplot2::geom_point(data = data[data$returned & data$hits > 0, ],
      ggplot2::aes(size = hits, fill = fold_enrichment,
                   colour = ifelse(significant, "Significant", "Not significant")),
      shape = 21, stroke = 0.32) +
    ggplot2::geom_text(data = data[data$significant, ],
      ggplot2::aes(label = "*"),
      size = 8 / (72.27 / 25.4), family = "Arial", colour = "#24313D",
      hjust = -4, vjust = 0.5) +
    ggplot2::geom_segment(data = data[data$returned & data$hits == 0, ],
      ggplot2::aes(x = column_index - 0.014, xend = column_index + 0.014, yend = y),
      linewidth = 0.25, colour = "#929DA8") +
    ggplot2::geom_point(data = data[!data$returned, ], shape = 4, size = 1.5, colour = "#929DA8") +
    ggplot2::scale_x_continuous(breaks = columns$column_index, labels = columns$column_label,
      limits = c(0.5, nrow(columns) + 0.5), expand = c(0, 0), position = "top") +
    ggplot2::scale_y_continuous(breaks = lookup$y, labels = lookup$wrapped_name,
      limits = c(0, total_height), expand = c(0, 0)) +
    ggplot2::scale_fill_gradientn(colours = c("#F3F5F7", "#D9E8EF", "#81ADC2", "#245C7A"),
      values = c(0, 0.2, 0.55, 1), limits = c(0, scales$fold_max),
      breaks = scales$fold_breaks, name = "Fold enrichment") +
    ggplot2::scale_colour_manual(values = c("Significant" = "#233540", "Not significant" = "#AEBBC5"), guide = "none") +
    ggplot2::scale_size_area(max_size = 6.5, limits = c(0, scales$hit_max), breaks = scales$hit_breaks,
      name = "Foreground lipids") +
    ggplot2::guides(
      fill = ggplot2::guide_colourbar(order = 1, display = "rectangles",
        barheight = grid::unit(24, "mm"), barwidth = grid::unit(3, "mm")),
      size = ggplot2::guide_legend(order = 2, override.aes = list(fill = "#A9C6D4", colour = "#5F7582"))
    ) +
    ggplot2::labs(x = NULL, y = NULL,
      title = if (main) "Lipid ontology enrichment across age windows" else "Lipid ontology enrichment: all tested terms",
      subtitle = if (main) paste0(length(term_ids), " terms with FDR q < ", fdr, " in at least one run") else
        "Complete results; shared colour and size scales",
      caption = NULL) +
    ggplot2::theme_classic(base_size = 8, base_family = "Arial") +
    ggplot2::theme(
      axis.line = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(size = 8, face = "bold", colour = "#26333D",
        margin = ggplot2::margin(b = 7)),
      axis.text.y = ggplot2::element_text(size = 7.5, colour = "#26333D", lineheight = 1.0,
        margin = ggplot2::margin(r = 7)),
      plot.title = ggplot2::element_text(size = 9, face = "bold", margin = ggplot2::margin(b = 5)),
      plot.subtitle = ggplot2::element_text(size = 7.5, colour = "#596873", margin = ggplot2::margin(b = 9)),
      plot.caption = ggplot2::element_blank(),
      plot.title.position = "plot",
      legend.title = ggplot2::element_text(size = 7.5), legend.text = ggplot2::element_text(size = 7),
      legend.background = ggplot2::element_blank(), legend.key = ggplot2::element_blank(),
      legend.key.height = grid::unit(7, "mm"), legend.key.width = grid::unit(7, "mm"),
      legend.box.spacing = grid::unit(3, "mm"), legend.spacing.y = grid::unit(4, "mm"),
      plot.margin = ggplot2::margin(8, 8, 8, 8)
    )
  list(plot = plot, height_mm = max(86, total_height + 26), rows = nrow(lookup))
}

target_save_bundle <- function(spec, stem, output_dir, width_mm, dpi) {
  if (spec$height_mm > 170) stop("Figure height exceeds 170 mm; reduce supplementary page row budget.")
  file <- file.path(output_dir, stem)
  grDevices::cairo_pdf(paste0(file, ".pdf"), width = width_mm / 25.4,
    height = spec$height_mm / 25.4, family = "Arial")
  print(spec$plot)
  grDevices::dev.off()
  svglite::svglite(paste0(file, ".svg"), width = width_mm / 25.4, height = spec$height_mm / 25.4)
  print(spec$plot)
  grDevices::dev.off()
  ragg::agg_png(paste0(file, ".png"), width = width_mm, height = spec$height_mm,
    units = "mm", res = dpi, background = "white")
  print(spec$plot)
  grDevices::dev.off()
  ragg::agg_tiff(paste0(file, ".tiff"), width = width_mm, height = spec$height_mm,
    units = "mm", res = dpi, compression = "lzw", background = "white")
  print(spec$plot)
  grDevices::dev.off()
  # Preview is produced directly by R, never by a different plotting backend.
  ragg::agg_png(paste0(file, "_preview.png"), width = width_mm, height = spec$height_mm,
    units = "mm", res = 300, background = "white")
  print(spec$plot)
  grDevices::dev.off()
  message("Saved: ", paste0(file, ".pdf"))
}

target_figure_main <- function(argv = commandArgs(trailingOnly = TRUE)) {
  script_dir <- target_figure_dir()
  args <- target_figure_args(argv, script_dir)
  args$ages <- target_resolve_ages(args$input, args$ages, args$selections)
  message("Target-list main-peak ages: ", paste(args$ages, collapse = ", "))
  packages <- c("ggplot2", "svglite", "ragg")
  absent <- packages[!vapply(packages, requireNamespace, logical(1L), quietly = TRUE)]
  if (length(absent)) stop("Missing R packages: ", paste(absent, collapse = ", "))
  if (!capabilities("cairo")) stop("R needs Cairo for editable, embedded-font PDF export.")
  results <- target_read_results(args$input, args$ages, args$selections)
  manifest <- target_check_manifest(args$input, results, args$ages, args$selections)
  terms <- target_select_terms(results, args$fdr)
  grid <- target_complete_grid(results, terms, args$ages, args$fdr)
  dir.create(args$output, recursive = TRUE, showWarnings = FALSE)
  dir.create(args$source_output, recursive = TRUE, showWarnings = FALSE)
  write.csv(grid, file.path(args$source_output, "lion_target_list_all_source_data.csv"), row.names = FALSE, na = "")
  write.csv(grid[grid$selected, ], file.path(args$source_output, "lion_target_list_main_source_data.csv"),
    row.names = FALSE, na = "")
  write.csv(terms, file.path(args$source_output, "lion_target_list_term_selection.csv"), row.names = FALSE)
  provenance <- unique(results[c("run_id", "age", "selection", "result_file", "result_md5")])
  provenance$returned_terms <- vapply(provenance$run_id, function(id) sum(results$run_id == id), integer(1L))
  provenance$significant_terms <- vapply(provenance$run_id, function(id) sum(results$run_id == id & results$q_value < args$fdr), integer(1L))
  if (!is.null(manifest)) {
    provenance$submitted_foreground <- manifest$target_lipids
    provenance$submitted_background <- manifest$background_lipids
    provenance$input_md5 <- manifest$input_md5
    provenance$background_md5 <- manifest$background_md5
  }
  write.csv(provenance, file.path(args$source_output, "lion_target_list_figure_runs.csv"), row.names = FALSE)
  fold_max <- max(2, ceiling(max(results$fold_enrichment)))
  hit_max <- max(5, ceiling(max(results$hits) / 10) * 10)
  scales <- list(fold_max = fold_max, fold_breaks = pretty(c(0, fold_max), n = 4),
                 hit_max = hit_max, hit_breaks = c(5, 25, 50)[c(5, 25, 50) <= hit_max])
  main <- target_make_plot(grid, terms, args$ages, terms$term_id[terms$selected], args$fdr, scales)
  target_save_bundle(main, "lion_target_list_enrichment_main", args$output, args$width_mm, args$dpi)
  pages <- target_paginate(terms)
  for (i in seq_along(pages)) {
    spec <- target_make_plot(grid, terms, args$ages, pages[[i]], args$fdr, scales, main = FALSE)
    target_save_bundle(spec, sprintf("lion_target_list_all_terms_page_%02d", i),
      args$output, args$width_mm, args$dpi)
  }
  caption <- c(
    "Lipid ontology enrichment of age-associated lipids.", "",
    paste0("LION/web target-list enrichment at the main peak(s): ", paste(args$ages, collapse = "- and "),
      "-year analysis windows. Rows include all terms with a supplied FDR q < ", args$fdr,
      " in at least one age/direction run and show the corresponding results for every run."),
    paste0("Foreground selections: ", paste(args$selections, collapse = ", "),
      ". Directional lists use the lipid-level BH threshold from the export manifest and the sign of the DE-SWAN coefficient, with one common mapped background."),
    "Circle area represents the number of foreground lipid identifiers annotated to each term.",
    "Colour indicates fold enrichment, calculated as the reported observed count divided by the reported Expected count; ratios are approximate because Expected is rounded in the downloaded results.",
    paste0("Asterisks and dark outlines indicate q < ", args$fdr, "; unmarked bubbles did not meet this threshold."),
    "In the complete-results figures, a dash indicates zero hits and a cross indicates a term not returned by LION/web.",
    "Enrichment was assessed by one-sided Fisher exact tests in LION/web, with Benjamini-Hochberg correction within each web run. Original P values and q values were retained without further adjustment.",
    "Diacyl PE denotes diacylglycerophosphoethanolamines. Ontology terms can overlap and are not independent biological mechanisms.",
    "The circles summarize lipid membership counts, not participant means or independent biological replicates; no uncertainty intervals were supplied or constructed.",
    "Per-run q values do not control a pooled family spanning directions, ages or analysis modes. Differences in significance are not a direct comparison of effects. Source data are provided as a Source Data file."
  )
  writeLines(caption, file.path(args$source_output, "figure_legend_en.txt"))
  notes <- c(
    "# LION target-list figure notes", "",
    paste0("Input rows: ", nrow(results), "; distinct terms: ", nrow(terms), "; age windows: ", length(args$ages), "."),
    paste0("Main rows: ", sum(terms$selected), "; main cells: ", sum(grid$selected),
      "; complete grid cells: ", nrow(grid), "; supplementary pages: ", length(pages), "."),
    paste0("Main selection: supplied q < ", args$fdr, " in any requested age/direction run; no manual term exclusion or inferred ontology pruning."),
    paste0("Selections: ", paste(args$selections, collapse = ", "), "; nonempty runs: ", nrow(provenance), "."),
    "All supplied P/q values remain unchanged. Missing result rows are not converted to q=1.",
    "Fold enrichment = Significant / Expected. Expected > 0 is required; zero hits remain zero.",
    "Aliases affect labels only. Original ontology IDs and descriptions are retained in Source Data.",
    "Bottom caption is hidden; symbol definitions are retained in figure_legend_en.txt.",
    "Submitted list counts are recorded separately from the unknown effective LION matching counts.",
    "Only upstream main peaks are included; candidate peaks are excluded even when legacy files remain on disk.",
    "No claim of between-age significance, pathway activity or independent validation is made.",
    "No replicate-level raw data or intervals were supplied for these enrichment summaries; no error bars are invented.",
    paste0("Backend: R; width ", args$width_mm, " mm; raster resolution ", args$dpi,
      " dpi; text floor 5 pt; Arial; vector PDF and SVG with editable text."),
    "",
    "Rerun: Rscript 02-papers/Result-02/03_visualize_lion_target_list_results.R"
  )
  writeLines(notes, file.path(args$source_output, "lion_target_list_figure_notes.md"))
  writeLines(capture.output(sessionInfo()), file.path(args$source_output, "session_info.txt"))
  message("Done. Significant term counts: ", paste(paste0(provenance$run_id, " = ", provenance$significant_terms), collapse = "; "))
  invisible(list(results = results, terms = terms, grid = grid, provenance = provenance))
}

if (!isTRUE(getOption("lion.target.figure.functions_only", FALSE))) target_figure_main()
