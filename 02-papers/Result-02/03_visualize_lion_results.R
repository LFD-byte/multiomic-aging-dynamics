#!/usr/bin/env Rscript
# Data-driven LION/web visualization. No curated term IDs or biological groups.
# Reads RANKING results only. Target-list inputs/results have their own manifest
# in lion_web_target_list_inputs and must not replace these ranking CSV files.
#
# Question: which ontology terms deviate from the expected signed-rank distribution?
# Main: every term with q < --fdr in ANY run, shown in ALL runs.
# Rows: UP, DOWN, then direction-varying terms; minimum q within each group.
# Columns: Upregulated / Downregulated for each ascending main-peak age.
# These are display columns of ONE two-sided KS run, not two separate tests.
# Direction: the website's Regulated field, checked against ES when present.
# Fill: signed -log10(q), red UP / blue DOWN. Open circles: not significant.
# Opposite-direction cells stay blank; missing terms have crosses.
# Area: annotated fraction of input lipids, NOT an abundance effect size.
# Supplement: all returned terms, including never-significant terms, paginated.
# No ontology de-redundancy is claimed without term-to-lipid association data.
#
# Rscript 03_visualize_lion_results.R
# Optional: --input=... --output=... --fdr=0.05 --top_n=0 --rows_per_page=20
# Default width: 100 mm; height follows the term labels; no bottom caption.
# Default input: outputs/lipid_pathway.
# Each two-sided run ranks the complete mapped set, including nonsignificant
# lipids and both signs. LION name exclusions are recorded in the input audit.
# Figures default to figures/lipid_pathway beside this script;
# only classified main-peak runs are read. --output overrides this directory.
# Legacy directional manifests must be regenerated for the two-sided workflow.
# --top_n=0 retains the full significant union. Positive values limit the main
# view only. Full results and selection decisions are always exported.
# Tests may source functions with options(lion.visualizer.functions_only=TRUE).
# Resolve paths from this script without changing the caller's working directory.
lion_visualizer_dir <- function() {
  paths <- as.character(unlist(lapply(sys.frames(), function(frame) frame$ofile)))
  paths <- paths[basename(paths) == "03_visualize_lion_results.R"]
  if (length(paths)) return(dirname(normalizePath(tail(paths, 1L), winslash = "/")))
  arg <- grep("^--file=", commandArgs(), value = TRUE)
  if (length(arg)) return(dirname(normalizePath(sub("^--file=", "", arg[[1L]]), winslash = "/")))
  candidates <- c(getwd(), file.path(getwd(), "02-papers", "Result-02"))
  candidates <- candidates[file.exists(file.path(candidates, "03_visualize_lion_results.R"))]
  if (!length(candidates)) stop("Cannot locate 03_visualize_lion_results.R.")
  normalizePath(candidates[[1L]], winslash = "/")
}

source(file.path(lion_visualizer_dir(), "lion_custom_statistics_helpers.R"), local = TRUE, encoding = "UTF-8")

lion_visualizer_args <- function(argv, script_dir) {
  args <- list(
    input = lion_default_paths(script_dir)$output,
    output = "auto",
    fdr = "0.05", top_n = "0", rows_per_page = "20", width_mm = "100",
    dpi = "600"
  )
  for (arg in argv) {
    if (!grepl("^--[^=]+=", arg)) stop("Expected --key=value, received: ", arg)
    key <- sub("^--([^=]+)=.*$", "\\1", arg)
    if (!key %in% names(args)) stop("Unknown option: --", key)
    args[[key]] <- sub("^--[^=]+=", "", arg)
  }
  if (identical(args$output, "auto")) args$output <- file.path(script_dir, "figures", "lipid_pathway")
  for (key in c("fdr", "top_n", "rows_per_page", "width_mm", "dpi")) {
    args[[key]] <- suppressWarnings(as.numeric(args[[key]]))
    if (!is.finite(args[[key]])) stop("Invalid numeric option: --", key)
  }
  if (args$fdr <= 0 || args$fdr >= 1) stop("--fdr must be between 0 and 1.")
  if (args$top_n < 0 || args$top_n != floor(args$top_n)) stop("--top_n must be a non-negative integer.")
  if (args$rows_per_page < 5 || args$rows_per_page > 50 ||
      args$rows_per_page != floor(args$rows_per_page)) stop("--rows_per_page must be an integer from 5 to 50.")
  if (args$width_mm < 100 || args$width_mm > 220) stop("--width_mm must be between 100 and 220.")
  if (args$dpi < 300) stop("--dpi must be at least 300.")
  args
}

lion_load_runs <- function(input_dir, verify_hashes = TRUE) {
  manifest_path <- file.path(input_dir, "lion_web_inputs", "lion_web_run_manifest.csv")
  if (!file.exists(manifest_path)) stop("Run 03_prepare_lion_custom_statistics.R first; missing: ", manifest_path)
  runs <- read.csv(manifest_path, check.names = FALSE, stringsAsFactors = FALSE)
  required <- c("requested_peak_age", "biological_direction", "score_multiplier", "score_definition",
                "web_ranking_direction", "web_alternative_hypothesis", "exported_lipids",
                "input_file", "input_md5", "result_file")
  if (!all(required %in% names(runs))) {
    stop("Incomplete two-sided LION manifest. Regenerate inputs with 03_prepare_lion_custom_statistics.R.")
  }
  runs <- lion_main_peak_rows(runs, lookup = lion_main_peak_metadata(input_dir))
  for (column in c("requested_peak_age", "score_multiplier", "exported_lipids")) {
    runs[[column]] <- suppressWarnings(as.numeric(runs[[column]]))
    if (any(!is.finite(runs[[column]]))) stop("Invalid manifest column: ", column)
  }
  if (!nrow(runs) || any(runs$exported_lipids <= 0) ||
      any(runs$exported_lipids != floor(runs$exported_lipids))) stop("Invalid input lipid counts.")
  if (anyNA(runs$biological_direction) || any(runs$biological_direction != "two_sided") ||
      anyNA(runs$web_alternative_hypothesis) || any(runs$web_alternative_hypothesis != "two-tailed")) {
    stop("Expected one two-tailed ranking per main peak; directional one-tailed runs cannot be mixed in.")
  }
  if (anyNA(runs$web_ranking_direction) || any(runs$web_ranking_direction != "high-to-low") ||
      any(runs$score_multiplier != 1) || anyNA(runs$score_definition) ||
      any(runs$score_definition != "signed_z")) {
    stop("Two-sided ranking requires unmodified signed z, ordered high-to-low.")
  }
  if (anyDuplicated(runs$requested_peak_age)) stop("Expected exactly one two-sided run per main peak.")
  ages <- sort(unique(runs$requested_peak_age))
  runs <- runs[match(ages, runs$requested_peak_age), , drop = FALSE]
  age_label <- vapply(runs$requested_peak_age, function(age) {
    gsub("[^A-Za-z0-9]+", "_", format(age, scientific = FALSE, trim = TRUE))
  }, character(1L))
  expected_names <- paste0("lion_age_", age_label, "_two_sided_result.csv")
  if (anyNA(runs$result_file) || any(basename(runs$result_file) != expected_names)) {
    stop("Result filenames must identify the main-peak age and two_sided test.")
  }
  if (anyNA(runs$input_file) || any(basename(runs$input_file) !=
      paste0("lion_age_", age_label, "_two_sided_high_to_low.txt"))) {
    stop("Input filenames must identify the two-sided signed-z ranking.")
  }
  # Resolve within the selected folder rather than importing from an old checkout.
  runs$result_file <- file.path(normalizePath(input_dir, winslash = "/"), expected_names)
  runs$input_file <- file.path(input_dir, "lion_web_inputs", basename(runs$input_file))
  absent <- c(runs$result_file[!file.exists(runs$result_file)], runs$input_file[!file.exists(runs$input_file)])
  if (length(absent)) stop("Missing LION files: ", paste(basename(absent), collapse = ", "))
  if (verify_hashes && (anyNA(runs$input_md5) ||
      any(unname(tools::md5sum(runs$input_file)) != runs$input_md5))) {
    stop("Input TXT checksum differs from its export manifest. Reconcile inputs before plotting.")
  }
  # Check dimensions and sorting independently of the checksum. No positive-
  # score subset or reverse orientation is expected for this two-sided input.
  inputs <- lapply(seq_len(nrow(runs)), function(i) {
    x <- read.delim(runs$input_file[[i]], header = FALSE, quote = "", comment.char = "",
                    colClasses = "character", check.names = FALSE)
    if (ncol(x) != 2L || nrow(x) != runs$exported_lipids[[i]]) stop("Ranking TXT dimensions disagree with manifest.")
    names(x) <- c("lipid", "score")
    x$score <- suppressWarnings(as.numeric(x$score))
    if (anyNA(x$lipid) || any(!nzchar(x$lipid)) || anyDuplicated(x$lipid) ||
        any(!is.finite(x$score)) || any(diff(x$score) > 0)) stop("Invalid high-to-low ranking TXT.")
    x
  })
  audit_path <- file.path(input_dir, "lion_web_inputs", "lion_ranking_input_audit.csv")
  if (!file.exists(audit_path)) stop("Missing ranking input audit; regenerate LION inputs.")
  audit <- read.csv(audit_path, check.names = FALSE, stringsAsFactors = FALSE)
  if (!all(c("requested_peak_age", "selected_for_ranking", "standard_name", "signed_z_score") %in% names(audit))) {
    stop("Incomplete ranking input audit.")
  }
  for (i in seq_along(inputs)) {
    expected <- audit[audit$requested_peak_age == runs$requested_peak_age[[i]] &
      audit$selected_for_ranking %in% TRUE, , drop = FALSE]
    x <- inputs[[i]]
    if (anyDuplicated(expected$standard_name) || !setequal(x$lipid, expected$standard_name) ||
        !isTRUE(all.equal(x$score, expected$signed_z_score[match(x$lipid, expected$standard_name)],
                         tolerance = 1e-12))) {
      stop("Two-sided input must retain the complete original signed-z ranking from its audit.")
    }
  }
  runs$run_id <- paste0("age_", age_label, "_", runs$biological_direction)
  runs$column_index <- seq_len(nrow(runs))
  runs$column_label <- paste0(format(runs$requested_peak_age, trim = TRUE), " y\nTwo-sided KS")
  runs$result_md5 <- unname(tools::md5sum(runs$result_file))
  runs
}

lion_read_results <- function(runs) {
  rows <- lapply(seq_len(nrow(runs)), function(i) {
    x <- read.csv(runs$result_file[[i]], check.names = FALSE, stringsAsFactors = FALSE)
    if (any(c("Significant", "Expected") %in% names(x))) {
      stop("Target-list results cannot be used as ranking results: ", basename(runs$result_file[[i]]))
    }
    description <- intersect(c("Discription", "Description"), names(x))
    if (!length(description) || !all(c("Term ID", "Annotated", "p-value", "FDR q-value") %in% names(x))) {
      stop("Unexpected LION result schema: ", basename(runs$result_file[[i]]))
    }
    if (!"Regulated" %in% names(x)) {
      stop("Two-column plots require the LION Regulated (UP/DOWN) field. ",
           "Download the complete two-sided result table: ", basename(runs$result_file[[i]]),
           ". Direction cannot be inferred from a p or q value.")
    }
    regulated <- toupper(trimws(as.character(x$Regulated)))
    if (anyNA(regulated) || any(!regulated %in% c("UP", "DOWN"))) {
      stop("Regulated must be UP or DOWN for every returned term: ", basename(runs$result_file[[i]]))
    }
    enrichment_score <- if ("ES" %in% names(x)) suppressWarnings(as.numeric(x$ES)) else rep(NA_real_, nrow(x))
    if ("ES" %in% names(x) && (any(!is.finite(enrichment_score)) ||
        any(enrichment_score > 0 & regulated == "DOWN") || any(enrichment_score < 0 & regulated == "UP"))) {
      stop("LION ES and Regulated disagree or contain invalid scores: ", basename(runs$result_file[[i]]))
    }
    one <- data.frame(
      term_id = trimws(x[["Term ID"]]), term = trimws(x[[description[[1L]]]]),
      annotated = suppressWarnings(as.numeric(x[["Annotated"]])),
      p_value = suppressWarnings(as.numeric(x[["p-value"]])),
      q_value = suppressWarnings(as.numeric(x[["FDR q-value"]])),
      enrichment_score = enrichment_score,
      regulated = regulated,
      direction_source = "LION/web Regulated",
      run_id = runs$run_id[[i]], stringsAsFactors = FALSE
    )
    if (!nrow(one) || anyNA(one$term_id) || any(!nzchar(one$term_id)) ||
        anyDuplicated(one$term_id) || anyNA(one$term) || any(!nzchar(one$term))) {
      stop("Empty/duplicate term IDs or labels in ", basename(runs$result_file[[i]]))
    }
    if (any(!is.finite(as.matrix(one[, c("annotated", "p_value", "q_value")])))) {
      stop("Non-finite LION result values.")
    }
    if (any(one$p_value < 0 | one$p_value > 1 | one$q_value < 0 | one$q_value > 1) ||
        any(one$annotated < 0 | one$annotated > runs$exported_lipids[[i]]) ||
        any(one$annotated != floor(one$annotated))) stop("Invalid LION probabilities or annotation counts.")
    one
  })
  results <- do.call(rbind, rows)
  labels <- unique(results[, c("term_id", "term")])
  if (anyDuplicated(labels$term_id)) stop("Conflicting names for the same LION term across runs.")
  results
}

lion_select_terms <- function(results, fdr = 0.05, top_n = 0L) {
  summaries <- lapply(split(results, results$term_id), function(x) data.frame(
    term_id = x$term_id[[1L]], term = x$term[[1L]], min_q = min(x$q_value),
    significant_runs = sum(x$q_value < fdr), tested_runs = nrow(x),
    direction_group = if (length(unique(x$regulated)) == 1L) x$regulated[[1L]] else "VARIES",
    stringsAsFactors = FALSE
  ))
  terms <- do.call(rbind, summaries)
  terms <- terms[order(terms$min_q, -terms$significant_runs, terms$term_id, method = "radix"), ]
  rownames(terms) <- NULL
  terms$eligible <- terms$significant_runs > 0L
  eligible_index <- which(terms$eligible)
  selected_index <- if (top_n > 0L) head(eligible_index, top_n) else eligible_index
  terms$selected <- seq_len(nrow(terms)) %in% selected_index
  terms$selection_reason <- ifelse(terms$selected, "q_below_threshold_in_at_least_one_run",
                                  ifelse(terms$eligible, "excluded_by_explicit_top_n", "no_run_below_threshold"))
  # Select by evidence first; grouping must not make top_n favour one direction.
  terms <- terms[order(match(terms$direction_group, c("UP", "DOWN", "VARIES")),
                       terms$min_q, -terms$significant_runs, terms$term_id, method = "radix"), ]
  terms$data_order <- seq_len(nrow(terms))
  rownames(terms) <- NULL
  terms
}

lion_display_columns <- function(runs) {
  columns <- runs[rep(seq_len(nrow(runs)), each = 2L), , drop = FALSE]
  columns$display_direction <- rep(c("UP", "DOWN"), nrow(runs))
  columns$display_id <- paste(columns$run_id, columns$display_direction, sep = "_")
  columns$column_index <- seq_len(nrow(columns))
  columns$column_label <- paste0(format(columns$requested_peak_age, trim = TRUE), " y\n",
    ifelse(columns$display_direction == "UP", "Upregulated", "Downregulated"))
  rownames(columns) <- NULL
  columns
}

lion_complete_grid <- function(results, runs, terms, fdr) {
  columns <- lion_display_columns(runs)
  grid <- expand.grid(term_id = terms$term_id, display_id = columns$display_id, stringsAsFactors = FALSE)
  grid <- merge(grid, columns, by = "display_id", all.x = TRUE, sort = FALSE)
  grid <- merge(grid, results[, setdiff(names(results), "term"), drop = FALSE],
                by = c("term_id", "run_id"), all.x = TRUE, sort = FALSE)
  grid <- merge(grid, terms, by = "term_id", all.x = TRUE, sort = FALSE)
  grid$term_returned_in_run <- is.finite(grid$q_value)
  grid$returned <- grid$term_returned_in_run & grid$regulated == grid$display_direction
  grid$returned[is.na(grid$returned)] <- FALSE
  # The opposite column is not a second nonsignificant test. Leave its values
  # missing in plot source data as well as its mark blank. The unexpanded result
  # table separately preserves each original P/q/ES exactly once per term/run.
  opposite <- grid$term_returned_in_run & !grid$returned
  for (column in c("p_value", "q_value", "enrichment_score", "annotated")) grid[[column]][opposite] <- NA_real_
  grid$significant <- grid$returned & grid$q_value < fdr
  grid$log10_fdr <- -log10(pmax(grid$q_value, .Machine$double.xmin))
  grid$signed_log10_fdr <- ifelse(grid$regulated == "UP", 1, -1) * grid$log10_fdr
  grid$input_lipid_pct <- 100 * grid$annotated / grid$exported_lipids
  grid$status <- ifelse(!grid$term_returned_in_run, "not_returned",
    ifelse(opposite, "opposite_direction", ifelse(grid$significant, "significant", "not_significant")))
  grid[order(grid$data_order, grid$column_index), , drop = FALSE]
}

lion_make_plot <- function(grid, runs, term_ids, fdr, fill_limits, title, subtitle) {
  columns <- lion_display_columns(runs)
  if (!length(term_ids)) {
    plot <- ggplot2::ggplot() +
      ggplot2::annotate("text", x = 0, y = 0, label = paste0("No LION terms meet FDR < ", fdr, " in any run."),
                        size = 3, family = "Helvetica") + ggplot2::theme_void() + ggplot2::labs(title = title)
    return(list(plot = plot, height_mm = 85, rows = 0L, columns = columns))
  }
  lookup <- unique(grid[match(term_ids, grid$term_id), c("term_id", "term")])
  lookup$label <- vapply(lookup$term, function(x) paste(strwrap(x, width = 35), collapse = "\n"), character(1L))
  line_count <- lengths(strsplit(lookup$label, "\n", fixed = TRUE))
  row_height <- pmax(4.3, 2.6 * line_count + 0.8)
  total_height <- sum(row_height)
  lookup$y <- total_height - cumsum(row_height) + row_height / 2
  data <- grid[grid$term_id %in% term_ids, ]
  data$y <- lookup$y[match(data$term_id, lookup$term_id)]
  plot <- ggplot2::ggplot(data, ggplot2::aes(column_index, y)) +
    ggplot2::geom_hline(yintercept = lookup$y, colour = "#E9EDF2", linewidth = 0.2) +
    ggplot2::geom_vline(xintercept = columns$column_index, colour = "#E9EDF2", linewidth = 0.2) +
    ggplot2::geom_point(data = data[data$returned & !data$significant, ],
                        ggplot2::aes(size = input_lipid_pct, colour = regulated), shape = 21,
                        fill = "white", stroke = 0.3) +
    ggplot2::geom_point(data = data[data$significant, ],
                        ggplot2::aes(size = input_lipid_pct, fill = signed_log10_fdr), shape = 21,
                        colour = "#263E59", stroke = 0.3) +
    ggplot2::geom_point(data = data[data$status == "not_returned", ], shape = 4, size = 1.4, colour = "#737D89") +
    ggplot2::scale_x_continuous(breaks = columns$column_index, labels = columns$column_label,
                               limits = c(0.5, nrow(columns) + 0.5), expand = c(0, 0), position = "top") +
    ggplot2::scale_y_continuous(breaks = lookup$y, labels = lookup$label,
                               limits = c(0, total_height), expand = c(0, 0)) +
    ggplot2::scale_fill_gradient2(low = "#5B7EC9", mid = "white", high = "#D35F62",
      midpoint = 0, limits = c(-max(abs(fill_limits)), max(abs(fill_limits))),
      name = "Signed evidence\nDirection x -log10(q)") +
    ggplot2::scale_colour_manual(values = c(UP = "#D35F62", DOWN = "#5B7EC9"), guide = "none") +
    ggplot2::scale_size_area(max_size = 6.5, limits = c(0, 100), breaks = c(5, 20, 50),
                             name = "Input coverage (%)") +
    ggplot2::guides(
      fill = ggplot2::guide_colourbar(order = 1, display = "rectangles",
        direction = "horizontal", title.position = "top",
        barheight = grid::unit(3, "mm"), barwidth = grid::unit(26, "mm")),
      size = ggplot2::guide_legend(order = 2, nrow = 1, title.position = "top",
        override.aes = list(fill = "#D5DFE9", colour = "#53687F"))
    ) +
    ggplot2::labs(
      x = NULL, y = NULL, title = title, subtitle = subtitle,
      caption = NULL
    ) +
    ggplot2::theme_classic(base_size = 8, base_family = "Helvetica") +
    ggplot2::theme(
      axis.line = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
      axis.text.y = ggplot2::element_text(size = 7.5, colour = "#202B36", lineheight = 1.0,
                                          margin = ggplot2::margin(r = 5)),
      axis.text.x = ggplot2::element_text(size = 7.5, face = "bold", colour = "#202B36",
                                          margin = ggplot2::margin(b = 4), lineheight = 1.05),
      plot.title = ggplot2::element_text(size = 9, face = "bold", margin = ggplot2::margin(b = 5)),
      plot.subtitle = ggplot2::element_text(size = 7.5, colour = "#576575", margin = ggplot2::margin(b = 9)),
      plot.caption = ggplot2::element_blank(),
      plot.title.position = "plot",
      legend.position = "top", legend.location = "plot",
      legend.box = "horizontal", legend.justification = "left",
      legend.title = ggplot2::element_text(size = 7.5), legend.text = ggplot2::element_text(size = 7),
      legend.key.height = grid::unit(7, "mm"), legend.key.width = grid::unit(7, "mm"),
      legend.box.spacing = grid::unit(1.5, "mm"), legend.spacing.x = grid::unit(4, "mm"),
      legend.spacing.y = grid::unit(4, "mm"),
      plot.margin = ggplot2::margin(6, 8, 6, 8)
    )
  heading_height_mm <- 6 * as.integer(!is.null(title)) + 5 * as.integer(!is.null(subtitle))
  list(plot = plot, height_mm = max(80, total_height + 28 + heading_height_mm),
       rows = nrow(lookup), columns = columns)
}

lion_align_widths <- function(specifications) {
  grobs <- lapply(specifications, function(x) ggplot2::ggplotGrob(x$plot))
  if (length(unique(vapply(grobs, function(x) length(x$widths), integer(1L)))) != 1L) {
    stop("Cannot align the generated plot layouts.")
  }
  common_widths <- do.call(grid::unit.pmax, lapply(grobs, function(x) x$widths))
  for (i in seq_along(grobs)) {
    grobs[[i]]$widths <- common_widths
    specifications[[i]]$plot <- grobs[[i]]
  }
  specifications
}

lion_draw_plot <- function(plot) {
  if (inherits(plot, "grob")) {
    grid::grid.newpage()
    grid::grid.draw(plot)
  } else {
    print(plot)
  }
}

lion_save_bundle <- function(plot, stem, output_dir, width_mm, height_mm, dpi) {
  grDevices::cairo_pdf(file.path(output_dir, paste0(stem, ".pdf")), width = width_mm / 25.4,
                       height = height_mm / 25.4, family = "Helvetica")
  lion_draw_plot(plot)
  grDevices::dev.off()
  svglite::svglite(file.path(output_dir, paste0(stem, ".svg")), width = width_mm / 25.4, height = height_mm / 25.4)
  lion_draw_plot(plot)
  grDevices::dev.off()
  ragg::agg_png(file.path(output_dir, paste0(stem, ".png")), width = width_mm, height = height_mm,
                units = "mm", res = dpi, background = "white")
  lion_draw_plot(plot)
  grDevices::dev.off()
  ragg::agg_tiff(file.path(output_dir, paste0(stem, ".tiff")), width = width_mm, height = height_mm,
                 units = "mm", res = dpi, compression = "lzw", background = "white")
  lion_draw_plot(plot)
  grDevices::dev.off()
}

lion_visualizer_main <- function(argv = commandArgs(trailingOnly = TRUE)) {
  script_dir <- lion_visualizer_dir()
  args <- lion_visualizer_args(argv, script_dir)
  packages <- c("ggplot2", "svglite", "ragg")
  missing_packages <- packages[!vapply(packages, requireNamespace, logical(1L), quietly = TRUE)]
  if (length(missing_packages)) stop("Missing R packages: ", paste(missing_packages, collapse = ", "))
  runs <- lion_load_runs(args$input)
  results <- lion_read_results(runs)
  terms <- lion_select_terms(results, args$fdr, args$top_n)
  grid <- lion_complete_grid(results, runs, terms, args$fdr)
  columns <- lion_display_columns(runs)
  runs$returned_terms <- vapply(runs$run_id, function(id) sum(results$run_id == id), integer(1L))
  runs$significant_terms <- vapply(runs$run_id, function(id) sum(results$run_id == id & results$q_value < args$fdr), integer(1L))
  for (direction in c("UP", "DOWN")) {
    runs[[paste0(tolower(direction), "_terms")]] <- vapply(runs$run_id,
      function(id) sum(results$run_id == id & results$regulated == direction), integer(1L))
    runs[[paste0("significant_", tolower(direction), "_terms")]] <- vapply(runs$run_id,
      function(id) sum(results$run_id == id & results$regulated == direction & results$q_value < args$fdr), integer(1L))
  }
  dir.create(args$output, recursive = TRUE, showWarnings = FALSE)
  write.csv(runs, file.path(args$input, "lion_visualization_runs.csv"), row.names = FALSE)
  write.csv(columns, file.path(args$input, "lion_visualization_columns.csv"), row.names = FALSE)
  write.csv(results, file.path(args$input, "lion_ranking_results_with_direction.csv"), row.names = FALSE, na = "")
  write.csv(terms, file.path(args$input, "lion_term_selection.csv"), row.names = FALSE)
  write.csv(grid, file.path(args$input, "figure2_lion_enrichment_source_data.csv"), row.names = FALSE, na = "")
  write.csv(grid[grid$selected, ], file.path(args$input, "figure2_lion_bubble_source_data.csv"), row.names = FALSE, na = "")
  fill_limits <- c(-log10(args$fdr), max(-log10(args$fdr) + 0.5, ceiling(max(grid$log10_fdr, na.rm = TRUE))))
  main_ids <- terms$term_id[terms$selected]
  main <- lion_make_plot(grid, runs, main_ids, args$fdr, fill_limits,
                         title = NULL, subtitle = NULL)
  pages <- split(terms$term_id, ceiling(seq_len(nrow(terms)) / args$rows_per_page))
  page_plots <- lapply(seq_along(pages), function(i) {
    lion_make_plot(grid, runs, pages[[i]], args$fdr, fill_limits,
                   "LION enrichment: all returned terms",
                   paste0("Page ", i, " / ", length(pages), "; grouped by direction, then minimum FDR"))
  })
  # Measure all labels, then share the same column widths on every page.
  aligned <- lion_align_widths(c(list(main), page_plots))
  main <- aligned[[1L]]
  page_plots <- aligned[-1L]
  main_stem <- "lion_ontology_bubble"
  lion_save_bundle(main$plot, main_stem, args$output, args$width_mm, main$height_mm, args$dpi)

  # The former main-figure stem is a byte-identical compatibility alias.
  for (extension in c("pdf", "svg", "png", "tiff")) {
    file.copy(file.path(args$output, paste0(main_stem, ".", extension)),
              file.path(args$output, paste0("lion_ranked_enrichment.", extension)), overwrite = TRUE)
  }
  supp_height <- max(vapply(page_plots, function(x) x$height_mm, numeric(1L)))
  supp_stem <- "supplementary_lion_ranked_enrichment"
  for (i in seq_along(page_plots)) {
    stem <- paste0(supp_stem, "_page_", sprintf("%02d", i))
    lion_save_bundle(page_plots[[i]]$plot, stem, args$output, args$width_mm, supp_height, args$dpi)
  }
  grDevices::cairo_pdf(file.path(args$output, paste0(supp_stem, ".pdf")), width = args$width_mm / 25.4,
                       height = supp_height / 25.4, family = "Helvetica", onefile = TRUE)
  for (page in page_plots) lion_draw_plot(page$plot)
  grDevices::dev.off()
  for (extension in c("svg", "png", "tiff")) {
    file.copy(file.path(args$output, paste0(supp_stem, "_page_01.", extension)),
              file.path(args$output, paste0(supp_stem, ".", extension)), overwrite = TRUE)
  }
  notes <- c(
    "# Data-driven LION visualization", "",
    "Figure contract: a quantitative grid identifying ontology enrichment toward UP or DOWN in the signed ranking at each main peak.",
    "One ggplot panel contains both display columns. Direction and evidence are visual summaries of the existing web test, not independent analyses.",
    "Only one two-sided KS run per classified main peak is read; candidates and legacy one-tailed runs are excluded.",
    "Each input retains the complete eligible signed-z ranking, including nonsignificant lipids; membership and scores are checked against its audit.",
    "Every age has Upregulated and Downregulated display columns of the SAME two-sided KS test; no second test or new FDR is computed.",
    "Direction comes from the result Regulated field (UP/DOWN), checked against the sign of ES when supplied. Missing or conflicting direction information stops plotting.",
    "A two-sided q value indicates a rank-distribution difference; it does not by itself establish up/downregulation or a large abundance change.",
    "No manually defined biological groups, fixed term IDs, or fixed term-name lists are used.",
    paste0("Selection: q < ", args$fdr, " in at least one run; top_n = ", args$top_n, " (0 = all eligible terms)."),
    "Selection by minimum q precedes display grouping. Ordering: UP, DOWN, direction-varying terms; minimum q within each group, then significant-run count and term ID.",
    paste0("Returned rows: ", nrow(results), "; unique terms: ", nrow(terms),
           "; eligible terms: ", sum(terms$eligible), "; displayed main terms: ", length(main_ids), "."),
    "Each returned term/run is plotted exactly once in its reported direction. The opposite cell is blank with status=opposite_direction and missing P/q/ES/count values.",
    "Missing terms retain status=not_returned in both display columns; these are not two independent missing tests. Missing values are never set to q=1.",
    "Filled circles: q below threshold. Open circles: returned but not significant. Crosses: term not returned in that age's result.",
    "The bottom caption is hidden; its symbol definitions are retained in this document.",
    "Red represents UP and blue DOWN; colour is direction * -log10(q), not the LION ES, NES or an abundance effect. Circle area encodes 100*Annotated/input lipids.",
    "UP/DOWN describe enrichment orientation in the submitted signed ranking; they do not imply every member lipid changed in that direction or pathway activation.",
    "Same-background runs can have equal circle areas. This is not evidence of equal abundance or effect.",
    "The imported q-values are per-web-run FDR values; no pooled cross-run correction is claimed.",
    "LION terms overlap hierarchically. Association-based de-redundancy is not possible from these result tables alone.",
    "Input checksums and result filenames are validated; result CSVs do not independently record the actual web setting.",
    "Differing significance across ages is not a direct statistical test of between-age differences.",
    "These are ontology annotations, not functional measurements of pathway activation, membranes, or lipid droplets.",
    "All pages share measured gtable column widths, keeping the two direction columns of each main peak aligned.",
    paste0("Export dimensions: ", args$width_mm, " mm wide; raster resolution: ", args$dpi, " dpi."),
    "lion_ranked_enrichment is a compatibility alias of lion_ontology_bubble.",
    "The supplementary PDF includes every term; its unsuffixed PNG/SVG/TIFF preview shows page 1 only.",
    "Source data: figure2_lion_enrichment_source_data.csv (all cells), figure2_lion_bubble_source_data.csv (main cells),",
    "lion_term_selection.csv (every inclusion/exclusion decision), lion_visualization_runs.csv (run provenance).", "",
    "lion_ranking_results_with_direction.csv retains every original term/run P/q/ES once; lion_visualization_columns.csv maps the two display columns to the same web run.",
    paste0("Runtime: ", R.version.string),
    "LION/web definitions: https://www.lipidontology.com/faq.html"
  )
  writeLines(notes, file.path(args$input, "README_lion_visualization.md"))
  writeLines(capture.output(sessionInfo()), file.path(args$input, "lion_visualization_session_info.txt"))
  message("Main view: ", length(main_ids), " / ", nrow(terms), " terms, ", nrow(runs), " contrasts.")
  message("Significant counts: ", paste(paste0(runs$run_id, "=", runs$significant_terms), collapse = "; "))
  message("Figures: ", normalizePath(args$output, winslash = "/"))
  invisible(list(runs = runs, terms = terms, data = grid))
}

if (!isTRUE(getOption("lion.visualizer.functions_only", FALSE))) lion_visualizer_main()
