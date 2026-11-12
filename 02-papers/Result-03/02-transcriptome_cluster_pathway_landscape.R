# =============================================================================
# Transcriptomic pathway landscape across five molecular-age clusters
#
# Input:
#   1. Five-cluster assignments exported by 01-molecule_clustering.py
#   2. A frozen local Reactome GMT file and offline GO BP / KEGG annotations
#
# Analysis:
#   - Transcriptomics genes only; identifiers are canonicalized to upper case.
#   - Each database uses its own annotated clustered-gene background.
#   - GO BP / KEGG use unique, unambiguously mapped Entrez genes; Reactome
#     retains the original measured-symbol analysis.
#   - For each cluster, pathway over-representation is tested against the other
#     four clusters with the one-sided Fisher/hypergeometric test.
#   - BH correction is applied within each cluster AND database across all
#     eligible terms, including terms with zero overlapping genes.
#   - Display terms are selected independently per cluster and de-redundified by
#     measured-gene Jaccard similarity. The full test table is always retained.
#
# Figure:
#   - x: molecular-age cluster (C1-C5)
#   - y: representative pathways, grouped by the cluster that selected them
#   - size: gene ratio (term genes / database-annotated genes in the cluster)
#   - colour: identity of the tested cluster (matrix column)
#   - filled: FDR-significant enrichment; open: non-significant overlap
#   - small neutral dots locate cells with no overlapping genes
#
# This is over-representation analysis, not GSEA. The cluster output contains no
# continuous per-gene ranking statistic from which a defensible NES can be made.
# =============================================================================

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), winslash = "/")))
  }
  source_file <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(source_file)) {
    return(dirname(normalizePath(source_file, winslash = "/")))
  }
  normalizePath(getwd(), winslash = "/")
}

SCRIPT_DIR <- get_script_dir()
setwd(SCRIPT_DIR)

# Match the reference workflow's optional local Bioconductor library override.
extra_library <- Sys.getenv("TRANSCRIPTOMICS_R_LIBS", unset = "")
if (nzchar(extra_library)) .libPaths(c(extra_library, .libPaths()))

DATABASE_LABELS <- c(Reactome = "Reactome", GO_BP = "GO Biological Process", KEGG = "KEGG")
DATABASE_SLUGS <- c(Reactome = "reactome", GO_BP = "go_bp", KEGG = "kegg")
DATABASES <- unique(trimws(strsplit(Sys.getenv(
  "CLUSTER_PATHWAY_DATABASES", unset = "Reactome,GO_BP,KEGG"
), ",", fixed = TRUE)[[1]]))
if (!length(DATABASES) || any(!DATABASES %in% names(DATABASE_LABELS))) {
  stop("CLUSTER_PATHWAY_DATABASES must be a comma-separated subset of Reactome,GO_BP,KEGG")
}

required_packages <- c(
  "dplyr", "ggplot2", "jsonlite", "ragg", "scales", "stringr", "svglite", "systemfonts", "tidyr"
)
if (any(c("GO_BP", "KEGG") %in% DATABASES)) {
  required_packages <- c(required_packages, "AnnotationDbi", "org.Hs.eg.db")
}
if ("GO_BP" %in% DATABASES) required_packages <- c(required_packages, "GO.db")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))
]
if (length(missing_packages) > 0L) {
  stop("Missing required R packages: ", paste(missing_packages, collapse = ", "))
}

# =============================================================================
# 1. Configuration
# =============================================================================

GROUP_IDS <- 1:5
FDR_THRESHOLD <- as.numeric(Sys.getenv("CLUSTER_PATHWAY_FDR", unset = "0.05"))
MIN_SET_SIZE <- as.integer(Sys.getenv("CLUSTER_PATHWAY_MIN_SET_SIZE", unset = "5"))
MAX_SET_FRACTION <- as.numeric(Sys.getenv("CLUSTER_PATHWAY_MAX_SET_FRACTION", unset = "0.80"))
TOP_N_PER_CLUSTER <- as.integer(Sys.getenv("CLUSTER_PATHWAY_TOP_N", unset = "4"))
DISPLAY_MAX_SET_FRACTION <- as.numeric(Sys.getenv(
  "CLUSTER_PATHWAY_DISPLAY_MAX_SET_FRACTION", unset = "0.15"
))
DISPLAY_MIN_OVERLAP <- as.integer(Sys.getenv(
  "CLUSTER_PATHWAY_DISPLAY_MIN_OVERLAP", unset = "2"
))
DISPLAY_JACCARD_CUTOFF <- as.numeric(Sys.getenv(
  "CLUSTER_PATHWAY_DISPLAY_JACCARD", unset = "0.50"
))
DISPLAY_EXCLUDE_TERMS <- Sys.getenv(
  "CLUSTER_PATHWAY_DISPLAY_EXCLUDE_TERMS",
  unset = "disease;therapeutic;SARS;COVID-19;COVID"
)
# Retain the capped FDR column for source-data compatibility; hue is categorical.
FDR_COLOUR_CAP <- as.numeric(Sys.getenv("CLUSTER_PATHWAY_FDR_COLOUR_CAP", unset = "10"))

# Fixed final width: typography and column geometry are designed for 100 mm.
FIGURE_WIDTH_MM <- 100
FIGURE_HEIGHT_MM <- as.numeric(Sys.getenv("CLUSTER_PATHWAY_FIGURE_HEIGHT_MM", unset = "100"))
PREVIEW_DPI <- as.integer(Sys.getenv("CLUSTER_PATHWAY_PREVIEW_DPI", unset = "450"))
TIFF_DPI <- as.integer(Sys.getenv("CLUSTER_PATHWAY_TIFF_DPI", unset = "600"))
PDF_DEVICE <- Sys.getenv("CLUSTER_PATHWAY_PDF_DEVICE", unset = "cairo_pdf")
FIGURE_FONT <- if (PDF_DEVICE == "pdf") "Helvetica" else "Arial"
PATHWAY_FONT_PT <- 6.5
CLUSTER_FONT_PT <- 7.0
LEGEND_FONT_PT <- 6.0
LABEL_LINEHEIGHT <- 0.98
PLOT_MARGIN_MM <- 2.0
PANEL_WIDTH_MM <- FIGURE_WIDTH_MM - 2 * PLOT_MARGIN_MM
LABEL_BLOCK_LEFT <- 0.25
LABEL_X <- 1.75
LABEL_BLOCK_RIGHT <- 51.5
LABEL_TEXT_WIDTH_MM <- LABEL_BLOCK_RIGHT - LABEL_X - 1.5
MATRIX_FIRST_X <- LABEL_BLOCK_RIGHT + 3.3
MATRIX_LAST_X <- PANEL_WIDTH_MM - 2.5
BUBBLE_MAX_MM <- 3.5
ROW_MIN_HEIGHT_MM <- 3.5
ROW_VERTICAL_PADDING_MM <- 0.75

CLUSTER_FILE <- Sys.getenv(
  "MOLECULE_CLUSTER_FILE",
  unset = file.path(
    "outputs", "molecule_clustering", "molecule_cluster_assignments_agglomerative.csv"
  )
)
REACTOME_GMT <- Sys.getenv(
  "REACTOME_GMT_FILE",
  unset = file.path("..", "Result-02", "resources", "reactome_gmt", "ReactomePathways.gmt")
)
KEGG_ANNOTATION_FILE <- Sys.getenv(
  "KEGG_ANNOTATION_FILE",
  unset = file.path("resources", "kegg_hsa", "KEGG_offline_data_hsa.RData")
)
OUTPUT_DIR <- Sys.getenv(
  "CLUSTER_PATHWAY_OUTPUT_DIR",
  unset = file.path("outputs", "transcriptome_cluster_pathway_landscape")
)
FIGURE_DIR <- Sys.getenv(
  "CLUSTER_PATHWAY_FIGURE_DIR",
  unset = file.path("figures", "transcriptome_cluster_pathway_landscape")
)
FIGURE_STEM <- file.path(FIGURE_DIR, "transcriptome_cluster_pathway_landscape")

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)

assert_scalar_range <- function(value, name, lower, upper, inclusive_upper = TRUE) {
  valid_upper <- if (inclusive_upper) value <= upper else value < upper
  if (length(value) != 1L || !is.finite(value) || value <= lower || !valid_upper) {
    stop(name, " must be in (", lower, if (inclusive_upper) ", " else ", ", upper,
         if (inclusive_upper) "]" else ")")
  }
}

assert_file <- function(path) {
  if (!file.exists(path)) stop("Required input does not exist: ", path)
  normalizePath(path, winslash = "/")
}

assert_columns <- function(frame, required, object_name) {
  missing <- setdiff(required, names(frame))
  if (length(missing) > 0L) {
    stop(object_name, " is missing columns: ", paste(missing, collapse = ", "))
  }
}

assert_scalar_range(FDR_THRESHOLD, "CLUSTER_PATHWAY_FDR", 0, 1, FALSE)
assert_scalar_range(MAX_SET_FRACTION, "CLUSTER_PATHWAY_MAX_SET_FRACTION", 0, 1, TRUE)
assert_scalar_range(
  DISPLAY_MAX_SET_FRACTION, "CLUSTER_PATHWAY_DISPLAY_MAX_SET_FRACTION", 0, 1, TRUE
)
assert_scalar_range(DISPLAY_JACCARD_CUTOFF, "CLUSTER_PATHWAY_DISPLAY_JACCARD", 0, 1, TRUE)
if (MIN_SET_SIZE < 2L) stop("CLUSTER_PATHWAY_MIN_SET_SIZE must be >= 2")
if (TOP_N_PER_CLUSTER < 1L) stop("CLUSTER_PATHWAY_TOP_N must be >= 1")
if (DISPLAY_MIN_OVERLAP < 1L) stop("CLUSTER_PATHWAY_DISPLAY_MIN_OVERLAP must be >= 1")
if (!is.finite(FDR_COLOUR_CAP) || FDR_COLOUR_CAP <= 0) {
  stop("CLUSTER_PATHWAY_FDR_COLOUR_CAP must be positive")
}
if (!PDF_DEVICE %in% c("pdf", "cairo_pdf")) {
  stop("CLUSTER_PATHWAY_PDF_DEVICE must be 'pdf' or 'cairo_pdf'")
}

CLUSTER_FILE <- assert_file(CLUSTER_FILE)
if ("Reactome" %in% DATABASES) REACTOME_GMT <- assert_file(REACTOME_GMT)
if ("KEGG" %in% DATABASES && !file.exists(KEGG_ANNOTATION_FILE)) {
  stop("Missing KEGG snapshot. Run update_cluster_pathway_kegg.R first, or set KEGG_ANNOTATION_FILE.")
}
source(file.path(SCRIPT_DIR, "transcriptome_cluster_pathway_methods.R"), local = TRUE)

# =============================================================================
# 2. Input parsing and integrity checks
# =============================================================================

canonical_gene <- function(x) {
  toupper(trimws(as.character(x)))
}

cluster_table <- utils::read.csv(
  CLUSTER_FILE, stringsAsFactors = FALSE, check.names = FALSE, encoding = "UTF-8"
)
assert_columns(
  cluster_table, c("feature", "omics", "pattern_id", "pattern"), "cluster assignment table"
)

assignments <- cluster_table |>
  dplyr::filter(.data$omics == "Transcriptomics") |>
  dplyr::transmute(
    feature_original = trimws(as.character(.data$feature)),
    feature = canonical_gene(.data$feature),
    group_id = as.integer(.data$pattern_id),
    group_label = trimws(as.character(.data$pattern))
  ) |>
  dplyr::filter(!is.na(.data$feature), nzchar(.data$feature))

if (nrow(assignments) == 0L) stop("No Transcriptomics rows were found in the cluster file")
if (!setequal(sort(unique(assignments$group_id)), GROUP_IDS)) {
  stop("Transcriptomics pattern_id values must contain exactly 1, 2, 3, 4 and 5")
}
if (anyDuplicated(assignments$feature)) {
  duplicate_examples <- unique(assignments$feature[duplicated(assignments$feature)])
  stop(
    "Transcript genes have duplicate cluster assignments: ",
    paste(utils::head(duplicate_examples, 10L), collapse = ", ")
  )
}

group_label_table <- assignments |>
  dplyr::distinct(.data$group_id, .data$group_label)
label_conflicts <- group_label_table |>
  dplyr::count(.data$group_id, name = "n_labels") |>
  dplyr::filter(.data$n_labels != 1L)
if (nrow(label_conflicts) > 0L) stop("Each cluster must have exactly one trajectory label")
group_labels <- stats::setNames(
  group_label_table$group_label, as.character(group_label_table$group_id)
)

read_reactome_gmt <- function(path, measured_genes) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  records <- strsplit(lines, "\t", fixed = TRUE)
  records <- records[lengths(records) >= 3L]
  is_human <- vapply(
    records, function(record) startsWith(record[[2]], "R-HSA-"), logical(1)
  )
  records <- records[is_human]
  if (length(records) == 0L) stop("No human R-HSA Reactome records were found")

  term_id <- vapply(records, function(record) record[[2]], character(1))
  term <- vapply(records, function(record) trimws(record[[1]]), character(1))
  measured_members <- lapply(records, function(record) {
    intersect(unique(canonical_gene(record[-c(1, 2)])), measured_genes)
  })
  keep <- lengths(measured_members) > 0L
  term_id <- term_id[keep]
  term <- term[keep]
  measured_members <- measured_members[keep]

  terms <- data.frame(term_id = term_id, term = term, stringsAsFactors = FALSE) |>
    dplyr::distinct(.data$term_id, .keep_all = TRUE)
  membership <- dplyr::bind_rows(lapply(seq_along(measured_members), function(i) {
    data.frame(
      term_id = term_id[[i]], feature = measured_members[[i]], stringsAsFactors = FALSE
    )
  })) |>
    dplyr::distinct(.data$term_id, .data$feature)

  list(
    terms = terms,
    membership = membership,
    human_terms = sum(is_human),
    terms_with_measured_genes = nrow(terms)
  )
}

# =============================================================================
# 3. Database-specific annotation backgrounds and exact over-representation
# =============================================================================

entrez_mapping <- NULL
if (any(c("GO_BP", "KEGG") %in% DATABASES)) {
  message("Mapping measured transcript identifiers to unique Entrez genes for GO BP / KEGG...")
  entrez_mapping <- landscape_map_entrez(assignments)
}
identity_map <- assignments |>
  dplyr::transmute(analysis_feature = .data$feature, measured_feature = .data$feature)

database_contexts <- stats::setNames(lapply(DATABASES, function(database) {
  message("Preparing ", DATABASE_LABELS[[database]], " annotations...")
  if (database == "Reactome") {
    annotation <- read_reactome_gmt(REACTOME_GMT, assignments$feature)
    annotation$provenance <- list(
      mode = "frozen human Reactome GMT, measured symbols",
      annotation_file = REACTOME_GMT,
      annotation_md5 = unname(tools::md5sum(REACTOME_GMT)),
      human_terms_in_gmt = annotation$human_terms
    )
    analysis_assignments <- assignments
    feature_map <- identity_map
  } else {
    annotation <- if (database == "GO_BP") {
      landscape_load_go_bp(entrez_mapping$assignments$feature)
    } else {
      landscape_load_kegg(
        entrez_mapping$assignments$feature, KEGG_ANNOTATION_FILE
      )
    }
    analysis_assignments <- entrez_mapping$assignments
    feature_map <- entrez_mapping$feature_map
  }
  landscape_prepare_database(
    annotation, analysis_assignments, feature_map, DATABASE_LABELS[[database]],
    MIN_SET_SIZE, MAX_SET_FRACTION
  )
}), DATABASES)

ora_by_database <- lapply(database_contexts, landscape_run_ora,
  group_ids = GROUP_IDS, group_labels = group_labels,
  fdr_threshold = FDR_THRESHOLD, fdr_cap = FDR_COLOUR_CAP
)
ora_results <- dplyr::bind_rows(ora_by_database) |>
  dplyr::arrange(
    .data$group_id, .data$fdr, dplyr::desc(.data$log2_odds_ratio), .data$source, .data$term_id
  )

# Cross-database display redundancy is evaluated in the SAME measured-feature
# identifier space; Entrez IDs must never be compared directly with symbols.
display_term_sets <- list()
for (context in database_contexts) {
  sets <- context$display_sets
  names(sets) <- landscape_term_key(context$source, names(sets))
  display_term_sets <- c(display_term_sets, sets)
}
database_summary <- dplyr::bind_rows(lapply(names(database_contexts), function(database) {
  context <- database_contexts[[database]]
  data.frame(
    database = database, source = context$source,
    analysis_unit = if (database == "Reactome") "measured gene symbol" else "unique Entrez gene",
    annotated_universe_size = length(context$universe),
    terms_with_measured_genes = context$terms_with_measured_genes,
    eligible_tested_terms_per_cluster = length(context$term_sets),
    total_cluster_term_tests = nrow(ora_by_database[[database]])
  )
}))

# =============================================================================
# 4. Representative pathway selection with Jaccard de-redundancy
# =============================================================================

jaccard_similarity <- function(a, b) {
  union_size <- length(union(a, b))
  if (union_size == 0L) return(0)
  length(intersect(a, b)) / union_size
}

display_set_limit <- function(universe_size) {
  pmax(MIN_SET_SIZE, floor(DISPLAY_MAX_SET_FRACTION * universe_size))
}
display_exclude_terms <- trimws(strsplit(DISPLAY_EXCLUDE_TERMS, ";", fixed = TRUE)[[1]])
display_exclude_terms <- display_exclude_terms[nzchar(display_exclude_terms)]
is_excluded_display_term <- function(term) {
  if (length(display_exclude_terms) == 0L) return(rep(FALSE, length(term)))
  vapply(
    as.character(term),
    function(value) {
      value_lower <- tolower(value)
      any(vapply(
        display_exclude_terms,
        function(pattern) grepl(tolower(pattern), value_lower, fixed = TRUE),
        logical(1)
      ))
    },
    logical(1)
  )
}

# Disease- or treatment-labelled terms remain in the complete inference table.
# They are excluded only from the compact representative-label ranking because
# the figure's question is functional ageing biology rather than disease therapy.
display_candidate_audit <- ora_results |>
  dplyr::mutate(
    base_display_candidate =
      .data$overlap_group >= DISPLAY_MIN_OVERLAP &
      .data$log2_odds_ratio > 0 &
      .data$set_size <= display_set_limit(.data$universe_size),
    excluded_by_label_rule =
      .data$base_display_candidate &
      is_excluded_display_term(.data$term)
  ) |>
  dplyr::group_by(.data$source, .data$group_id, .data$group_label) |>
  dplyr::summarise(
    base_display_candidates = sum(.data$base_display_candidate),
    excluded_by_label_rule_count = sum(.data$excluded_by_label_rule),
    eligible_after_label_rule = sum(
      .data$base_display_candidate & !.data$excluded_by_label_rule
    ),
    fdr_significant_after_label_rule = sum(
      .data$base_display_candidate & !.data$excluded_by_label_rule &
        .data$fdr < FDR_THRESHOLD
    ),
    .groups = "drop"
  ) |>
  dplyr::arrange(.data$source, .data$group_id)

select_representatives <- function(group_id) {
  candidates <- ora_results |>
    dplyr::filter(
      .data$group_id == .env$group_id,
      .data$overlap_group >= DISPLAY_MIN_OVERLAP,
      .data$log2_odds_ratio > 0,
      .data$set_size <= display_set_limit(.data$universe_size),
      !is_excluded_display_term(.data$term)
    )
  # A non-significant cluster remains a matrix column, but does not select
  # its own pathway block. Apply the same significance rule to every cluster.
  pool <- candidates |>
    dplyr::filter(.data$fdr < FDR_THRESHOLD) |>
    dplyr::arrange(
      .data$fdr, .data$p_value, dplyr::desc(.data$log2_odds_ratio),
      dplyr::desc(.data$overlap_group), .data$source, .data$term_id
    )
  if (nrow(pool) == 0L) {
    message("C", group_id, ": no FDR-significant display candidates; retaining its matrix column only.")
  }

  selected_rows <- integer(0)
  max_similarity <- numeric(0)
  for (i in seq_len(nrow(pool))) {
    term_key <- landscape_term_key(pool$source[[i]], pool$term_id[[i]])
    similarities <- if (length(selected_rows) == 0L) {
      numeric(0)
    } else {
      vapply(
        landscape_term_key(pool$source[selected_rows], pool$term_id[selected_rows]),
        function(selected_key) jaccard_similarity(
          display_term_sets[[term_key]], display_term_sets[[selected_key]]
        ),
        numeric(1)
      )
    }
    current_max <- if (length(similarities) == 0L) 0 else max(similarities)
    if (current_max < DISPLAY_JACCARD_CUTOFF) {
      selected_rows <- c(selected_rows, i)
      max_similarity <- c(max_similarity, current_max)
    }
    if (length(selected_rows) >= TOP_N_PER_CLUSTER) break
  }

  chosen <- pool[selected_rows, , drop = FALSE]
  chosen$source_cluster <- rep.int(group_id, nrow(chosen))
  chosen$source_cluster_label <- rep.int(unname(group_labels[as.character(group_id)]), nrow(chosen))
  chosen$selection_status <- rep.int("fdr_significant", nrow(chosen))
  chosen$max_jaccard_to_earlier_selected <- max_similarity
  chosen$display_rank <- seq_len(nrow(chosen))
  chosen
}

selected_pathways <- dplyr::bind_rows(lapply(GROUP_IDS, select_representatives)) |>
  dplyr::arrange(.data$source_cluster, .data$display_rank)

if (nrow(selected_pathways) == 0L) {
  stop("No FDR-significant representative pathways remain in any cluster")
}
DISPLAY_GROUP_IDS <- sort(unique(selected_pathways$source_cluster))
omitted_row_clusters <- setdiff(GROUP_IDS, DISPLAY_GROUP_IDS)

# =============================================================================
# 5. Source-data tables and figure layout data
# =============================================================================

selected_rows <- selected_pathways |>
  dplyr::transmute(
    source_cluster = .data$source_cluster,
    source_cluster_label = .data$source_cluster_label,
    selection_status = .data$selection_status,
    display_rank = .data$display_rank,
    source = .data$source,
    term_id = .data$term_id,
    term = .data$term,
    source_fdr = .data$fdr,
    source_p_value = .data$p_value,
    source_gene_ratio = .data$gene_ratio,
    source_log2_odds_ratio = .data$log2_odds_ratio,
    max_jaccard_to_earlier_selected = .data$max_jaccard_to_earlier_selected
  ) |>
  dplyr::mutate(
    row_id = paste0("C", .data$source_cluster, "__", .data$source, "__", .data$term_id),
    display_order = dplyr::row_number()
  )

# Wrap whole words using font metrics at high resolution, rather than scaling
# the 210-mm text down to an unreadable size or abbreviating pathway names.
wrap_label <- function(x, width_mm = LABEL_TEXT_WIDTH_MM) {
  width_of <- function(text) {
    systemfonts::string_width(
      text, family = FIGURE_FONT, size = PATHWAY_FONT_PT, res = 720
    ) * 25.4 / 720
  }
  vapply(as.character(x), function(value) {
    words <- strsplit(trimws(value), "[[:space:]]+")[[1]]
    if (any(width_of(words) > width_mm)) {
      stop("A pathway word exceeds the available label width: ", value)
    }
    lines <- character(0)
    current <- ""
    for (word in words) {
      candidate <- if (nzchar(current)) paste(current, word) else word
      if (nzchar(current) && width_of(candidate) > width_mm) {
        lines <- c(lines, current)
        current <- word
      } else {
        current <- candidate
      }
    }
    paste(c(lines, current), collapse = "\n")
  }, character(1))
}

# Multiline rows receive additional height; all five bubbles remain centred
# on the full pathway label. Group spacing reserves room for the heading.
GROUP_GAP <- 5.5
LABEL_LINE_STEP_MM <- PATHWAY_FONT_PT * 25.4 / 72 * 1.2 * LABEL_LINEHEIGHT
selected_rows <- selected_rows |>
  dplyr::mutate(
    display_label = wrap_label(.data$term),
    label_lines = stringr::str_count(.data$display_label, "\n") + 1L,
    row_height_mm = pmax(
      ROW_MIN_HEIGHT_MM, PATHWAY_FONT_PT * 25.4 / 72 * 1.2 +
        (.data$label_lines - 1L) * LABEL_LINE_STEP_MM + ROW_VERTICAL_PADDING_MM
    ),
    y = -(cumsum(.data$row_height_mm) - .data$row_height_mm / 2) -
      (match(.data$source_cluster, DISPLAY_GROUP_IDS) - 1L) * GROUP_GAP
  )

plot_data <- tidyr::crossing(
  row_id = selected_rows$row_id,
  cluster_id = GROUP_IDS
) |>
  dplyr::left_join(selected_rows, by = "row_id") |>
  dplyr::left_join(
    ora_results |>
      dplyr::select(
        "group_id", "source", "term_id", "group_label", "universe_size", "set_size",
        "group_size", "rest_size", "overlap_group", "overlap_rest", "gene_ratio",
        "background_ratio", "fold_enrichment", "log2_odds_ratio", "p_value", "fdr",
        "minus_log10_fdr", "minus_log10_fdr_capped", "significant"
      ) |>
      dplyr::rename(cluster_id = "group_id", cluster_label = "group_label"),
    by = c("cluster_id", "source", "term_id"), relationship = "many-to-one"
  ) |>
  dplyr::arrange(.data$display_order, .data$cluster_id)

if (anyNA(plot_data$fdr)) stop("The display matrix contains missing ORA results")

cluster_colours <- c(
  `1` = "#E64B35",
  `2` = "#4DBBD5",
  `3` = "#00A087",
  `4` = "#3C5488",
  `5` = "#F39B7F"
)

group_bands <- selected_rows |>
  dplyr::group_by(.data$source_cluster, .data$source_cluster_label, .data$selection_status) |>
  dplyr::summarise(
    ymin = min(.data$y - .data$row_height_mm / 2) - 0.40,
    ymax = max(.data$y + .data$row_height_mm / 2),
    ymid = mean(range(.data$y)),
    .groups = "drop"
  ) |>
  dplyr::mutate(
    group_colour = unname(cluster_colours[as.character(.data$source_cluster)]),
    group_fill = scales::alpha(.data$group_colour, 0.10),
    heading_y = .data$ymax + 2.10,
    group_display = paste0("C", .data$source_cluster)
  )

write_source_csv <- function(frame, filename) {
  utils::write.csv(frame, file.path(OUTPUT_DIR, filename), row.names = FALSE, na = "")
}
write_source_csv(ora_results, "all_databases_cluster_vs_rest_ora.csv")
write_source_csv(database_summary, "database_analysis_summary.csv")
for (database in names(database_contexts)) {
  context <- database_contexts[[database]]
  slug <- DATABASE_SLUGS[[database]]
  write_source_csv(ora_by_database[[database]], paste0(slug, "_cluster_vs_rest_ora.csv"))
  write_source_csv(context$term_catalog, paste0(slug, "_tested_term_catalog.csv"))
  write_source_csv(context$assignments, paste0(slug, "_annotated_cluster_genes.csv"))
  write_source_csv(
    context$measured_membership |>
      dplyr::rename(feature = "measured_feature") |>
      dplyr::arrange(.data$term_id, .data$feature),
    paste0(slug, "_measured_gene_membership.csv")
  )
  if (database != "Reactome") {
    write_source_csv(context$membership, paste0(slug, "_entrez_gene_membership.csv"))
  }
}
if (!is.null(entrez_mapping)) {
  write_source_csv(entrez_mapping$audit, "entrez_mapping_audit.csv")
  write_source_csv(entrez_mapping$coverage, "entrez_mapping_coverage.csv")
  write_source_csv(entrez_mapping$method_coverage, "entrez_mapping_method_coverage.csv")
  write_source_csv(entrez_mapping$assignments, "entrez_analysis_genes.csv")
}
utils::write.csv(
  selected_pathways,
  file.path(OUTPUT_DIR, "representative_pathways_by_cluster.csv"),
  row.names = FALSE,
  na = ""
)
utils::write.csv(
  display_candidate_audit,
  file.path(OUTPUT_DIR, "display_selection_audit.csv"),
  row.names = FALSE,
  na = ""
)
utils::write.csv(
  plot_data,
  file.path(OUTPUT_DIR, "figure_source_data.csv"),
  row.names = FALSE,
  na = ""
)

# =============================================================================
# 6. Publication figure
# =============================================================================

theme_pathway_landscape <- function() {
  ggplot2::theme_void(base_size = PATHWAY_FONT_PT, base_family = FIGURE_FONT) +
    ggplot2::theme(
      plot.title = ggplot2::element_blank(),
      plot.subtitle = ggplot2::element_blank(),
      plot.caption = ggplot2::element_blank(),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.box.just = "left",
      legend.title = ggplot2::element_text(size = PATHWAY_FONT_PT, colour = "#223638"),
      legend.text = ggplot2::element_text(size = LEGEND_FONT_PT, colour = "#3D5359"),
      legend.key.height = grid::unit(4.0, "mm"),
      legend.key.width = grid::unit(4.0, "mm"),
      legend.spacing.x = grid::unit(2.0, "mm"),
      legend.box.spacing = grid::unit(0.8, "mm"),
      legend.margin = ggplot2::margin(0, 0, 0, 0),
      plot.background = ggplot2::element_rect(fill = "white", colour = NA),
      plot.margin = ggplot2::margin(
        PLOT_MARGIN_MM, PLOT_MARGIN_MM, PLOT_MARGIN_MM, PLOT_MARGIN_MM, unit = "mm"
      )
    )
}

size_upper <- max(0.20, ceiling(max(plot_data$gene_ratio, na.rm = TRUE) * 100) / 100)
size_breaks <- c(0.01, 0.05, 0.10, 0.20)
size_breaks <- size_breaks[size_breaks <= size_upper]
omitted_row_note <- if (length(omitted_row_clusters) > 0L) {
  paste0(
    paste0("C", omitted_row_clusters, collapse = ", "),
    ": no display-eligible pathway passed FDR < ",
    format(FDR_THRESHOLD, trim = TRUE),
    "; no dedicated representative row block is shown, but the matrix column is retained."
  )
} else {
  character(0)
}
# The x coordinate is expressed in panel millimetres at the requested width.
# Label blocks and the matrix meet at LABEL_BLOCK_RIGHT with no blank gutter.
# Only the first bubble's radius and a small breathing space separate them.
if (MATRIX_LAST_X <= MATRIX_FIRST_X + 20.0) {
  stop("Figure width is too narrow for the pathway labels and five matrix columns")
}
column_x <- stats::setNames(
  seq(MATRIX_FIRST_X, MATRIX_LAST_X, length.out = length(GROUP_IDS)), GROUP_IDS
)
plot_data$column_x <- unname(column_x[as.character(plot_data$cluster_id)])
COLUMN_LABEL_Y <- max(group_bands$heading_y) + 3.60
column_labels <- data.frame(
  cluster_id = GROUP_IDS,
  column_x = unname(column_x),
  label = paste0("C", GROUP_IDS)
)
column_guides <- tidyr::crossing(
  group_bands |> dplyr::select("source_cluster", "ymin", "ymax"),
  column_x = unname(column_x)
)
# Rounded backgrounds span the cluster heading AND all pathway labels in its
# block. They are drawn first; text remains fully visible above the pale fill.
label_capsules <- lapply(seq_len(nrow(group_bands)), function(i) {
  ggplot2::annotation_custom(
    grid::roundrectGrob(
      r = grid::unit(1.6, "mm"),
      gp = grid::gpar(fill = group_bands$group_fill[[i]], col = NA)
    ),
    xmin = LABEL_BLOCK_LEFT, xmax = LABEL_BLOCK_RIGHT,
    ymin = group_bands$ymin[[i]], ymax = group_bands$heading_y[[i]] + 1.60
  )
})
plot <- ggplot2::ggplot() +
  label_capsules +
  ggplot2::geom_segment(
    data = column_guides,
    ggplot2::aes(
      x = .data$column_x, xend = .data$column_x,
      y = .data$ymin + 0.10, yend = .data$ymax - 0.10
    ),
    inherit.aes = FALSE,
    colour = "#EDF0F2", linewidth = 0.16, linetype = "solid"
  ) +
  ggplot2::geom_segment(
    data = group_bands,
    ggplot2::aes(
      x = LABEL_BLOCK_RIGHT, xend = PANEL_WIDTH_MM - 2.0,
      y = .data$heading_y, yend = .data$heading_y
    ),
    inherit.aes = FALSE,
    colour = "#DDE1E5", linewidth = 0.20, linetype = "solid", lineend = "butt"
  ) +
  ggplot2::geom_text(
    data = group_bands,
    ggplot2::aes(x = LABEL_X, y = .data$heading_y, label = .data$group_display),
    family = FIGURE_FONT, colour = "#263D42",
    size = CLUSTER_FONT_PT / ggplot2::.pt, fontface = "bold", hjust = 0,
    show.legend = FALSE
  ) +
  ggplot2::geom_text(
    data = selected_rows,
    ggplot2::aes(x = LABEL_X, y = .data$y, label = .data$display_label),
    family = FIGURE_FONT, colour = "#354B50",
    size = PATHWAY_FONT_PT / ggplot2::.pt, lineheight = LABEL_LINEHEIGHT, hjust = 0
  ) +
  ggplot2::geom_point(
    data = plot_data |> dplyr::filter(.data$overlap_group == 0L),
    ggplot2::aes(x = .data$column_x, y = .data$y),
    inherit.aes = FALSE,
    shape = 16, size = 0.40, colour = "#BBC9CD"
  ) +
  ggplot2::geom_point(
    data = plot_data |>
      dplyr::filter(.data$overlap_group > 0L, !.data$significant),
    ggplot2::aes(
      x = .data$column_x, y = .data$y, size = .data$gene_ratio,
      colour = factor(.data$cluster_id, levels = GROUP_IDS)
    ),
    inherit.aes = FALSE,
    shape = 21, fill = "white", stroke = 0.18,
    show.legend = FALSE
  ) +
  ggplot2::geom_point(
    data = plot_data |> dplyr::filter(.data$significant),
    ggplot2::aes(
      x = .data$column_x, y = .data$y,
      size = .data$gene_ratio,
      fill = factor(.data$cluster_id, levels = GROUP_IDS),
      colour = factor(.data$cluster_id, levels = GROUP_IDS)
    ),
    inherit.aes = FALSE,
    shape = 21, stroke = 0.18
  ) +
  ggplot2::geom_text(
    data = column_labels,
    ggplot2::aes(x = .data$column_x, y = COLUMN_LABEL_Y, label = .data$label),
    family = FIGURE_FONT, size = CLUSTER_FONT_PT / ggplot2::.pt,
    fontface = "bold", colour = "#263D42"
  ) +
  ggplot2::scale_colour_manual(
    values = cluster_colours, limits = as.character(GROUP_IDS), guide = "none"
  ) +
  ggplot2::scale_fill_manual(
    values = cluster_colours, limits = as.character(GROUP_IDS), guide = "none"
  ) +
  ggplot2::scale_size_area(
    name = "Gene ratio",
    max_size = BUBBLE_MAX_MM,
    limits = c(0, size_upper),
    breaks = size_breaks,
    labels = scales::label_percent(accuracy = 1),
    guide = ggplot2::guide_legend(
      order = 1, title.position = "left", nrow = 1,
      override.aes = list(fill = "#BBC4CA", colour = "#72818A", stroke = 0.18)
    )
  ) +
  ggplot2::scale_x_continuous(limits = c(0, PANEL_WIDTH_MM), expand = c(0, 0)) +
  ggplot2::scale_y_continuous(
    limits = c(min(group_bands$ymin) - 0.50, COLUMN_LABEL_Y + 1.80),
    expand = c(0, 0)
  ) +
  ggplot2::coord_cartesian(clip = "off") +
  ggplot2::labs(
    title = NULL,
    subtitle = NULL,
    x = NULL,
    y = NULL,
    caption = NULL
  ) +
  theme_pathway_landscape()

render_plot_device <- function(open_device, plot_object) {
  open_device()
  device_id <- grDevices::dev.cur()
  on.exit({
    open_devices <- grDevices::dev.list()
    if (!is.null(open_devices) && device_id %in% open_devices) {
      grDevices::dev.off(device_id)
    }
  }, add = TRUE)
  print(plot_object)
  grDevices::dev.off(device_id)
  invisible(NULL)
}

save_publication_figure <- function(plot_object, stem) {
  width_in <- FIGURE_WIDTH_MM / 25.4
  height_in <- FIGURE_HEIGHT_MM / 25.4
  render_plot_device(
    function() svglite::svglite(
      paste0(stem, ".svg"), width = width_in, height = height_in,
      bg = "white", pointsize = 7
    ),
    plot_object
  )
  if (PDF_DEVICE == "cairo_pdf") {
    render_plot_device(
      function() grDevices::cairo_pdf(
        paste0(stem, ".pdf"), width = width_in, height = height_in,
        family = FIGURE_FONT, onefile = TRUE, bg = "white"
      ),
      plot_object
    )
  } else {
    render_plot_device(
      function() grDevices::pdf(
        paste0(stem, ".pdf"), width = width_in, height = height_in,
        family = "Helvetica", onefile = TRUE, bg = "white", useDingbats = FALSE
      ),
      plot_object
    )
  }
  render_plot_device(
    function() ragg::agg_tiff(
      paste0(stem, ".tiff"), width = width_in, height = height_in,
      units = "in", res = TIFF_DPI, compression = "lzw", background = "white"
    ),
    plot_object
  )
  # Stage the preview before replacing an existing file. A failed device write
  # must not leave an old preview alongside newly exported vector figures.
  preview_file <- paste0(stem, ".png")
  preview_temporary <- tempfile("landscape-preview-", tmpdir = dirname(stem), fileext = ".png")
  on.exit(unlink(preview_temporary), add = TRUE)
  render_plot_device(
    function() ragg::agg_png(
      preview_temporary, width = width_in, height = height_in,
      units = "in", res = PREVIEW_DPI, background = "white"
    ),
    plot_object
  )
  if (!file.exists(preview_temporary) || file.info(preview_temporary)$size <= 0 ||
      !file.copy(preview_temporary, preview_file, overwrite = TRUE)) {
    stop("Could not replace the PNG preview: ", preview_file)
  }
}

save_publication_figure(plot, FIGURE_STEM)

# The scientific content uses one ggplot plot area. A multi-panel alignment
# comparison is therefore not applicable, but the decision is preserved in JSON.
jsonlite::write_json(
  list(
    verdict = "NOT APPLICABLE",
    reason = "Single ggplot plot area; the five clusters share one coordinate system.",
    backend = "R",
    tolerance_pt = 1.5
  ),
  paste0(FIGURE_STEM, ".alignment.json"),
  pretty = TRUE,
  auto_unbox = TRUE
)

# =============================================================================
# 7. Reproducibility manifest and figure contract
# =============================================================================

full_counts <- assignments |>
  dplyr::count(.data$group_id, name = "clustered_transcript_genes") |>
  dplyr::arrange(.data$group_id)
significant_counts <- ora_results |>
  dplyr::group_by(.data$source, .data$group_id) |>
  dplyr::summarise(
    fdr_significant_enriched_terms = sum(.data$significant), .groups = "drop"
  ) |>
  dplyr::arrange(.data$source, .data$group_id)
annotated_counts <- dplyr::bind_rows(lapply(database_contexts, function(context) {
  context$assignments |>
    dplyr::count(.data$group_id, name = "database_annotated_genes") |>
    dplyr::mutate(source = context$source, universe_size = length(context$universe))
}))
cluster_audit <- annotated_counts |>
  dplyr::left_join(full_counts, by = "group_id") |>
  dplyr::left_join(significant_counts, by = c("source", "group_id")) |>
  dplyr::mutate(group_label = unname(group_labels[as.character(.data$group_id)])) |>
  dplyr::arrange(.data$source, .data$group_id)

utils::write.csv(
  cluster_audit,
  file.path(OUTPUT_DIR, "cluster_analysis_summary.csv"),
  row.names = FALSE,
  na = ""
)

manifest <- list(
  created_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  analysis = "Database-specific cluster-versus-rest over-representation",
  databases = DATABASES,
  backend = "R",
  r_version = R.version.string,
  inputs = list(
    cluster_file = CLUSTER_FILE,
    cluster_file_md5 = unname(tools::md5sum(CLUSTER_FILE)),
    source_code_md5 = as.list(tools::md5sum(c(
      "02-transcriptome_cluster_pathway_landscape.R", "transcriptome_cluster_pathway_methods.R"
    ))),
    annotation_provenance = lapply(database_contexts, function(context) context$provenance)
  ),
  identifier_mapping = if (!is.null(entrez_mapping)) list(
    input_type = entrez_mapping$id_type,
    analysis_units = stats::setNames(as.list(database_summary$analysis_unit), database_summary$database),
    policy = paste(
      "Per identifier: version-stripped Ensembl or original SYMBOL, then case-normalized SYMBOL, then aliases.",
      "A current SYMBOL takes precedence; alias ambiguity is retained for audit, never resolved arbitrarily.",
      "Exclude unmapped and one-to-many identifiers and Entrez IDs assigned to multiple clusters;",
      "collapse same-cluster aliases to one Entrez gene; audit every input feature."
    ),
    accepted_entrez_genes = nrow(entrez_mapping$assignments),
    coverage = entrez_mapping$coverage,
    method_coverage = entrez_mapping$method_coverage,
    orgdb_metadata = entrez_mapping$orgdb_metadata
  ) else NULL,
  backgrounds = list(
    definition = "Per database: all assigned analysis genes with at least one annotation, before set-size filtering",
    summaries = database_summary
  ),
  parameters = list(
    fdr_threshold = FDR_THRESHOLD,
    min_set_size = MIN_SET_SIZE,
    max_set_fraction = MAX_SET_FRACTION,
    top_n_per_cluster = TOP_N_PER_CLUSTER,
    display_max_set_fraction = DISPLAY_MAX_SET_FRACTION,
    display_min_overlap = DISPLAY_MIN_OVERLAP,
    display_jaccard_cutoff = DISPLAY_JACCARD_CUTOFF,
    display_exclude_terms = display_exclude_terms,
    fdr_colour_cap = FDR_COLOUR_CAP,
    pdf_device = PDF_DEVICE,
    fdr_scope = "BH separately within each cluster and database across ALL eligible terms, including zero overlaps",
    display_ranking = "Pool database-specific adjusted results for ranking only; no pooled cross-database FDR claim or database quota",
    display_significance_rule = "Only FDR-significant eligible terms select representative rows; no non-significant fallback",
    jaccard_identifier_space = "original measured-feature identifiers shared across databases",
    test = "one-sided Fisher/hypergeometric over-representation"
  ),
  figure = list(
    style = "white canvas, softly tinted rounded backgrounds spanning each cluster heading and its pathway labels, light-grey separators and cluster-coloured bubbles",
    title_subtitle_caption = "omitted from artwork; methods and interpretation in external figure legend",
    pathway_label_size_pt = PATHWAY_FONT_PT,
    cluster_label_size_pt = CLUSTER_FONT_PT,
    legend_text_size_pt = LEGEND_FONT_PT,
    label_wrap_width_mm = LABEL_TEXT_WIDTH_MM,
    label_wrap_method = "whole words wrapped by high-resolution font metrics; row heights follow line count",
    maximum_bubble_size_mm = BUBBLE_MAX_MM,
    minimum_row_height_layout_mm = ROW_MIN_HEIGHT_MM,
    row_vertical_padding_layout_mm = ROW_VERTICAL_PADDING_MM,
    group_gap_layout_mm = GROUP_GAP,
    legend_layout = "one horizontal line with title on the left",
    fixed_width = TRUE,
    width_mm = FIGURE_WIDTH_MM,
    height_mm = FIGURE_HEIGHT_MM,
    font = FIGURE_FONT,
    preview_dpi = PREVIEW_DPI,
    tiff_dpi = TIFF_DPI,
    bubble_area = "gene_ratio",
    filled_bubbles = "FDR-significant positive enrichment",
    open_bubbles = "positive overlap without significant enrichment",
    locator_dots = "zero overlapping genes; locator size does not encode gene ratio",
    fill_scale = "categorical tested-cluster identity (matrix column); colour does not encode FDR magnitude",
    cluster_palette = stats::setNames(as.list(unname(cluster_colours)), paste0("C", GROUP_IDS)),
    capsule_fill_alpha = 0.10,
    displayed_row_groups = paste0("C", DISPLAY_GROUP_IDS),
    retained_matrix_columns = paste0("C", GROUP_IDS),
    clusters_without_representative_row_block = paste0("C", omitted_row_clusters),
    label_block_right_panel_mm = LABEL_BLOCK_RIGHT,
    first_column_panel_mm = MATRIX_FIRST_X,
    label_block_to_first_column_centre_mm = MATRIX_FIRST_X - LABEL_BLOCK_RIGHT,
    displayed_rows = nrow(selected_rows),
    displayed_cells = nrow(plot_data),
    filled_cells = sum(plot_data$significant),
    open_cells = sum(plot_data$overlap_group > 0L & !plot_data$significant),
    zero_overlap_cells = sum(plot_data$overlap_group == 0L)
  ),
  counts = list(
    clustered_transcript_genes = nrow(assignments),
    per_database = database_summary,
    total_cluster_term_tests = nrow(ora_results)
  ),
  cluster_summary = cluster_audit,
  package_versions = stats::setNames(
    lapply(required_packages, function(package) as.character(utils::packageVersion(package))),
    required_packages
  ),
  interpretation = paste0(
    "The analysis tests relative functional representation among genes assigned to one ",
    "trajectory cluster versus the other four. It does not estimate pathway activation, ",
    "expression fold change or GSEA NES. Only FDR-significant display-eligible terms select ",
    "representative rows. Clusters without such terms retain their tested column without a dedicated row block."
  )
)
jsonlite::write_json(
  manifest,
  file.path(OUTPUT_DIR, "analysis_manifest.json"),
  pretty = TRUE,
  auto_unbox = TRUE,
  digits = 15,
  na = "null"
)

figure_contract <- c(
  paste0("Evidence scope: Relative functional enrichment across the five molecular-age transcript clusters; databases: ", paste(unname(DATABASE_LABELS[DATABASES]), collapse = ", "), "."),
  "Results-level question: Which nonredundant pathways are preferentially represented in each trajectory cluster relative to the other four?",
  "Figure archetype: Single-panel quantitative bubble matrix.",
  "Backend: R (ggplot2) for analysis-linked drawing, preview and all exports.",
  paste0("Final size: ", FIGURE_WIDTH_MM, " x ", FIGURE_HEIGHT_MM, " mm."),
  "Evidence hierarchy: Filled bubbles show FDR-significant enrichment; open bubbles show non-significant overlaps. Hue identifies the tested cluster (matrix column), not FDR magnitude or the cluster selecting the row. Area encodes gene ratio. Small neutral dots locate zero overlaps and do not encode a nonzero ratio.",
  "Group headings identify the cluster that selected the representative rows; all five clusters are evaluated in every row.",
  "Artwork labels: short cluster IDs only; rounded pale backgrounds contain each displayed cluster heading and its pathway names. C1-C5 matrix columns are retained. No plot title, subtitle, explanatory header or bottom caption. Full interpretation remains in the external figure legend.",
  paste0("Statistics: one-sided Fisher/hypergeometric ORA; independent annotated background and BH correction within each cluster/database; FDR threshold ", FDR_THRESHOLD, "."),
  paste0("Representative selection: up to ", TOP_N_PER_CLUSTER, " FDR-significant terms per cluster; measured-gene Jaccard < ", DISPLAY_JACCARD_CUTOFF, "; no non-significant fallback."),
  paste0("Display-only label exclusion terms: ", paste(display_exclude_terms, collapse = ", "), "; all terms remain in the full inference table."),
  "Source data: figure_source_data.csv (source and database-specific denominator in every row); full inference: all_databases_cluster_vs_rest_ora.csv and separate reactome/go_bp/kegg tables.",
  "Alignment gate: NOT APPLICABLE (one shared ggplot plot area).",
  paste0("Clusters without a dedicated representative row block: ", if (length(omitted_row_clusters)) paste0("C", omitted_row_clusters, collapse = ", ") else "none", "; all remain matrix columns."),
  paste0("Label block and matrix: shared boundary, with ", MATRIX_FIRST_X - LABEL_BLOCK_RIGHT, " mm from the background edge to the first bubble centre at the requested width."),
  "Interpretation boundary: enrichment is relative gene-set representation, not pathway activity or GSEA NES.",
  "Rendered PDF collision and glyph-floor audits are performed after export and recorded separately."
)
writeLines(
  figure_contract,
  file.path(FIGURE_DIR, "figure_contract_and_qa_notes.txt"),
  useBytes = TRUE
)

figure_legend <- paste0(
  "Transcriptomic pathway landscape across molecular-age clusters. ",
  paste(unname(DATABASE_LABELS[DATABASES]), collapse = ", "),
  " term over-representation was tested for each cluster against the other ",
  "four using a one-sided Fisher/hypergeometric test, with Benjamini-Hochberg adjustment ",
  "separately within each cluster and database, including zero-overlap terms. ",
  "Each database used its own annotated clustered-gene background before set-size filtering ",
  "(numbers of background genes / tested terms: ",
  paste0(database_summary$source, ": ", database_summary$annotated_universe_size, " / ",
         database_summary$eligible_tested_terms_per_cluster, collapse = "; "), "). ",
  if (!is.null(entrez_mapping)) paste0(
    paste(unname(DATABASE_LABELS[intersect(DATABASES, c("GO_BP", "KEGG"))]), collapse = " and "),
    " used unique, unambiguously mapped Entrez genes. "
  ) else "",
  if ("Reactome" %in% DATABASES) "Reactome used measured symbols. " else "",
  if ("GO_BP" %in% DATABASES) "GO BP includes ancestor annotations. " else "",
  "Annotation versions and snapshot provenance are recorded in the manifest. ",
  "Up to ", TOP_N_PER_CLUSTER, " FDR-significant representative terms were selected per cluster with ",
  "measured-gene Jaccard similarity < ", format(DISPLAY_JACCARD_CUTOFF, trim = TRUE),
  " across pooled database results; adjustment remains database-specific. ",
  "Pale rounded label blocks identify the cluster used to select each row; all clusters are ",
  "shown for every selected pathway, including pathways selected in more than one group. ",
  "Bubble area represents the fraction of cluster genes annotated in that row's database belonging to the term. ",
  "Filled bubbles indicate significant positive enrichment (FDR < ",
  format(FDR_THRESHOLD, trim = TRUE), "); colour identifies the tested cluster (matrix column), ",
  "using coral red, cyan, mint green, deep blue and soft orange for C1-C5, respectively. ",
  "Colour does not encode FDR magnitude. Open bubbles indicate non-significant overlaps; small neutral dots ",
  "locate zero-overlap cells. ",
  if (length(omitted_row_clusters)) paste0(omitted_row_note, " ") else "",
  "These results describe relative gene-set representation, not pathway activation. ",
  "Source Data identify each row's database, denominators, selection rules and uncapped FDR values."
)
writeLines(figure_legend, file.path(FIGURE_DIR, "figure_legend.txt"), useBytes = TRUE)

message("Transcriptomic cluster pathway landscape analysis complete.")
print(cluster_audit)
print(
  selected_pathways |>
    dplyr::select(
      "source_cluster", "source_cluster_label", "display_rank", "source",
      "term", "overlap_group", "gene_ratio", "fdr", "selection_status"
    )
)
