# =============================================================================
# Transcriptomic pathway molecular-expression heatmap across aging
#
# Independent pipeline: DE-SWAN peak genes -> GO BP / KEGG / Reactome
# enrichment -> at most 15 significant pathways per crest -> age heatmap.
# Full enrichment results are saved for the separate similarity-network script.
#
# Rows are ranked across all three databases by BH-adjusted P, raw P,
# database and term ID. BH correction is performed on all tested terms within
# each crest/database before significance filtering or top-15 selection.
# For every pathway, its crest-associated measured transcripts
# are standardized across all expression samples, averaged with equal weight,
# and smoothed over age. The fitted pathway trajectory is then standardized
# across the common 40-70-year grid for shape-focused heatmap comparison.
# =============================================================================

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/")))
  }
  source_file <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(source_file)) {
    return(dirname(normalizePath(source_file, winslash = "/")))
  }
  normalizePath(getwd(), winslash = "/")
}

setwd(get_script_dir())

extra_r_libraries <- strsplit(
  Sys.getenv("TRANSCRIPTOMICS_R_LIBS", unset = ""), .Platform$path.sep, fixed = TRUE
)[[1]]
.libPaths(c(.libPaths(), extra_r_libraries[nzchar(extra_r_libraries)]))

required_packages <- c(
  "data.table", "dplyr", "ggplot2", "patchwork",
  "ragg", "scales", "svglite", "DEswan", "clusterProfiler",
  "AnnotationDbi", "org.Hs.eg.db"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))
]
if (length(missing_packages) > 0L) {
  stop("Missing required R packages: ", paste(missing_packages, collapse = ", "))
}


# =============================================================================
# 1. Configuration and publication contract
# =============================================================================

DESWAN_INPUT_DIR <- Sys.getenv(
  "TRANSCRIPTOMICS_DESWAN_DIR", unset = file.path("outputs", "deswan-window_7.5")
)
ENRICHMENT_RESULT_DIR <- Sys.getenv(
  "TRANSCRIPTOMICS_ENRICHMENT_DIR",
  unset = Sys.getenv(
    "TRANSCRIPTOMICS_ENRICHMENT_RESULT_DIR",
    unset = file.path("outputs", "transcriptomics_pathway", "kegg_enrichment")
  )
)
EXPRESSION_FILE <- Sys.getenv(
  "TRANSCRIPTOMICS_EXPRESSION_FILE",
  unset = file.path("..", "01-interim_data", "omics_data", "transcriptomics_with_metadata.csv")
)
CREST_FILE <- Sys.getenv(
  "TRANSCRIPTOMICS_CREST_FILE",
  unset = file.path(DESWAN_INPUT_DIR, "transcriptomics_crest_window_7p5.csv")
)
DESWAN_RDS_FILE <- Sys.getenv(
  "TRANSCRIPTOMICS_DESWAN_RDS",
  unset = file.path(DESWAN_INPUT_DIR, "transcriptomics_deswan_breakpoint_window_7p5.rds")
)
KEGG_ANNOTATION_FILE <- Sys.getenv(
  "KEGG_ANNOTATION_FILE", unset = file.path("resources", "KEGG_offline_data_hsa.RData")
)
KEGG_PATHWAY_NAME_FILE <- Sys.getenv(
  "KEGG_PATHWAY_NAME_FILE", unset = file.path("resources", "kegg_hsa_pathway_names.tsv")
)
REACTOME_GMT_FILE <- Sys.getenv(
  "REACTOME_GMT_FILE", unset = file.path("resources", "reactome_gmt", "ReactomePathways.gmt")
)
OUTPUT_DIR <- Sys.getenv(
  "TRANSCRIPTOMICS_PATHWAY_HEATMAP_OUTPUT_DIR",
  unset = file.path("outputs", "transcriptomics_pathway", "pathway_expression_heatmap")
)
FIGURE_DIR <- Sys.getenv(
  "TRANSCRIPTOMICS_PATHWAY_HEATMAP_FIGURE_DIR",
  unset = file.path("figures", "transcriptomics", "pathway_expression_heatmap")
)

AGE_MIN <- as.numeric(Sys.getenv("PATHWAY_HEATMAP_AGE_MIN", unset = "40"))
AGE_MAX <- as.numeric(Sys.getenv("PATHWAY_HEATMAP_AGE_MAX", unset = "70"))
FDR_METHOD <- "BH"
GENE_FDR_THRESHOLD <- 0.05
FDR_THRESHOLD <- 0.05
MIN_GENE_SET_SIZE <- 10L
MAX_GENE_SET_SIZE <- 500L
TOP_PATHWAYS_PER_CREST <- 15L
PATHWAY_RANKING_RULE <- "BH-adjusted P ascending; raw P ascending; database ascending; term ID ascending"
AGE_GRID_STEP <- as.numeric(Sys.getenv("PATHWAY_HEATMAP_AGE_STEP", unset = "0.25"))
LOESS_SPAN <- as.numeric(Sys.getenv("PATHWAY_HEATMAP_LOESS_SPAN", unset = "0.34"))
MIN_FINITE_GENE_SAMPLES <- as.integer(Sys.getenv(
  "PATHWAY_HEATMAP_MIN_GENE_SAMPLES", unset = "10"
))
MIN_TRANSCRIPTS_PER_PATHWAY <- as.integer(Sys.getenv(
  "PATHWAY_HEATMAP_MIN_PATHWAY_TRANSCRIPTS", unset = "3"
))
FIGURE_WIDTH_MM <- as.numeric(Sys.getenv("PATHWAY_HEATMAP_WIDTH_MM", unset = "100"))
FIGURE_HEIGHT_MM <- as.numeric(Sys.getenv("PATHWAY_HEATMAP_HEIGHT_MM", unset = "120"))
FIGURE_PREVIEW_DPI <- as.integer(Sys.getenv("PATHWAY_HEATMAP_PREVIEW_DPI", unset = "450"))
FIGURE_TIFF_DPI <- as.integer(Sys.getenv("PATHWAY_HEATMAP_TIFF_DPI", unset = "600"))

FIGURE_STEM <- file.path(FIGURE_DIR, "two_crest_pathway_molecular_expression_heatmap")
AGE_GRID <- seq(AGE_MIN, AGE_MAX, by = AGE_GRID_STEP)

FIGURE_CONTRACT <- c(
  "Core conclusion: The most significant enriched terms at each transcriptomic crest can be compared through the age trajectories of their measured contributing genes.",
  "Results-level question: How does the aggregate expression of crest-associated transcripts in each saved pathway change across age, especially around its assigned crest?",
  "Figure archetype: Quantitative grid with two vertically stacked, directly comparable panels.",
  "Panel sequence: One age-ordered panel per saved crest, showing up to 15 significant enriched terms.",
  "Evidence hierarchy: Database-specific enrichment FDR and rank determine selection; measured-gene trajectories describe age patterns; saved crest ages provide reference lines.",
  paste0("Display scope: At most ", TOP_PATHWAYS_PER_CREST, " significant terms per crest across all three databases; all retained expression observations are used."),
  "Expression encoding: Mean of equally weighted member-transcript sample Z scores, LOESS-smoothed over age, then Z-standardized across the common age grid per pathway.",
  "Interpretation boundary: The heatmap is descriptive; it does not show uncertainty or a causal age effect.",
  paste0("Target/output: ", FIGURE_WIDTH_MM, " mm x ", FIGURE_HEIGHT_MM, " mm artwork with editable SVG/PDF and 600-dpi TIFF; no added outer margins."),
  "Display labels: Pathway names only; centered blue Crest1 and orange Crest2 titles. Counts are reported in the separate summary document."
)


# =============================================================================
# 2. Validation and utility helpers
# =============================================================================

assert_file <- function(path) {
  if (!file.exists(path)) stop("Required input does not exist: ", path)
  normalizePath(path, winslash = "/")
}

assert_columns <- function(data, required, object_name) {
  missing <- setdiff(required, names(data))
  if (length(missing) > 0L) {
    stop(object_name, " is missing columns: ", paste(missing, collapse = ", "))
  }
  invisible(data)
}

assert_positive_scalar <- function(value, name) {
  if (length(value) != 1L || !is.finite(value) || value <= 0) {
    stop(name, " must be one positive finite value")
  }
  invisible(value)
}

for (item in list(
  AGE_GRID_STEP = AGE_GRID_STEP,
  LOESS_SPAN = LOESS_SPAN,
  MIN_FINITE_GENE_SAMPLES = MIN_FINITE_GENE_SAMPLES,
  MIN_TRANSCRIPTS_PER_PATHWAY = MIN_TRANSCRIPTS_PER_PATHWAY,
  FIGURE_WIDTH_MM = FIGURE_WIDTH_MM,
  FIGURE_HEIGHT_MM = FIGURE_HEIGHT_MM,
  FIGURE_PREVIEW_DPI = FIGURE_PREVIEW_DPI,
  FIGURE_TIFF_DPI = FIGURE_TIFF_DPI
)) {
  assert_positive_scalar(item, deparse(substitute(item)))
}
if (!is.finite(AGE_MIN) || !is.finite(AGE_MAX) || AGE_MIN >= AGE_MAX) {
  stop("Age limits must be finite and AGE_MIN must be below AGE_MAX")
}
if (LOESS_SPAN <= 0 || LOESS_SPAN > 1) {
  stop("PATHWAY_HEATMAP_LOESS_SPAN must lie in (0, 1]")
}
if (FIGURE_HEIGHT_MM > 170) {
  stop("Nature main-figure height must not exceed 170 mm")
}

split_transcripts <- function(value) {
  if (length(value) != 1L || is.na(value) || !nzchar(trimws(value))) {
    return(character())
  }
  values <- trimws(strsplit(gsub("/", ";", value, fixed = TRUE), ";", fixed = TRUE)[[1]])
  unique(values[nzchar(values)])
}

sentence_case <- function(value) {
  value <- trimws(as.character(value))
  has_text <- !is.na(value) & nzchar(value) &
    !grepl("^(mRNA|tRNA|rRNA|ncRNA|miRNA|lncRNA|cAMP|cGMP)\\b", value)
  value[has_text] <- paste0(
    toupper(substr(value[has_text], 1L, 1L)),
    substring(value[has_text], 2L)
  )
  value
}

safe_number <- function(value, digits = 2L) {
  format(round(value, digits), nsmall = digits, trim = TRUE, scientific = FALSE)
}

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)


# =============================================================================
# 3. Peak gene extraction, complete enrichment and independent top-15 ranking
# =============================================================================

format_age <- function(x) {
  ifelse(abs(x - round(x)) < 1e-8, as.character(round(x)), format(x, digits = 3))
}

ratio_to_numeric <- function(x) {
  vapply(strsplit(as.character(x), "/", fixed = TRUE), function(parts) {
    if (length(parts) != 2) return(NA_real_)
    numerator <- suppressWarnings(as.numeric(parts[1]))
    denominator <- suppressWarnings(as.numeric(parts[2]))
    if (is.na(numerator) || is.na(denominator) || denominator == 0) {
      return(NA_real_)
    }
    numerator / denominator
  }, numeric(1))
}

infer_id_type <- function(ids) {
  ids <- unique(stats::na.omit(as.character(ids)))
  if (length(ids) > 0 && mean(grepl("^ENSG[0-9]+(\\.[0-9]+)?$", ids)) >= 0.8) {
    return("ENSEMBL")
  }
  "SYMBOL"
}

map_to_entrez <- function(ids, id_type) {
  ids <- unique(stats::na.omit(as.character(ids)))
  if (id_type == "ENSEMBL") {
    ids <- sub("\\.[0-9]+$", "", ids)
  }

  mapped <- suppressMessages(
    clusterProfiler::bitr(
      ids,
      fromType = id_type,
      toType = "ENTREZID",
      OrgDb = org.Hs.eg.db::org.Hs.eg.db,
      drop = TRUE
    )
  )
  if (is.null(mapped) || nrow(mapped) == 0) {
    return(data.frame(feature = character(), ENTREZID = character()))
  }
  names(mapped)[names(mapped) == id_type] <- "feature"
  mapped |>
    dplyr::transmute(
      feature = as.character(.data$feature),
      ENTREZID = as.character(.data$ENTREZID)
    ) |>
    dplyr::distinct()
}

get_deswan_q_table <- function(rds_file) {
  deswan_result <- readRDS(rds_file)
  p_wide <- DEswan::reshape.DEswan(
    deswan_result,
    parameter = 1,
    factor = "qt"
  )
  q_wide <- DEswan::q.DEswan(p_wide, method = FDR_METHOD)
  if (!"variable" %in% names(q_wide)) {
    stop("The DE-SWAN RDS could not be converted to a window-level q-value table.")
  }
  q_wide
}

load_local_kegg <- function(path) {
  annotation_env <- new.env(parent = emptyenv())
  load(path, envir = annotation_env)
  object_names <- ls(annotation_env)

  if (!all(c("term2gene", "term2name") %in% object_names)) {
    stop(
      "KEGG annotation file must contain objects named term2gene and term2name: ",
      path
    )
  }

  term2gene <- get("term2gene", envir = annotation_env)
  term2name <- get("term2name", envir = annotation_env)
  term2gene <- as.data.frame(term2gene, stringsAsFactors = FALSE)[, 1:2, drop = FALSE]
  term2name <- as.data.frame(term2name, stringsAsFactors = FALSE)[, 1:2, drop = FALSE]
  names(term2gene) <- c("term", "gene")
  names(term2name) <- c("term", "name")
  term2gene[] <- lapply(term2gene, as.character)
  term2name[] <- lapply(term2name, as.character)

  list(
    term2gene = dplyr::distinct(term2gene),
    term2name = dplyr::distinct(term2name),
    mode = paste0("frozen TERM2GENE: ", normalizePath(path, winslash = "/"))
  )
}

build_orgdb_kegg <- function(pathway_name_file = NULL) {
  entrez_keys <- AnnotationDbi::keys(org.Hs.eg.db::org.Hs.eg.db, keytype = "ENTREZID")
  mapping <- suppressMessages(
    AnnotationDbi::select(
      org.Hs.eg.db::org.Hs.eg.db,
      keys = entrez_keys,
      columns = "PATH",
      keytype = "ENTREZID"
    )
  )
  term2gene <- mapping |>
    dplyr::filter(!is.na(.data$PATH), !is.na(.data$ENTREZID)) |>
    dplyr::transmute(
      term = paste0("hsa", .data$PATH),
      gene = as.character(.data$ENTREZID)
    ) |>
    dplyr::distinct()

  if (!is.null(pathway_name_file) && file.exists(pathway_name_file)) {
    term2name <- data.table::fread(
      pathway_name_file,
      header = FALSE,
      sep = "\t",
      data.table = FALSE,
      col.names = c("term", "name")
    ) |>
      dplyr::mutate(
        term = sub("^path:", "", as.character(.data$term)),
        name = sub(" - Homo sapiens \\(human\\)$", "", as.character(.data$name))
      ) |>
      dplyr::distinct()
    name_source <- normalizePath(pathway_name_file, winslash = "/")
  } else {
    warning(
      "KEGG pathway-name file not found; pathway IDs will be used as labels: ",
      pathway_name_file
    )
    term2name <- data.frame(
      term = unique(term2gene$term),
      name = unique(term2gene$term),
      stringsAsFactors = FALSE
    )
    name_source <- "pathway IDs only"
  }

  list(
    term2gene = term2gene,
    term2name = term2name,
    mode = paste0(
      "offline org.Hs.eg.db PATH mapping; pathway names: ", name_source
    )
  )
}

run_kegg_ora <- function(genes, universe, local_kegg) {
  genes <- intersect(unique(as.character(genes)), unique(as.character(universe)))
  universe <- unique(as.character(universe))

  if (length(genes) == 0) return(NULL)

  clusterProfiler::enricher(
    gene = genes,
    universe = universe,
    TERM2GENE = local_kegg$term2gene,
    TERM2NAME = local_kegg$term2name,
    pvalueCutoff = 1,
    qvalueCutoff = 1,
    pAdjustMethod = "BH",
    minGSSize = MIN_GENE_SET_SIZE,
    maxGSSize = MAX_GENE_SET_SIZE
  )
}

read_reactome_gmt <- function(path) {
  if (!file.exists(path)) {
    stop(
      "Reactome GMT file not found: ", path,
      ". Download ReactomePathways.gmt.zip from the official Reactome release."
    )
  }

  records <- strsplit(readLines(path, warn = FALSE, encoding = "UTF-8"), "\t", fixed = TRUE)
  records <- records[lengths(records) >= 3]
  term2name <- data.frame(
    term = vapply(records, function(record) record[[2]], character(1)),
    name = vapply(records, function(record) record[[1]], character(1)),
    stringsAsFactors = FALSE
  ) |>
    dplyr::distinct()
  term2gene <- dplyr::bind_rows(lapply(records, function(record) {
    data.frame(
      term = record[2],
      gene = record[-c(1, 2)],
      stringsAsFactors = FALSE
    )
  })) |>
    dplyr::filter(nzchar(.data$gene)) |>
    dplyr::distinct()

  list(term2gene = term2gene, term2name = term2name)
}

empty_enrichment_table <- function() {
  data.table::data.table(
    ID = character(), Description = character(), GeneRatio = character(),
    BgRatio = character(), pvalue = numeric(), p.adjust = numeric(),
    qvalue = numeric(), geneID = character(), Count = integer(),
    database = character(), crest_id = integer(), crest_label = character(),
    peak_age = numeric(), n_crest_genes = integer(),
    gene_ratio_numeric = numeric(), background_ratio_numeric = numeric(),
    fold_enrichment = numeric(), minus_log10_fdr = numeric(),
    measured_transcripts = character()
  )
}

annotate_enrichment <- function(result, crest, database_name, feature_map, n_genes) {
  if (is.null(result) || nrow(result@result) == 0L) return(empty_enrichment_table())
  # Read the complete tested table before any enrichment significance filter.
  table <- data.table::as.data.table(result@result)
  table[, p.adjust := stats::p.adjust(pvalue, method = "BH")]
  table[, `:=`(
    database = database_name, crest_id = as.integer(crest$crest_id),
    crest_label = crest$crest_label, peak_age = as.numeric(crest$peak_age),
    n_crest_genes = as.integer(n_genes),
    gene_ratio_numeric = ratio_to_numeric(GeneRatio),
    background_ratio_numeric = ratio_to_numeric(BgRatio)
  )]
  table[, `:=`(
    fold_enrichment = gene_ratio_numeric / background_ratio_numeric,
    minus_log10_fdr = -log10(pmax(p.adjust, .Machine$double.xmin))
  )]
  # Restrict reverse mapping to this crest's actual significant transcripts.
  # This also preserves original versioned Ensembl IDs used in expression data.
  map <- feature_map[feature_map$crest_id == crest$crest_id, ]
  table[, measured_transcripts := vapply(strsplit(geneID, "/", fixed = TRUE), function(ids) {
    paste(sort(unique(map$feature[map$analysis_id %in% ids])), collapse = ";")
  }, character(1))]
  table
}

select_top_pathways <- function(enrichment, top_n = TOP_PATHWAYS_PER_CREST) {
  if (length(top_n) != 1L || !is.finite(top_n) || top_n < 1L || top_n %% 1 != 0) {
    stop("top_n must be a positive integer")
  }
  table <- data.table::copy(data.table::as.data.table(enrichment))
  assert_columns(table, c("crest_id", "ID", "database", "p.adjust", "pvalue"), "enrichment table")
  if (anyDuplicated(table[, .(crest_id, database, ID)])) {
    stop("Enrichment keys must be unique before pathway ranking")
  }
  table <- table[is.finite(p.adjust) & p.adjust < FDR_THRESHOLD]
  data.table::setorderv(table, c("crest_id", "p.adjust", "pvalue", "database", "ID"))
  table[, pathway_rank := seq_len(.N), by = crest_id]
  table[pathway_rank <= top_n]
}

run_crest_enrichment <- function(crests, rds_path) {
  message("Extracting significant transcripts at each saved DE-SWAN crest ...")
  q_wide <- get_deswan_q_table(rds_path)
  tested_transcripts <- as.character(q_wide$variable)
  if (anyNA(tested_transcripts) || any(!nzchar(tested_transcripts)) || anyDuplicated(tested_transcripts)) {
    stop("DE-SWAN transcript identifiers must be nonempty and unique")
  }
  window_ages <- suppressWarnings(as.numeric(sub("^X", "", names(q_wide)[-1])))
  if (anyNA(window_ages) || anyDuplicated(window_ages)) stop("DE-SWAN window ages must be unique numeric values")
  peak_genes <- data.table::rbindlist(lapply(seq_len(nrow(crests)), function(i) {
    crest <- crests[i]
    index <- match(crest$peak_age, window_ages)
    if (is.na(index)) stop("No DE-SWAN window matches peak age ", crest$peak_age)
    q <- as.numeric(q_wide[[index + 1L]])
    keep <- is.finite(q) & q < GENE_FDR_THRESHOLD
    data.table::data.table(
      crest_id = rep(crest$crest_id, sum(keep)),
      crest_label = rep(crest$crest_label, sum(keep)),
      peak_age = rep(crest$peak_age, sum(keep)),
      peak_n_reported = rep(crest$peak_n, sum(keep)),
      feature = tested_transcripts[keep], q_value = q[keep]
    )
  }))
  data.table::setorder(peak_genes, crest_id, q_value, feature)
  validation <- merge(
    crests[, .(crest_id, crest_label, peak_age, peak_n)],
    peak_genes[, .(n_recovered_at_peak = data.table::uniqueN(feature)), by = crest_id],
    by = "crest_id", all.x = TRUE
  )
  validation[is.na(n_recovered_at_peak), n_recovered_at_peak := 0L]
  validation[, count_difference := n_recovered_at_peak - peak_n]
  data.table::fwrite(validation, file.path(ENRICHMENT_RESULT_DIR, "crest_peak_count_validation.csv"))
  if (any(validation$count_difference != 0)) {
    stop("Peak gene counts disagree with the crest table. Check the selected RDS, crest file and q threshold.")
  }
  data.table::fwrite(peak_genes, file.path(ENRICHMENT_RESULT_DIR, "crest_associated_transcripts.csv"))
  data.table::fwrite(crests, file.path(ENRICHMENT_RESULT_DIR, "input_crests.csv"))

  id_type <- infer_id_type(tested_transcripts)
  mapped <- map_to_entrez(tested_transcripts, id_type)
  names(mapped)[names(mapped) == "feature"] <- "mapping_key"
  original_ids <- data.frame(
    feature = tested_transcripts,
    mapping_key = if (id_type == "ENSEMBL") sub("\\.[0-9]+$", "", tested_transcripts) else tested_transcripts
  )
  universe_map <- merge(original_ids, mapped, by = "mapping_key")
  universe_map <- unique(universe_map[, c("feature", "ENTREZID")])
  universe_entrez <- sort(unique(universe_map$ENTREZID))
  if (length(universe_entrez) == 0L) stop("No DE-SWAN-tested transcripts map to Entrez IDs")
  crest_entrez <- merge(as.data.frame(peak_genes), universe_map, by = "feature")
  entrez_features <- unique(data.frame(
    crest_id = crest_entrez$crest_id, feature = crest_entrez$feature,
    analysis_id = crest_entrez$ENTREZID
  ))
  data.table::fwrite(universe_map, file.path(ENRICHMENT_RESULT_DIR, "transcript_to_entrez.csv"))
  data.table::fwrite(data.frame(
    id_type = id_type, n_deswan_tested_transcripts = length(tested_transcripts),
    n_mapped_transcripts = length(unique(universe_map$feature)),
    n_mapped_entrez = length(universe_entrez),
    mapping_fraction = length(unique(universe_map$feature)) / length(tested_transcripts)
  ), file.path(ENRICHMENT_RESULT_DIR, "universe_mapping_coverage.csv"))
  coverage <- data.table::rbindlist(lapply(seq_len(nrow(crests)), function(i) {
    crest <- crests[i]
    map <- crest_entrez[crest_entrez$crest_id == crest$crest_id, ]
    n_input <- peak_genes[crest_id == crest$crest_id, data.table::uniqueN(feature)]
    data.table::data.table(
      crest_id = crest$crest_id, crest_label = crest$crest_label,
      n_input_transcripts = n_input, n_mapped_transcripts = length(unique(map$feature)),
      n_mapped_entrez = length(unique(map$ENTREZID)),
      mapping_fraction = if (n_input > 0L) length(unique(map$feature)) / n_input else NA_real_
    )
  }))
  data.table::fwrite(coverage, file.path(ENRICHMENT_RESULT_DIR, "crest_mapping_coverage.csv"))

  local_kegg <- if (file.exists(KEGG_ANNOTATION_FILE)) {
    load_local_kegg(KEGG_ANNOTATION_FILE)
  } else {
    build_orgdb_kegg(KEGG_PATHWAY_NAME_FILE)
  }
  reactome_sets <- read_reactome_gmt(REACTOME_GMT_FILE)
  entrez_symbols <- suppressMessages(AnnotationDbi::select(
    org.Hs.eg.db::org.Hs.eg.db, keys = universe_entrez,
    keytype = "ENTREZID", columns = "SYMBOL"
  ))
  entrez_symbols <- unique(entrez_symbols[!is.na(entrez_symbols$SYMBOL), c("ENTREZID", "SYMBOL")])
  reactome_universe <- sort(unique(entrez_symbols$SYMBOL))
  crest_symbols <- merge(crest_entrez, entrez_symbols, by = "ENTREZID")
  symbol_features <- unique(data.frame(
    crest_id = crest_symbols$crest_id, feature = crest_symbols$feature,
    analysis_id = crest_symbols$SYMBOL
  ))
  results <- list()
  test_summary <- list()
  for (i in seq_len(nrow(crests))) {
    crest <- crests[i]
    entrez <- sort(unique(entrez_features$analysis_id[entrez_features$crest_id == crest$crest_id]))
    symbols <- sort(unique(symbol_features$analysis_id[symbol_features$crest_id == crest$crest_id]))
    message("Enrichment for ", crest$crest_label, ": ", length(entrez), " mapped Entrez genes")
    go <- if (length(entrez) == 0L) NULL else clusterProfiler::enrichGO(
      gene = entrez, universe = universe_entrez, OrgDb = org.Hs.eg.db::org.Hs.eg.db,
      keyType = "ENTREZID", ont = "BP", pvalueCutoff = 1, qvalueCutoff = 1,
      pAdjustMethod = "BH", minGSSize = MIN_GENE_SET_SIZE,
      maxGSSize = MAX_GENE_SET_SIZE, readable = FALSE
    )
    kegg <- run_kegg_ora(entrez, universe_entrez, local_kegg)
    reactome <- if (length(symbols) == 0L) NULL else clusterProfiler::enricher(
      gene = symbols, universe = reactome_universe,
      TERM2GENE = reactome_sets$term2gene, TERM2NAME = reactome_sets$term2name,
      pvalueCutoff = 1, qvalueCutoff = 1, pAdjustMethod = "BH",
      minGSSize = MIN_GENE_SET_SIZE, maxGSSize = MAX_GENE_SET_SIZE
    )
    tables <- list(
      `GO Biological Process` = annotate_enrichment(go, crest, "GO Biological Process", entrez_features, length(entrez)),
      KEGG = annotate_enrichment(kegg, crest, "KEGG", entrez_features, length(entrez)),
      Reactome = annotate_enrichment(reactome, crest, "Reactome", symbol_features, length(symbols))
    )
    results <- c(results, unname(tables))
    for (db in names(tables)) {
      table <- tables[[db]]
      test_summary[[length(test_summary) + 1L]] <- data.table::data.table(
        crest_id = crest$crest_id, crest_label = crest$crest_label, database = db,
        n_tested = nrow(table), n_significant = sum(is.finite(table$p.adjust) & table$p.adjust < FDR_THRESHOLD)
      )
    }
  }
  full <- data.table::rbindlist(results, use.names = TRUE, fill = TRUE)
  data.table::setorderv(full, c("crest_id", "p.adjust", "pvalue", "database", "ID"))
  file_keys <- list(
    `GO Biological Process` = c("go_bp_enrichment_all_terms.csv", "go_bp_enrichment_fdr_0.05.csv"),
    KEGG = c("kegg_enrichment_all_pathways.csv", "kegg_enrichment_fdr_0.05.csv"),
    Reactome = c("reactome_enrichment_all_pathways.csv", "reactome_enrichment_fdr_0.05.csv")
  )
  for (db in names(file_keys)) {
    table <- full[database == db]
    data.table::fwrite(table, file.path(ENRICHMENT_RESULT_DIR, file_keys[[db]][1]))
    data.table::fwrite(table[is.finite(p.adjust) & p.adjust < FDR_THRESHOLD], file.path(ENRICHMENT_RESULT_DIR, file_keys[[db]][2]))
  }
  data.table::fwrite(full, file.path(ENRICHMENT_RESULT_DIR, "all_pathway_enrichment_results.csv"))
  data.table::fwrite(data.table::rbindlist(test_summary), file.path(ENRICHMENT_RESULT_DIR, "significant_term_counts_by_crest.csv"))
  writeLines(c(
    paste0("analysis_date=", Sys.Date()),
    paste0("crest_file=", normalizePath(CREST_FILE, winslash = "/")),
    paste0("crest_md5=", unname(tools::md5sum(CREST_FILE))),
    paste0("deswan_rds_file=", normalizePath(rds_path, winslash = "/")),
    paste0("deswan_rds_md5=", unname(tools::md5sum(rds_path))),
    "gene_selection=BH q < 0.05 at each saved peak_age; all listed crests included",
    "background=all DE-SWAN-tested transcripts mapped to the identifiers used by each annotation database",
    "gene_set_size=10..500 after background intersection",
    "all_results=all tested ORA rows including nonsignificant rows; tests require an overlapping input gene",
    "fdr_method=BH within each crest and database, before significance filtering and ranking",
    "fdr_threshold=0.05; database-specific FDR is not a pooled cross-database FDR",
    paste0("heatmap_ranking=", PATHWAY_RANKING_RULE),
    "heatmap_scope=at most 15 significant terms per crest across all databases; no module selection or semantic pruning",
    paste0("kegg_annotation=", local_kegg$mode),
    paste0("reactome_gmt=", normalizePath(REACTOME_GMT_FILE, winslash = "/")),
    paste0("reactome_gmt_md5=", unname(tools::md5sum(REACTOME_GMT_FILE))),
    paste0("R=", R.version.string),
    paste0("clusterProfiler=", utils::packageVersion("clusterProfiler")),
    paste0("org.Hs.eg.db=", utils::packageVersion("org.Hs.eg.db"))
  ), file.path(ENRICHMENT_RESULT_DIR, "analysis_manifest.txt"))
  full
}

dir.create(ENRICHMENT_RESULT_DIR, recursive = TRUE, showWarnings = FALSE)
crest_path <- assert_file(CREST_FILE)
rds_path <- assert_file(DESWAN_RDS_FILE)
expression_path <- assert_file(EXPRESSION_FILE)
crests <- data.table::fread(crest_path)
assert_columns(crests, c("crest_id", "peak_age", "peak_n"), "DE-SWAN crest table")
if (nrow(crests) == 0L || anyDuplicated(crests$crest_id) ||
    any(!is.finite(crests$peak_age)) || any(!is.finite(crests$crest_id)) ||
    any(crests$crest_id %% 1 != 0) || any(!is.finite(crests$peak_n)) ||
    any(crests$peak_n < 0 | crests$peak_n %% 1 != 0)) {
  stop("Crest table requires unique integer IDs, finite ages and nonnegative integer peak counts")
}
data.table::setorder(crests, peak_age, crest_id)
crests[, crest_label := paste0("Age ", format_age(peak_age), " crest")]
crests[, crest_display_label := paste0("Crest", seq_len(.N))]
crests[, crest_colour := rep(c("#0072B2", "#E69F00"), length.out = .N)]
enrichment <- run_crest_enrichment(crests, rds_path)
pathways <- select_top_pathways(enrichment)
selected_pathway_path <- file.path(ENRICHMENT_RESULT_DIR, "top15_significant_pathways_per_crest.csv")
data.table::fwrite(pathways, selected_pathway_path)
pathways[, `:=`(
  term_id = as.character(ID), description = as.character(Description),
  p_adjust = as.numeric(p.adjust), term_uid = paste(database, ID, sep = "::")
)]
pathways[, measured_transcript_list := lapply(measured_transcripts, split_transcripts)]
database_abbreviation <- c(`GO Biological Process` = "GO BP", KEGG = "KEGG", Reactome = "Reactome")
pathways[, pathway_y := .N - pathway_rank + 1L, by = crest_id]
pathways[, database_short := unname(database_abbreviation[database])]
pathways[, pathway_label := vapply(
  sentence_case(description),
  function(label) paste(strwrap(label, width = 54L), collapse = "\n"), character(1)
)]
pathways[, pathway_uid := paste(crest_id, database, term_id, sep = "::")]
pathways_per_crest <- merge(
  crests[, .(crest_id, crest_label, peak_age)], pathways[, .N, by = crest_id],
  by = "crest_id", all.x = TRUE
)
pathways_per_crest[is.na(N), N := 0L]
MAX_DISPLAY_ROWS <- max(c(1L, pathways_per_crest$N))
pathways[, pathway_y := MAX_DISPLAY_ROWS - pathway_rank + 1L]
data.table::fwrite(pathways_per_crest, file.path(OUTPUT_DIR, "selected_pathway_counts_by_crest.csv"))
message("Selected pathways per crest (at most 15 across all databases):")
print(pathways_per_crest)


# =============================================================================
# 4. Load all required expression observations and standardize transcripts
# =============================================================================

requested_transcripts <- sort(unique(as.character(unlist(
  pathways$measured_transcript_list, use.names = FALSE
))))
expression_header <- names(data.table::fread(
  expression_path, nrows = 0L, data.table = TRUE, check.names = FALSE
))
assert_columns(
  data.frame(matrix(nrow = 0L, ncol = length(expression_header),
                    dimnames = list(NULL, expression_header))),
  c("participant_id", "age"),
  "expression table"
)

present_transcripts <- requested_transcripts[
  requested_transcripts %in% expression_header
]
missing_transcripts <- setdiff(requested_transcripts, present_transcripts)
message(
  "Reading ", length(present_transcripts), " requested transcript columns from ",
  basename(expression_path), " ..."
)
expression <- data.table::fread(
  expression_path,
  select = c("participant_id", "age", present_transcripts),
  data.table = TRUE,
  check.names = FALSE,
  showProgress = interactive()
)
expression[, age := as.numeric(age)]
samples_before_age_filter <- nrow(expression)
finite_age <- is.finite(expression$age)
in_age_range <- finite_age & expression$age >= AGE_MIN & expression$age <= AGE_MAX
expression <- expression[in_age_range]
samples_after_age_filter <- nrow(expression)
if (samples_after_age_filter < 10L) {
  stop("Too few samples remain in the requested age range")
}

expression_matrix <- as.matrix(expression[, ..present_transcripts])
storage.mode(expression_matrix) <- "double"
expression_matrix[!is.finite(expression_matrix)] <- NA_real_

finite_counts <- colSums(is.finite(expression_matrix))
gene_means <- colMeans(expression_matrix, na.rm = TRUE)
gene_sds <- if (ncol(expression_matrix) > 0L) apply(expression_matrix, 2L, stats::sd, na.rm = TRUE) else numeric()
present_gene_qc <- data.table::data.table(
  transcript = present_transcripts,
  present_in_expression = TRUE,
  n_finite_samples = as.integer(finite_counts),
  expression_mean = as.numeric(gene_means),
  expression_sd = as.numeric(gene_sds)
)
present_gene_qc[, included := (
  n_finite_samples >= MIN_FINITE_GENE_SAMPLES &
    is.finite(expression_sd) & expression_sd > 1e-10
)]
present_gene_qc[, exclusion_reason := data.table::fcase(
  included, "",
  n_finite_samples < MIN_FINITE_GENE_SAMPLES, "too_few_finite_samples",
  !is.finite(expression_sd), "non_finite_standard_deviation",
  expression_sd <= 1e-10, "constant_or_near_constant_expression",
  default = "unspecified"
)]
missing_gene_qc <- data.table::data.table(
  transcript = missing_transcripts,
  present_in_expression = FALSE,
  n_finite_samples = 0L,
  expression_mean = NA_real_,
  expression_sd = NA_real_,
  included = FALSE,
  exclusion_reason = "absent_from_expression_header"
)
gene_qc <- data.table::rbindlist(
  list(present_gene_qc, missing_gene_qc),
  use.names = TRUE,
  fill = TRUE
)
data.table::setorder(gene_qc, transcript)
included_transcripts <- gene_qc[included == TRUE, transcript]
if (length(included_transcripts) == 0L && nrow(pathways) > 0L) {
  warning("No selected transcripts pass expression QC; selected pathways will be retained as unavailable (grey).")
}

included_index <- match(included_transcripts, colnames(expression_matrix))
standardized_expression <- sweep(
  expression_matrix[, included_index, drop = FALSE],
  2L,
  gene_means[included_index],
  FUN = "-"
)
standardized_expression <- sweep(
  standardized_expression,
  2L,
  gene_sds[included_index],
  FUN = "/"
)
colnames(standardized_expression) <- included_transcripts


# =============================================================================
# 5. Build pathway-level molecular trajectories
# =============================================================================

fit_pathway_trajectory <- function(pathway_index) {
  pathway <- pathways[pathway_index]
  members <- pathway$measured_transcript_list[[1]]
  retained <- intersect(members, included_transcripts)
  fitted_score <- trajectory_z <- rep(NA_real_, length(AGE_GRID))
  n_samples <- 0L
  median_observed <- NA_real_
  status <- "insufficient_qc_transcripts"
  if (length(retained) >= MIN_TRANSCRIPTS_PER_PATHWAY) {
    pathway_matrix <- standardized_expression[, retained, drop = FALSE]
    observed <- rowSums(is.finite(pathway_matrix))
    pathway_score <- rowMeans(pathway_matrix, na.rm = TRUE)
    pathway_score[observed == 0L] <- NA_real_
    fit_data <- data.frame(age = expression$age, pathway_score = pathway_score)
    fit_data <- fit_data[is.finite(fit_data$age) & is.finite(fit_data$pathway_score), , drop = FALSE]
    n_samples <- nrow(fit_data)
    median_observed <- stats::median(observed)
    status <- "insufficient_finite_samples"
    if (n_samples >= 10L && length(unique(fit_data$age)) >= 4L) {
      prediction <- tryCatch({
        model <- stats::loess(
          pathway_score ~ age, data = fit_data, span = LOESS_SPAN,
          degree = 1L, family = "gaussian", na.action = stats::na.exclude,
          control = stats::loess.control(surface = "direct", statistics = "none", trace.hat = "approximate")
        )
        as.numeric(stats::predict(model, newdata = data.frame(age = AGE_GRID)))
      }, error = function(e) rep(NA_real_, length(AGE_GRID)))
      status <- "non_finite_fit"
      if (all(is.finite(prediction))) {
        fitted_score <- prediction
        fitted_sd <- stats::sd(fitted_score)
        status <- "constant_fitted_trajectory"
        if (is.finite(fitted_sd) && fitted_sd > 1e-10) {
          trajectory_z <- (fitted_score - mean(fitted_score)) / fitted_sd
          status <- "ok"
        }
      }
    }
  }
  pre <- AGE_GRID >= pathway$peak_age - 2 & AGE_GRID <= pathway$peak_age - 0.5
  post <- AGE_GRID >= pathway$peak_age + 0.5 & AGE_GRID <= pathway$peak_age + 2
  list(
    trajectory = data.table::data.table(
      pathway_uid = pathway$pathway_uid, crest_id = pathway$crest_id,
      crest_label = pathway$crest_label, peak_age = pathway$peak_age,
      pathway_rank = pathway$pathway_rank, pathway_y = pathway$pathway_y,
      database = pathway$database, term_id = pathway$term_id,
      description = pathway$description, pathway_label = pathway$pathway_label,
      age = AGE_GRID, smoothed_mean_transcript_z = fitted_score,
      pathway_trajectory_z = trajectory_z, trajectory_status = status
    ),
    summary = data.table::data.table(
      pathway_uid = pathway$pathway_uid, n_measured_transcripts = length(members),
      n_present_transcripts = sum(members %in% present_transcripts),
      n_qc_transcripts = length(retained),
      transcript_coverage_fraction = if (length(members)) length(retained) / length(members) else NA_real_,
      n_finite_pathway_samples = n_samples, median_observed_transcripts_per_sample = median_observed,
      smoothed_score_amplitude = if (all(is.finite(fitted_score))) diff(range(fitted_score)) else NA_real_,
      crest_delta_z = if (any(pre) && any(post)) mean(trajectory_z[post]) - mean(trajectory_z[pre]) else NA_real_,
      local_one_year_change_z = trajectory_z[which.min(abs(AGE_GRID - (pathway$peak_age + 0.5)))] -
        trajectory_z[which.min(abs(AGE_GRID - (pathway$peak_age - 0.5)))],
      qc_transcripts = paste(retained, collapse = ";"), trajectory_status = status
    )
  )
}


message("Fitting ", nrow(pathways), " pathway trajectories ...")
trajectory_results <- lapply(seq_len(nrow(pathways)), fit_pathway_trajectory)
trajectory_data <- data.table::rbindlist(lapply(
  trajectory_results, function(result) result$trajectory
))
pathway_summary <- data.table::rbindlist(lapply(
  trajectory_results, function(result) result$summary
))
if (length(trajectory_results) == 0L) {
  trajectory_data <- data.table::data.table(
    pathway_uid = character(), crest_id = integer(), crest_label = character(),
    peak_age = numeric(), pathway_rank = integer(), pathway_y = numeric(),
    database = character(), term_id = character(), description = character(),
    pathway_label = character(), age = numeric(), smoothed_mean_transcript_z = numeric(),
    pathway_trajectory_z = numeric(), trajectory_status = character()
  )
  pathway_summary <- data.table::data.table(
    pathway_uid = character(), n_measured_transcripts = integer(),
    n_present_transcripts = integer(), n_qc_transcripts = integer(),
    transcript_coverage_fraction = numeric(), n_finite_pathway_samples = integer(),
    median_observed_transcripts_per_sample = numeric(), smoothed_score_amplitude = numeric(),
    crest_delta_z = numeric(), local_one_year_change_z = numeric(),
    qc_transcripts = character(), trajectory_status = character()
  )
}

pathway_key <- merge(
  pathways,
  pathway_summary,
  by = "pathway_uid",
  all.x = TRUE,
  sort = FALSE
)
data.table::setorder(
  pathway_key,
  crest_id, pathway_rank
)
if (nrow(trajectory_data) != nrow(pathways) * length(AGE_GRID)) {
  stop("Unexpected pathway-by-age source-data dimensions")
}

observed_max_abs_z <- max(c(0, abs(trajectory_data$pathway_trajectory_z)), na.rm = TRUE)
automatic_colour_limit <- max(2.5, ceiling(observed_max_abs_z * 2) / 2)
colour_limit_env <- Sys.getenv("PATHWAY_HEATMAP_Z_LIMIT", unset = "")
COLOUR_LIMIT <- if (nzchar(colour_limit_env)) {
  as.numeric(colour_limit_env)
} else {
  automatic_colour_limit
}
assert_positive_scalar(COLOUR_LIMIT, "pathway heatmap Z limit")
LEGEND_TICK <- max(0.5, COLOUR_LIMIT - 0.5)
n_colour_squished <- sum(
  abs(trajectory_data$pathway_trajectory_z) > COLOUR_LIMIT,
  na.rm = TRUE
)


# =============================================================================
# 6. Source data and audit records
# =============================================================================

source_data_path <- file.path(
  OUTPUT_DIR, "pathway_molecular_expression_heatmap_source_data.csv"
)
pathway_key_path <- file.path(
  OUTPUT_DIR, "pathway_molecular_expression_heatmap_pathway_key.csv"
)
gene_qc_path <- file.path(
  OUTPUT_DIR, "pathway_molecular_expression_heatmap_gene_qc.csv"
)

pathway_key_export <- data.table::copy(pathway_key)
pathway_key_export[, measured_transcript_list := NULL]
data.table::fwrite(trajectory_data, source_data_path)
data.table::fwrite(pathway_key_export, pathway_key_path)
data.table::fwrite(gene_qc, gene_qc_path)

# Keep counts and crest ages available without crowding the 100-mm figure.
crest_summary <- data.table::rbindlist(lapply(seq_len(nrow(crests)), function(i) {
  crest <- crests[i]
  selected <- pathways[crest_id == crest$crest_id]
  selected_transcripts <- unique(unlist(selected$measured_transcript_list, use.names = FALSE))
  data.table::data.table(
    crest_id = crest$crest_id,
    crest = crest$crest_display_label,
    peak_age = crest$peak_age,
    significant_peak_transcripts = crest$peak_n,
    significant_pathways_all_databases = enrichment[
      crest_id == crest$crest_id & is.finite(p.adjust) & p.adjust < FDR_THRESHOLD, .N
    ],
    displayed_significant_pathways = nrow(selected),
    contributing_transcripts_before_qc = length(selected_transcripts),
    contributing_transcripts_after_qc = length(intersect(selected_transcripts, included_transcripts)),
    unavailable_trajectories = pathway_key[crest_id == crest$crest_id, sum(trajectory_status != "ok")]
  )
}))
crest_summary_csv <- file.path(OUTPUT_DIR, "pathway_molecular_expression_heatmap_crest_summary.csv")
crest_summary_document <- file.path(OUTPUT_DIR, "pathway_molecular_expression_heatmap_summary.md")
data.table::fwrite(crest_summary, crest_summary_csv)
writeLines(c(
  "# Pathway molecular-expression heatmap: counts and display notes",
  "",
  "| Crest | Peak age (years) | Significant peak transcripts | Significant pathways (all databases) | Displayed significant pathways | Contributing transcripts after QC | Unavailable trajectories |",
  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
  vapply(seq_len(nrow(crest_summary)), function(i) {
    row <- crest_summary[i]
    paste0("| ", paste(c(
      row$crest, format_age(row$peak_age), row$significant_peak_transcripts,
      row$significant_pathways_all_databases, row$displayed_significant_pathways,
      row$contributing_transcripts_after_qc, row$unavailable_trajectories
    ), collapse = " | "), " |")
  }, character(1)),
  "",
  paste0("- Significant peak transcripts: DE-SWAN BH q < ", GENE_FDR_THRESHOLD, " at the corresponding peak age."),
  paste0("- Significant pathways: BH-adjusted P < ", FDR_THRESHOLD, ", corrected separately within each crest and database. Counts retain distinct database/term IDs; similar terms across databases are not merged."),
  paste0("- Displayed pathways: up to ", TOP_PATHWAYS_PER_CREST, " significant terms per crest, ranked by ", PATHWAY_RANKING_RULE, "."),
  "- Contributing transcripts: unique measured, crest-associated transcript features in the displayed pathways that pass expression QC; shared features are counted once per crest. These are transcript-feature counts, not a separate Entrez-gene count.",
  "- Unavailable trajectories remain grey in the heatmap.",
  "- Pathway ranks, database types, term IDs, adjusted P values and per-pathway transcript counts remain in the pathway-key CSV.",
  paste0("- Figure dimensions: ", FIGURE_WIDTH_MM, " x ", FIGURE_HEIGHT_MM, " mm. Crest1 is blue; Crest2 is orange. Titles are centered over their heatmap panels."),
  paste0("- Machine-readable crest summary: ", basename(crest_summary_csv), ".")
), crest_summary_document)

sample_audit <- data.table::data.table(
  metric = c(
    "samples_before_age_filter",
    "samples_after_age_filter",
    "samples_excluded_nonfinite_age",
    "samples_excluded_outside_age_range",
    "unique_participants_after_filter",
    "requested_transcripts",
    "transcripts_present_in_expression",
    "transcripts_passing_qc",
    "transcripts_excluded",
    "pathways_retained",
    "age_grid_points",
    "colour_cells_squished"
  ),
  value = c(
    samples_before_age_filter,
    samples_after_age_filter,
    sum(!finite_age),
    sum(finite_age & !in_age_range),
    data.table::uniqueN(expression$participant_id),
    length(requested_transcripts),
    length(present_transcripts),
    length(included_transcripts),
    sum(!gene_qc$included),
    nrow(pathways),
    length(AGE_GRID),
    n_colour_squished
  )
)
data.table::fwrite(
  sample_audit,
  file.path(OUTPUT_DIR, "pathway_molecular_expression_heatmap_sample_audit.csv")
)

manifest <- c(
  paste0("analysis_date=", Sys.Date()),
  paste0("deswan_rds_file=", normalizePath(rds_path, winslash = "/")),
  paste0("source_pathway_result=", normalizePath(selected_pathway_path, winslash = "/")),
  "pathway_filter=at most 15 terms with BH-adjusted P < 0.05 per crest, pooled for display across GO BP, KEGG and Reactome",
  "fdr_scope=all tested terms within each crest/database; no pooled cross-database FDR claim",
  paste0("ranking_rule=", PATHWAY_RANKING_RULE),
  paste0("full_enrichment_directory=", normalizePath(ENRICHMENT_RESULT_DIR, winslash = "/")),
  "member_transcripts=saved measured_transcripts from the matching enrichment table",
  "sample_filter=finite age within inclusive 40-70-year range",
  "sample_subsampling=none",
  "transcript_standardization=(expression-sample_mean)/sample_SD across retained observations",
  "pathway_sample_score=unweighted mean of available standardized member transcripts",
  paste0("trajectory_model=LOESS degree 1; Gaussian family; span=", LOESS_SPAN),
  paste0("age_grid=", AGE_MIN, "-", AGE_MAX, " by ", AGE_GRID_STEP, " years"),
  "heatmap_value=per-pathway Z score of the fitted pathway trajectory across the common age grid",
  "crest_delta=mean trajectory Z at peak+0.5..2 years minus peak-2..0.5 years",
  paste0("shared_colour_limit=", COLOUR_LIMIT),
  paste0("colour_cells_squished=", n_colour_squished),
  paste0("figure_dimensions_mm=", FIGURE_WIDTH_MM, "x", FIGURE_HEIGHT_MM),
  "pathway_axis_labels=pathway names only; no ranks or database prefixes",
  paste0("count_summary_document=", basename(crest_summary_document)),
  paste0("raster_exports=PNG_", FIGURE_PREVIEW_DPI, "dpi;TIFF_", FIGURE_TIFF_DPI, "dpi"),
  "vector_exports=editable SVG;PDF",
  paste0("R=", R.version.string),
  paste0("data.table=", packageVersion("data.table")),
  paste0("ggplot2=", packageVersion("ggplot2")),
  paste0("patchwork=", packageVersion("patchwork"))
)
writeLines(
  manifest,
  file.path(OUTPUT_DIR, "pathway_molecular_expression_heatmap_manifest.txt")
)


# =============================================================================
# 7. Nature-style two-panel heatmap
# =============================================================================

# Saturated blue -> white (z = 0) -> restrained terracotta red.
# Intermediate stops make typical |z| values visible while retaining the full
# symmetric colour range. The colourbar uses exactly the same mapping.
HEATMAP_COLOURS <- c(
  "#2166AC", "#4393C3", "#FFFFFF", "#D58A80", "#B85C56"
)
HEATMAP_MID_Z <- min(0.8, COLOUR_LIMIT / 2)
HEATMAP_COLOUR_ANCHORS <- c(
  -COLOUR_LIMIT, -HEATMAP_MID_Z, 0, HEATMAP_MID_Z, COLOUR_LIMIT
)

theme_pathway_heatmap <- function(crest_colour) {
  ggplot2::theme_classic(base_size = 6, base_family = "Helvetica") +
    ggplot2::theme(
      axis.line.y = ggplot2::element_blank(),
      axis.line.x = ggplot2::element_line(colour = "#252525", linewidth = 0.35),
      axis.ticks.y = ggplot2::element_blank(),
      axis.ticks.x = ggplot2::element_line(colour = "#252525", linewidth = 0.35),
      axis.ticks.length.x = grid::unit(0.8, "mm"),
      axis.title.x = ggplot2::element_text(
        size = 6, colour = "#222222", margin = ggplot2::margin(t = 1, unit = "mm")
      ),
      axis.title.y = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(size = 5.5, colour = "#303030"),
      axis.text.y = ggplot2::element_text(
        size = 6.5, colour = "#202020", hjust = 1, lineheight = 0.9,
        margin = ggplot2::margin(r = 1, unit = "mm")
      ),
      plot.title = ggplot2::element_text(
        size = 7.5, face = "bold", colour = crest_colour, hjust = 0.5,
        margin = ggplot2::margin(t = 0.4, b = 1, unit = "mm")
      ),
      plot.title.position = "panel",
      plot.subtitle = ggplot2::element_blank(),
      legend.title = ggplot2::element_text(size = 5.5, face = "bold"),
      legend.text = ggplot2::element_text(
        size = 5.2,
        margin = ggplot2::margin(t = 0.5, unit = "mm")
      ),
      legend.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm"),
      legend.box.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm"),
      legend.box.spacing = grid::unit(1.5, "mm"),
      panel.grid = ggplot2::element_blank(),
      plot.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm")
    )
}

make_crest_panel <- function(crest_id, show_x_title = TRUE) {
  target_crest_id <- crest_id
  panel_pathways <- pathways[crest_id == target_crest_id]
  panel_data <- trajectory_data[crest_id == target_crest_id]
  crest_row <- crests[crest_id == target_crest_id]
  n_pathways <- nrow(panel_pathways)
  regular_ticks <- seq(AGE_MIN, AGE_MAX, by = 5)
  tick_breaks <- sort(unique(c(regular_ticks[abs(regular_ticks - crest_row$peak_age) >= 2], crest_row$peak_age)))
  plot <- ggplot2::ggplot(panel_data, ggplot2::aes(
    x = .data$age, y = .data$pathway_y, fill = .data$pathway_trajectory_z
  )) +
    ggplot2::geom_raster(interpolate = FALSE) +
    ggplot2::geom_vline(xintercept = if (n_pathways > 0L) crest_row$peak_age else numeric(), colour = "#6D3A31", linewidth = 0.38, linetype = "22") +
    ggplot2::geom_hline(
      yintercept = if (n_pathways > 1L) MAX_DISPLAY_ROWS - seq_len(n_pathways - 1L) + 0.5 else numeric(),
      colour = "#FFFFFF", linewidth = 0.12
    ) +
    ggplot2::scale_fill_gradientn(
      colours = HEATMAP_COLOURS,
      values = scales::rescale(HEATMAP_COLOUR_ANCHORS),
      limits = c(-COLOUR_LIMIT, COLOUR_LIMIT), breaks = c(-LEGEND_TICK, 0, LEGEND_TICK),
      labels = c(paste0("-", safe_number(LEGEND_TICK, 1L)), "0", paste0("+", safe_number(LEGEND_TICK, 1L))),
      oob = scales::squish, na.value = "#DEDEDE",
      name = "Pathway expression trajectory (z-score)",
      guide = ggplot2::guide_colourbar(
        title.position = "top", title.hjust = 0.5,
        barwidth = grid::unit(40, "mm"), barheight = grid::unit(2, "mm"),
        ticks = TRUE, frame.colour = "#5B6065", frame.linewidth = 0.25
      )
    ) +
    ggplot2::scale_x_continuous(
      limits = c(AGE_MIN - AGE_GRID_STEP / 2, AGE_MAX + AGE_GRID_STEP / 2),
      breaks = tick_breaks, expand = ggplot2::expansion(mult = 0)
    ) +
    ggplot2::scale_y_continuous(
      limits = c(0.5, MAX_DISPLAY_ROWS + 0.5), breaks = panel_pathways$pathway_y,
      labels = panel_pathways$pathway_label, expand = ggplot2::expansion(mult = 0)
    ) +
    ggplot2::coord_cartesian(expand = FALSE, clip = "off") +
    ggplot2::labs(
      title = crest_row$crest_display_label,
      x = if (show_x_title) "Age (years)" else NULL, y = NULL
    ) + theme_pathway_heatmap(crest_row$crest_colour) +
    ggplot2::theme(plot.margin = ggplot2::margin(
      # Reserve only the half-width of the rightmost age label at the page edge.
      t = if (target_crest_id == crests$crest_id[1]) 0 else 2, r = 1.1, unit = "mm"
    ))
  if (n_pathways == 0L) {
    plot <- plot + ggplot2::annotate(
      "text", x = (AGE_MIN + AGE_MAX) / 2, y = (MAX_DISPLAY_ROWS + 1) / 2,
      label = "No pathways with BH-adjusted P < 0.05", size = 6 / ggplot2::.pt,
      colour = "#555555"
    )
  }
  plot
}

crest_levels <- crests$crest_id
panels <- lapply(seq_along(crest_levels), function(i) {
  make_crest_panel(crest_levels[i], show_x_title = i == length(crest_levels))
})
figure <- (patchwork::wrap_plots(panels, ncol = 1, heights = rep(1, length(panels)), guides = "collect") +
  patchwork::plot_annotation(theme = ggplot2::theme(
    plot.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm")
  ))) &
  ggplot2::theme(legend.position = "bottom", plot.tag = ggplot2::element_blank())


# =============================================================================
# 8. Exports
# =============================================================================

render_plot_device <- function(open_device, plot) {
  open_device()
  device_id <- grDevices::dev.cur()
  on.exit({
    devices <- grDevices::dev.list()
    if (!is.null(devices) && device_id %in% devices) {
      grDevices::dev.off(device_id)
    }
  }, add = TRUE)
  print(plot)
  grDevices::dev.off(device_id)
  invisible(NULL)
}

width_in <- FIGURE_WIDTH_MM / 25.4
height_in <- FIGURE_HEIGHT_MM / 25.4
render_plot_device(
  function() svglite::svglite(
    paste0(FIGURE_STEM, ".svg"),
    width = width_in,
    height = height_in,
    bg = "white",
    pointsize = 7
  ),
  figure
)
render_plot_device(
  function() grDevices::pdf(
    paste0(FIGURE_STEM, ".pdf"),
    width = width_in,
    height = height_in,
    family = "Helvetica",
    onefile = TRUE,
    bg = "white",
    useDingbats = FALSE
  ),
  figure
)
render_plot_device(
  function() ragg::agg_tiff(
    paste0(FIGURE_STEM, ".tiff"),
    width = width_in,
    height = height_in,
    units = "in",
    res = FIGURE_TIFF_DPI,
    compression = "lzw",
    background = "white"
  ),
  figure
)
render_plot_device(
  function() ragg::agg_png(
    paste0(FIGURE_STEM, ".png"),
    width = width_in,
    height = height_in,
    units = "in",
    res = FIGURE_PREVIEW_DPI,
    background = "white"
  ),
  figure
)

qa_notes <- c(
  FIGURE_CONTRACT,
  paste0("Samples before/after age filter: ", samples_before_age_filter, "/", samples_after_age_filter, "."),
  paste0("Unique participants after filter: ", data.table::uniqueN(expression$participant_id), "."),
  paste0("Requested/present/QC-passing transcripts: ", length(requested_transcripts), "/", length(present_transcripts), "/", length(included_transcripts), "."),
  paste0("Pathways retained: ", nrow(pathways), " (", paste(pathways_per_crest$N, collapse = " + "), ")."),
  paste0("Per-crest pathway and transcript counts: ", basename(crest_summary_document), "; ", basename(crest_summary_csv), "."),
  paste0("Shared heatmap colour limit: +/-", safe_number(COLOUR_LIMIT, 1L), "; squished cells: ", n_colour_squished, "."),
  paste0("Heatmap palette (low to high): ", paste(HEATMAP_COLOURS, collapse = " -> "), "; zero is white."),
  paste0("Colour anchors (trajectory z): ", paste(HEATMAP_COLOUR_ANCHORS, collapse = ", "), ". Intermediate stops increase colour contrast; the heatmap and colourbar share this mapping."),
  "All selected transcripts contribute equally within a pathway; genes shared by multiple pathways intentionally contribute to every annotated pathway.",
  "Repeated participant samples are retained as observed expression samples; no inferential test or confidence interval is shown.",
  "Only measured transcripts significant in the matching peak window contribute to each pathway score.",
  "Selected pathways with insufficient expression coverage or an unavailable fit remain as grey rows; none are replaced by lower-ranked pathways.",
  "Fewer than 15 significant terms are shown at their actual count; blank panel space is not additional data.",
  paste0("Source data: ", basename(source_data_path), "; ", basename(pathway_key_path), "; ", basename(gene_qc_path), ".")
)
writeLines(
  qa_notes,
  file.path(FIGURE_DIR, "pathway_molecular_expression_heatmap_contract_and_qa_notes.txt")
)

message("Pathway molecular-expression heatmap complete.")
message("Figure: ", normalizePath(paste0(FIGURE_STEM, ".pdf"), winslash = "/"))
print(sample_audit)
