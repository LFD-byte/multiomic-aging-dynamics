# Annotation and exact ORA helpers for the five-cluster pathway landscape.
# No graphics devices or analysis side effects are opened when this file is sourced.

landscape_term_key <- function(source, term_id) paste(source, term_id, sep = "::")

landscape_resolve_entrez <- function(assignments, links) {
  links <- links |>
    dplyr::mutate(feature = as.character(.data$feature), entrez_id = as.character(.data$entrez_id)) |>
    dplyr::filter(!is.na(.data$entrez_id), nzchar(.data$entrez_id)) |>
    dplyr::distinct()
  audit <- assignments |>
    dplyr::left_join(links, by = "feature", relationship = "one-to-many") |>
    dplyr::group_by(.data$feature) |>
    dplyr::mutate(
      n_entrez_ids = dplyr::n_distinct(.data$entrez_id, na.rm = TRUE),
      mapping_status = dplyr::case_when(
        .data$n_entrez_ids == 0L ~ "unmapped",
        .data$n_entrez_ids > 1L ~ "ambiguous_identifier",
        TRUE ~ "accepted"
      )
    ) |>
    dplyr::ungroup()
  conflicts <- audit |>
    dplyr::filter(.data$mapping_status == "accepted") |>
    dplyr::group_by(.data$entrez_id) |>
    dplyr::summarise(n_clusters = dplyr::n_distinct(.data$group_id), .groups = "drop") |>
    dplyr::filter(.data$n_clusters > 1L) |>
    dplyr::pull("entrez_id")
  audit <- audit |>
    dplyr::mutate(mapping_status = dplyr::if_else(
      .data$mapping_status == "accepted" & .data$entrez_id %in% conflicts,
      "cross_cluster_conflict", .data$mapping_status
    )) |>
    dplyr::arrange(.data$group_id, .data$feature, .data$entrez_id)
  accepted <- audit |> dplyr::filter(.data$mapping_status == "accepted")
  units <- accepted |>
    dplyr::group_by(.data$entrez_id, .data$group_id, .data$group_label) |>
    dplyr::summarise(
      feature_original = paste(sort(unique(.data$feature_original)), collapse = ";"),
      .groups = "drop"
    ) |>
    dplyr::transmute(
      feature_original = .data$feature_original, feature = .data$entrez_id,
      group_id = .data$group_id, group_label = .data$group_label
    ) |>
    dplyr::arrange(.data$group_id, .data$feature)
  coverage <- audit |>
    dplyr::group_by(.data$group_id, .data$group_label) |>
    dplyr::summarise(
      input_features = dplyr::n_distinct(.data$feature),
      accepted_features = dplyr::n_distinct(.data$feature[.data$mapping_status == "accepted"]),
      accepted_entrez_genes = dplyr::n_distinct(.data$entrez_id[.data$mapping_status == "accepted"]),
      unmapped_features = dplyr::n_distinct(.data$feature[.data$mapping_status == "unmapped"]),
      ambiguous_features = dplyr::n_distinct(.data$feature[.data$mapping_status == "ambiguous_identifier"]),
      cross_cluster_conflict_features = dplyr::n_distinct(.data$feature[.data$mapping_status == "cross_cluster_conflict"]),
      .groups = "drop"
    )
  list(
    audit = audit, assignments = units, coverage = coverage,
    feature_map = accepted |>
      dplyr::transmute(analysis_feature = .data$entrez_id, measured_feature = .data$feature) |>
      dplyr::distinct()
  )
}

# Match identifiers before resolving ambiguity. Lookup tables have columns
# lookup_key, entrez_id and matched_symbol; this pure helper is fixture-testable.
landscape_match_entrez <- function(assignments, symbol_lookup, alias_lookup, ensembl_lookup) {
  inputs <- assignments |>
    dplyr::transmute(
      feature = .data$feature, original_key = trimws(.data$feature_original),
      input_type = dplyr::if_else(grepl("^ENSG[0-9]+(\\.[0-9]+)?$", .data$feature), "ENSEMBL", "SYMBOL")
    )
  match_table <- function(query, lookup) {
    lookup <- lookup |>
      dplyr::filter(!is.na(.data$entrez_id), nzchar(.data$entrez_id)) |>
      dplyr::mutate(join_key = toupper(.data$lookup_key))
    query |>
      dplyr::inner_join(lookup, by = "join_key", relationship = "many-to-many") |>
      dplyr::select(-"join_key")
  }
  symbol_inputs <- inputs |>
    dplyr::filter(.data$input_type == "SYMBOL") |>
    dplyr::mutate(join_key = toupper(.data$original_key))
  symbol_links <- match_table(symbol_inputs, symbol_lookup) |>
    dplyr::mutate(priority = dplyr::if_else(.data$original_key == .data$lookup_key, 1L, 2L)) |>
    dplyr::group_by(.data$feature) |>
    dplyr::filter(.data$priority == min(.data$priority)) |>
    dplyr::ungroup() |>
    dplyr::mutate(mapping_method = dplyr::if_else(.data$priority == 1L, "symbol_exact", "symbol_case_normalized")) |>
    dplyr::select(-"priority")
  # A current SYMBOL takes precedence over aliases, including when its own
  # mapping is ambiguous. Never use an alias to arbitrarily resolve that ambiguity.
  alias_links <- match_table(
    symbol_inputs |> dplyr::filter(!.data$feature %in% symbol_links$feature), alias_lookup
  ) |>
    dplyr::mutate(mapping_method = "alias_case_normalized")
  ensembl_links <- match_table(
    inputs |>
      dplyr::filter(.data$input_type == "ENSEMBL") |>
      dplyr::mutate(join_key = sub("\\.[0-9]+$", "", toupper(.data$original_key))),
    ensembl_lookup
  ) |>
    dplyr::mutate(mapping_method = "ensembl_version_stripped")
  dplyr::bind_rows(symbol_links, alias_links, ensembl_links) |>
    dplyr::select(-"original_key", -"input_type") |>
    dplyr::distinct()
}

landscape_map_entrez <- function(assignments) {
  orgdb <- org.Hs.eg.db::org.Hs.eg.db
  is_ensembl <- grepl("^ENSG[0-9]+(\\.[0-9]+)?$", assignments$feature)
  assignments$input_type <- ifelse(is_ensembl, "ENSEMBL", "SYMBOL")
  fetch_lookup <- function(type, requested) {
    empty <- data.frame(lookup_key = character(), entrez_id = character(), matched_symbol = character())
    if (!length(requested)) return(empty)
    valid <- AnnotationDbi::keys(orgdb, keytype = type)
    hits <- valid[toupper(valid) %in% toupper(requested)]
    if (!length(hits)) return(empty)
    raw <- suppressMessages(AnnotationDbi::select(
      orgdb, keys = hits, keytype = type, columns = c("ENTREZID", "SYMBOL")
    ))
    data.frame(lookup_key = raw[[type]], entrez_id = as.character(raw$ENTREZID), matched_symbol = raw$SYMBOL)
  }
  symbols <- fetch_lookup("SYMBOL", assignments$feature[!is_ensembl])
  aliases <- fetch_lookup("ALIAS", assignments$feature[!is_ensembl])
  ensembl <- fetch_lookup("ENSEMBL", sub("\\.[0-9]+$", "", assignments$feature[is_ensembl]))
  links <- landscape_match_entrez(assignments, symbols, aliases, ensembl)
  result <- landscape_resolve_entrez(assignments, links)
  result$audit <- result$audit |>
    dplyr::mutate(mapping_method = dplyr::coalesce(.data$mapping_method, "unmapped"))
  if (!nrow(result$assignments)) stop("No unambiguous Entrez genes remain after mapping checks")
  result$id_type <- paste(sort(unique(assignments$input_type)), collapse = "+")
  result$method_coverage <- result$audit |>
    dplyr::distinct(.data$feature, .data$group_id, .data$mapping_method, .data$mapping_status) |>
    dplyr::count(.data$group_id, .data$mapping_method, .data$mapping_status, name = "input_features")
  result$orgdb_metadata <- AnnotationDbi::metadata(orgdb)
  result
}

landscape_load_go_bp <- function(entrez_ids) {
  raw <- suppressMessages(AnnotationDbi::select(
    org.Hs.eg.db::org.Hs.eg.db, keys = unique(entrez_ids), keytype = "ENTREZID",
    columns = c("GOALL", "ONTOLOGYALL")
  ))
  membership <- raw |>
    dplyr::filter(.data$ONTOLOGYALL == "BP", !is.na(.data$GOALL)) |>
    dplyr::transmute(term_id = .data$GOALL, feature = as.character(.data$ENTREZID)) |>
    dplyr::distinct()
  if (!nrow(membership)) stop("No GO Biological Process annotations overlap the mapped genes")
  descriptions <- suppressMessages(AnnotationDbi::select(
    GO.db::GO.db, keys = unique(membership$term_id), keytype = "GOID", columns = "TERM"
  ))
  terms <- descriptions |>
    dplyr::transmute(term_id = .data$GOID, term = dplyr::coalesce(.data$TERM, .data$GOID)) |>
    dplyr::distinct()
  list(
    terms = terms, membership = membership,
    provenance = list(
      mode = "offline org.Hs.eg.db GOALL, BP ontology, including ancestor terms",
      labels = "GO.db TERM", orgdb_version = as.character(utils::packageVersion("org.Hs.eg.db")),
      go_db_version = as.character(utils::packageVersion("GO.db"))
    )
  )
}

landscape_load_kegg <- function(entrez_ids, annotation_file) {
  if (!file.exists(annotation_file)) {
    stop("KEGG snapshot is missing: ", annotation_file,
         ". Run update_cluster_pathway_kegg.R or set KEGG_ANNOTATION_FILE to a frozen snapshot. ",
         "The obsolete org.Hs.eg.db PATH fallback is no longer used.")
  }
  env <- new.env(parent = emptyenv())
  load(annotation_file, envir = env)
  if (!all(c("term2gene", "term2name") %in% ls(env))) {
    stop("KEGG annotation RData must contain term2gene and term2name")
  }
  term2gene <- as.data.frame(env$term2gene, stringsAsFactors = FALSE)
  term2name <- as.data.frame(env$term2name, stringsAsFactors = FALSE)
  if (ncol(term2gene) != 2L || ncol(term2name) != 2L) stop("KEGG annotation objects need exactly two columns")
  provenance <- list(
    mode = "frozen offline TERM2GENE/TERM2NAME",
    annotation_file = normalizePath(annotation_file, winslash = "/"),
    annotation_md5 = unname(tools::md5sum(annotation_file)),
    snapshot = if (exists("kegg_provenance", envir = env, inherits = FALSE)) env$kegg_provenance else NULL
  )
  if (!is.null(provenance$snapshot$retrieved_utc)) {
    message("KEGG frozen snapshot retrieved: ", provenance$snapshot$retrieved_utc)
  }
  names(term2gene) <- c("term_id", "feature")
  names(term2name) <- c("term_id", "term")
  membership <- term2gene |>
    dplyr::transmute(
      term_id = sub("^path:", "", trimws(as.character(.data$term_id))),
      feature = sub("^hsa:", "", trimws(as.character(.data$feature)))
    ) |>
    dplyr::filter(grepl("^hsa[0-9]{5}$", .data$term_id), .data$feature %in% entrez_ids) |>
    dplyr::distinct()
  terms <- term2name |>
    dplyr::transmute(
      term_id = sub("^path:", "", trimws(as.character(.data$term_id))),
      term = sub(" - Homo sapiens \\(human\\)$", "", trimws(as.character(.data$term)))
    ) |>
    dplyr::distinct()
  if (anyDuplicated(terms$term_id)) stop("KEGG pathway names conflict for the same term ID")
  terms <- data.frame(term_id = sort(unique(membership$term_id))) |>
    dplyr::left_join(terms, by = "term_id") |>
    dplyr::mutate(term = dplyr::if_else(is.na(.data$term) | !nzchar(.data$term), .data$term_id, .data$term))
  if (!nrow(membership)) stop("No KEGG annotations overlap the mapped genes")
  list(terms = terms, membership = membership, provenance = provenance)
}

# Strictly validate the three official REST tables before creating a snapshot.
# Human KEGG IDs are explicitly converted to NCBI Gene IDs rather than assumed
# to be interchangeable. These functions never upload measured gene identifiers.
landscape_build_kegg_tables <- function(pathway_links, pathway_names, gene_conversion) {
  if (any(vapply(list(pathway_links, pathway_names, gene_conversion), ncol, integer(1)) != 2L)) {
    stop("Each KEGG REST table must have exactly two columns")
  }
  links <- stats::setNames(pathway_links, c("kegg_gene", "term_id"))
  names_table <- stats::setNames(pathway_names, c("term_id", "term"))
  conversion <- stats::setNames(gene_conversion, c("kegg_gene", "entrez_id"))
  valid <- nrow(links) > 0L && nrow(names_table) > 0L && nrow(conversion) > 0L &&
    all(grepl("^hsa:[0-9]+$", links$kegg_gene)) &&
    all(grepl("^(path:)?hsa[0-9]{5}$", links$term_id)) &&
    all(grepl("^(path:)?hsa[0-9]{5}$", names_table$term_id)) &&
    all(grepl("^hsa:[0-9]+$", conversion$kegg_gene)) &&
    all(grepl("^ncbi-geneid:[0-9]+$", conversion$entrez_id))
  if (!isTRUE(valid)) stop("Malformed, empty or nonhuman KEGG REST response")
  links$term_id <- sub("^path:", "", links$term_id)
  names_table$term_id <- sub("^path:", "", names_table$term_id)
  conversion$entrez_id <- sub("^ncbi-geneid:", "", conversion$entrez_id)
  conversion <- unique(conversion)
  names_table <- unique(names_table)
  if (anyDuplicated(conversion$kegg_gene)) stop("Ambiguous KEGG-to-Entrez conversion")
  if (anyDuplicated(names_table$term_id)) stop("Conflicting KEGG pathway descriptions")
  if (!all(links$term_id %in% names_table$term_id)) stop("KEGG pathway links lack descriptions")
  if (!all(links$kegg_gene %in% conversion$kegg_gene)) stop("KEGG pathway genes lack Entrez conversion")
  term2gene <- links |>
    dplyr::inner_join(conversion, by = "kegg_gene", relationship = "many-to-one") |>
    dplyr::transmute(term_id = .data$term_id, feature = .data$entrez_id) |>
    dplyr::distinct() |>
    dplyr::arrange(.data$term_id, .data$feature)
  term2name <- names_table |>
    dplyr::mutate(term = sub(" - Homo sapiens \\(human\\)$", "", .data$term)) |>
    dplyr::arrange(.data$term_id)
  list(term2gene = term2gene, term2name = term2name)
}

landscape_prepare_database <- function(annotation, assignments, feature_map, source,
                                       min_set_size, max_set_fraction) {
  if (anyDuplicated(assignments$feature)) stop(source, ": analysis genes must have unique cluster assignments")
  universe <- sort(intersect(assignments$feature, unique(annotation$membership$feature)))
  if (!length(universe)) stop(source, ": no annotated analysis background")
  annotated <- assignments |>
    dplyr::filter(.data$feature %in% universe) |>
    dplyr::arrange(.data$group_id, .data$feature)
  membership <- annotation$membership |>
    dplyr::filter(.data$feature %in% universe) |>
    dplyr::distinct(.data$term_id, .data$feature) |>
    dplyr::arrange(.data$term_id, .data$feature)
  sets <- lapply(split(membership$feature, membership$term_id), unique)
  set_sizes <- lengths(sets)
  maximum <- max(min_set_size, floor(max_set_fraction * length(universe)))
  eligible <- names(sets)[set_sizes >= min_set_size & set_sizes <= maximum]
  if (!length(eligible)) stop(source, ": no terms pass the set-size filters")
  sets <- sets[eligible]
  catalog <- annotation$terms |>
    dplyr::filter(.data$term_id %in% eligible) |>
    dplyr::mutate(set_size = as.integer(lengths(sets)[.data$term_id])) |>
    dplyr::arrange(.data$term_id)
  if (anyDuplicated(catalog$term_id) || !setequal(catalog$term_id, eligible)) {
    stop(source, ": each tested term must have exactly one description")
  }
  membership <- membership |> dplyr::filter(.data$term_id %in% eligible)
  measured_membership <- membership |>
    dplyr::inner_join(feature_map, by = c("feature" = "analysis_feature"), relationship = "many-to-many") |>
    dplyr::distinct(.data$term_id, .data$measured_feature)
  display_sets <- lapply(split(measured_membership$measured_feature, measured_membership$term_id), unique)
  if (!setequal(names(display_sets), eligible)) stop(source, ": measured-feature provenance is incomplete")
  list(
    source = source, universe = universe, assignments = annotated, membership = membership,
    term_catalog = catalog, term_sets = sets, display_sets = display_sets,
    measured_membership = measured_membership,
    terms_with_measured_genes = length(set_sizes), maximum_set_size = maximum,
    provenance = annotation$provenance
  )
}

landscape_run_ora <- function(context, group_ids, group_labels, fdr_threshold, fdr_cap) {
  sets <- context$term_sets
  set_sizes <- lengths(sets)
  universe_size <- length(context$universe)
  term_names <- stats::setNames(context$term_catalog$term, context$term_catalog$term_id)
  results <- lapply(group_ids, function(group_id) {
    group_genes <- context$assignments$feature[context$assignments$group_id == group_id]
    group_size <- length(group_genes)
    rest_size <- universe_size - group_size
    if (group_size == 0L || rest_size == 0L) stop(context$source, ": empty annotated group/rest for C", group_id)
    overlap <- vapply(sets, function(members) length(intersect(group_genes, members)), integer(1))
    pathway_size <- as.integer(set_sizes[names(sets)])
    overlap_rest <- pathway_size - overlap
    outside_group <- group_size - overlap
    outside_rest <- rest_size - overlap_rest
    p_value <- stats::phyper(overlap - 1L, pathway_size, universe_size - pathway_size,
                            group_size, lower.tail = FALSE)
    data.frame(
      group_id = group_id, group_label = unname(group_labels[as.character(group_id)]),
      source = context$source, term_id = names(sets), term = unname(term_names[names(sets)]),
      universe_size = universe_size, set_size = pathway_size, group_size = group_size,
      rest_size = rest_size, overlap_group = overlap, overlap_rest = overlap_rest,
      gene_ratio = overlap / group_size, background_ratio = pathway_size / universe_size,
      fold_enrichment = (overlap / group_size) / (pathway_size / universe_size),
      log2_odds_ratio = log2(((overlap + 0.5) * (outside_rest + 0.5)) /
                             ((outside_group + 0.5) * (overlap_rest + 0.5))),
      p_value = as.numeric(p_value), stringsAsFactors = FALSE
    ) |>
      dplyr::mutate(
        fdr = stats::p.adjust(.data$p_value, method = "BH"),
        minus_log10_fdr = -log10(pmax(.data$fdr, .Machine$double.xmin)),
        minus_log10_fdr_capped = pmin(.data$minus_log10_fdr, fdr_cap),
        significant = .data$fdr < fdr_threshold & .data$log2_odds_ratio > 0
      ) |>
      dplyr::arrange(.data$fdr, dplyr::desc(.data$log2_odds_ratio), .data$term_id)
  })
  dplyr::bind_rows(results) |>
    dplyr::arrange(.data$group_id, .data$fdr, dplyr::desc(.data$log2_odds_ratio), .data$term_id)
}
