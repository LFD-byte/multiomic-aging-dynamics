# Explicitly refresh the local human KEGG annotation from the official REST API.
# Normal figure runs remain offline. Raw downloads and dated snapshots are kept.
# Usage: Rscript update_cluster_pathway_kegg.R [output_RData_path]
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", grep("^--file=", args, value = TRUE)[[1]])
script_dir <- dirname(normalizePath(script, winslash = "/"))
extra_library <- Sys.getenv("TRANSCRIPTOMICS_R_LIBS", unset = "")
if (nzchar(extra_library)) .libPaths(c(extra_library, .libPaths()))
for (package in c("dplyr", "jsonlite")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing package: ", package)
}
source(file.path(script_dir, "transcriptome_cluster_pathway_methods.R"))
cli <- commandArgs(trailingOnly = TRUE)
if (length(cli) > 1L) stop("Supply at most one output RData path")
destination <- if (length(cli)) cli[[1]] else file.path(
  script_dir, "resources", "kegg_hsa", "KEGG_offline_data_hsa.RData"
)
dir.create(dirname(destination), recursive = TRUE, showWarnings = FALSE)
snapshot_dir <- file.path(dirname(destination), "snapshots", format(Sys.time(), "%Y%m%dT%H%M%SZ", tz = "UTC"))
if (dir.exists(snapshot_dir)) stop("Snapshot directory already exists: ", snapshot_dir)
dir.create(snapshot_dir, recursive = TRUE)
urls <- c(
  pathway_info = "https://rest.kegg.jp/info/pathway",
  human_info = "https://rest.kegg.jp/info/hsa",
  pathway_links = "https://rest.kegg.jp/link/pathway/hsa",
  pathway_names = "https://rest.kegg.jp/list/pathway/hsa",
  gene_conversion = "https://rest.kegg.jp/conv/ncbi-geneid/hsa"
)
paths <- stats::setNames(file.path(snapshot_dir, paste0(names(urls), ".txt")), names(urls))
options(timeout = max(120, getOption("timeout")))
for (name in names(urls)) {
  message("Downloading ", urls[[name]])
  # KEGG requests no more than three calls/second; use sequential downloads.
  Sys.sleep(0.4)
  status <- utils::download.file(urls[[name]], paths[[name]], method = "libcurl", mode = "wb", quiet = TRUE)
  if (status != 0L || !file.exists(paths[[name]]) || file.info(paths[[name]])$size == 0) {
    stop("KEGG download failed for ", name, "; the existing snapshot has not been replaced")
  }
}
read_pairs <- function(path) {
  utils::read.delim(path, header = FALSE, sep = "\t", quote = "", comment.char = "",
                    colClasses = "character", check.names = FALSE, fileEncoding = "UTF-8")
}
tables <- landscape_build_kegg_tables(
  read_pairs(paths[["pathway_links"]]), read_pairs(paths[["pathway_names"]]),
  read_pairs(paths[["gene_conversion"]])
)
pathway_info <- readLines(paths[["pathway_info"]], warn = FALSE)
human_info <- readLines(paths[["human_info"]], warn = FALSE)
pathway_version <- trimws(grep("Release|Last update", pathway_info, value = TRUE))
human_version <- trimws(grep("Release|Last update", human_info, value = TRUE))
if (!length(pathway_version) || !any(grepl("Homo sapiens", human_info, fixed = TRUE))) {
  stop("KEGG pathway version or human-organism identity is missing")
}
term2gene <- tables$term2gene
term2name <- tables$term2name
kegg_provenance <- list(
  source = "Official KEGG REST API", api_documentation = "https://www.kegg.jp/kegg/rest/keggapi.html",
  organism = "hsa", organism_name = "Homo sapiens", taxon_id = 9606L,
  retrieved_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  pathway_version = pathway_version,
  gene_version = if (length(human_version)) human_version else NULL,
  pathway_info = pathway_info, human_info = human_info,
  gene_identifier_conversion = "Official conv/ncbi-geneid/hsa; validated unique KEGG-to-Entrez mapping",
  download_resources = data.frame(
    resource = names(urls), url = unname(urls), filename = basename(paths),
    md5 = unname(tools::md5sum(paths)), bytes = file.info(paths)$size
  ),
  counts = list(pathway_names = nrow(term2name), pathways_with_genes = length(unique(term2gene$term_id)),
                unique_entrez_genes = length(unique(term2gene$feature)), membership_pairs = nrow(term2gene)),
  acquisition_script_md5 = unname(tools::md5sum(file.path(script_dir, "update_cluster_pathway_kegg.R")))
)
snapshot_file <- file.path(snapshot_dir, "KEGG_offline_data_hsa.RData")
save(term2gene, term2name, kegg_provenance, file = snapshot_file, compress = "xz")
jsonlite::write_json(
  c(kegg_provenance, list(snapshot_md5 = unname(tools::md5sum(snapshot_file)))),
  file.path(snapshot_dir, "snapshot_manifest.json"), pretty = TRUE, auto_unbox = TRUE
)
# Validate the exact serialized artifact before replacing the default local copy.
validated <- landscape_load_kegg(unique(term2gene$feature), snapshot_file)
if (nrow(validated$membership) != nrow(term2gene)) stop("Serialized KEGG snapshot lost membership pairs")
if (!file.copy(snapshot_file, destination, overwrite = TRUE)) stop("Cannot write default KEGG snapshot")
if (!identical(unname(tools::md5sum(snapshot_file)), unname(tools::md5sum(destination)))) {
  stop("Default KEGG snapshot checksum does not match the dated archive")
}
message("KEGG snapshot ready: ", normalizePath(destination, winslash = "/"))
print(kegg_provenance$counts)
message("Pathway version: ", paste(kegg_provenance$pathway_version, collapse = "; "))
message("Archive: ", normalizePath(snapshot_dir, winslash = "/"))
