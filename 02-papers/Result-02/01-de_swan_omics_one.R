# =============================================================================
# DE-SWAN stage 1: prepare data, fit the models and save the raw RDS
#
# This script runs only the time-consuming DEswan fits; significant counts, crests
# and figures are produced downstream from the saved RDS.
# Select the omics layer with DESWAN_OMICS (default: transcriptomics).
# buckets.size is the width on each side of the window centre.
# =============================================================================

local({
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg)) setwd(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
})

library(DEswan)
library(data.table)

set.seed(42)


# =============================================================================
# 1. Global configuration
# =============================================================================

AGE_MIN <- 40
AGE_MAX <- 70
MIN_SAMPLES <- 50
REUSE_DESWAN_RDS <- tolower(
  Sys.getenv("DESWAN_REUSE_RDS", unset = "false")
) %in% c("1", "true", "yes")
DESWAN_RDS_PATH <- Sys.getenv("DESWAN_RDS_PATH", unset = "")

# DEswan sliding-window parameters
WINDOW_CENTERS <- seq(AGE_MIN, AGE_MAX, by = 1)
BUCKET_SIZE <- 7.5

WINDOW_FILE_TAG <- paste0(
  "window_",
  gsub(
    "\\.",
    "p",
    format(BUCKET_SIZE, trim = TRUE, scientific = FALSE)
  )
)

# Covariates; also excluded from the feature matrix to prevent leakage
COVARIATE_COLS <- c("sex")

# Metadata columns excluded from the feature matrix
META_COLS_REMOVE <- c(
  "participant_id", "cohort", "research_stage", "array_index",
  "collection_date", "collection_timestamp", "age", "year_of_birth",
  "month_of_birth", "sex", "collection_datetime", "batch", "bmi",
  "pred_age", "omics", "collection_time"
)

PHENOTYPE_COL <- "age"

# Omics configuration; select the layer with DESWAN_OMICS
OMICS_CONFIG <- list(
  transcriptomics = list(
    label = "Transcriptomics",
    input = "../01-interim_data/omics_data/transcriptomics_with_metadata.csv"
  ),
  metabolomics = list(
    label = "Metabolomics",
    input = "../01-interim_data/omics_data/metabolomics_with_metadata.csv"
  ),
  lipidomics = list(
    label = "Lipidomics",
    input = "../01-interim_data/omics_data/lipidomics_with_metadata.csv"
  ),
  gut_microbiome = list(
    label = "Gut Microbiome",
    input = "../01-interim_data/omics_data/gut_microbiome_with_metadata.csv"
  ),
  oral_microbiome = list(
    label = "Oral Microbiome",
    input = "../01-interim_data/omics_data/oral_microbiome_with_metadata.csv"
  )
)

OMICS_KEY <- tolower(Sys.getenv("DESWAN_OMICS", unset = "transcriptomics"))
if (!OMICS_KEY %in% names(OMICS_CONFIG)) {
  stop(
    "Unknown DESWAN_OMICS='", OMICS_KEY, "'. Valid values: ",
    paste(names(OMICS_CONFIG), collapse = ", ")
  )
}
omics_config <- OMICS_CONFIG[[OMICS_KEY]]
omics_label <- omics_config$label

# Microbiome layers that receive the CLR transform
MICROBIOME_KEYS <- c("gut_microbiome", "oral_microbiome")

# Output directory for raw RDS files (override with DESWAN_OUTPUT_DIR)
OUTPUT_DIR <- Sys.getenv(
  "DESWAN_OUTPUT_DIR",
  unset = file.path("outputs", paste0("deswan-window_", BUCKET_SIZE))
)

# =============================================================================
# 2. Helper functions
# =============================================================================

#' Create directories.
ensure_dir <- function(...) {
  invisible(
    lapply(
      c(...),
      dir.create,
      recursive = TRUE,
      showWarnings = FALSE
    )
  )
}


#' Build an output file name tagged with the DEswan window width,
#' e.g. `<stem>_window_7p5.<extension>`.
windowed_output_filename <- function(stem, extension) {
  extension <- sub("^\\.", "", extension)
  paste0(stem, "_", WINDOW_FILE_TAG, ".", extension)
}


#' Centred log-ratio (CLR) transform for microbiome relative abundances
#' (samples × features).
clr_transform <- function(mat) {
  mat <- mat + 1
  mat <- mat / rowSums(mat)
  log_mat <- log(mat)

  log_mat - rowMeans(log_mat)
}


#' Prepare DEswan input.
#'
#' @return List with the feature matrix, phenotype vector and covariates, or
#'   `NULL` if the layer cannot be analysed.
prepare_deswan_input <- function(df, omics_name) {
  df <- as.data.frame(df)

  # Age filter
  if (!PHENOTYPE_COL %in% names(df)) {
    warning(
      "[", omics_name, "] '", PHENOTYPE_COL,
      "' column missing, skipping."
    )
    return(NULL)
  }

  df[[PHENOTYPE_COL]] <- suppressWarnings(
    as.numeric(df[[PHENOTYPE_COL]])
  )
  df <- df[
    !is.na(df[[PHENOTYPE_COL]]) &
      df[[PHENOTYPE_COL]] >= AGE_MIN &
      df[[PHENOTYPE_COL]] <= AGE_MAX,
  ]

  if (nrow(df) == 0) {
    warning(
      "[", omics_name,
      "] No rows remain after age filter, skipping."
    )
    return(NULL)
  }

  phenotype_vec <- df[[PHENOTYPE_COL]]

  # Covariates
  covariate_names <- intersect(COVARIATE_COLS, names(df))
  covariates <- NULL

  if (length(covariate_names) > 0) {
    covariates <- df[, covariate_names, drop = FALSE]

    if ("sex" %in% names(covariates)) {
      covariates$sex <- as.factor(covariates$sex)
    }
  }

  # Feature matrix: drop metadata and covariate columns
  columns_to_exclude <- union(META_COLS_REMOVE, COVARIATE_COLS)
  feature_df <- df[
    , !names(df) %in% columns_to_exclude,
    drop = FALSE
  ]

  # Keep numeric columns only
  is_numeric <- vapply(feature_df, is.numeric, logical(1))
  feature_df <- feature_df[, is_numeric, drop = FALSE]

  # CLR-transform microbiome layers before QC filtering
  if (omics_name %in% MICROBIOME_KEYS && ncol(feature_df) > 0) {
    feature_mat <- clr_transform(as.matrix(feature_df))
    feature_df <- as.data.frame(feature_mat)
  }

  # Filter by number of valid samples and variance
  is_valid <- vapply(
    feature_df,
    function(x) {
      valid_values <- x[!is.na(x)]
      length(valid_values) >= MIN_SAMPLES && sd(valid_values) > 1e-10
    },
    logical(1)
  )
  feature_df <- feature_df[, is_valid, drop = FALSE]

  if (ncol(feature_df) == 0) {
    warning("[", omics_name, "] No features pass QC, skipping.")
    return(NULL)
  }

  message(
    sprintf(
      "  [%s] %d samples x %d features (after QC)",
      omics_name,
      nrow(feature_df),
      ncol(feature_df)
    )
  )

  list(
    data_feature = feature_df,
    phenotype_vec = phenotype_vec,
    covariates = covariates
  )
}


#' Run DEswan for one omics layer.
#'
#' @param prep Output of `prepare_deswan_input()`.
#' @return DEswan result, or `NULL` if the analysis fails.
run_deswan_one <- function(prep, omics_name, rds_dir = OUTPUT_DIR) {
  if (is.null(prep)) {
    return(NULL)
  }

  rds_path <- file.path(
    rds_dir,
    windowed_output_filename(
      paste0(omics_name, "_deswan_breakpoint"),
      "rds"
    )
  )
  if (REUSE_DESWAN_RDS && file.exists(rds_path)) {
    message(sprintf("  [%s] Reusing DEswan RDS: %s", omics_name, rds_path))
    cached_result <- tryCatch(
      readRDS(rds_path),
      error = function(error) {
        warning(
          "[", omics_name, "] Could not read cached RDS: ",
          conditionMessage(error),
          "; recomputing DEswan."
        )
        NULL
      }
    )
    if (!is.null(cached_result)) {
      return(cached_result)
    }
  }

  message(sprintf("  [%s] Running DEswan ...", omics_name))

  result <- tryCatch(
    DEswan(
      data.df = prep$data_feature,
      qt = prep$phenotype_vec,
      window.center = WINDOW_CENTERS,
      buckets.size = BUCKET_SIZE,
      covariates = prep$covariates
    ),
    error = function(error) {
      warning(
        "[", omics_name, "] DEswan() failed: ",
        conditionMessage(error)
      )
      NULL
    }
  )

  if (is.null(result)) {
    return(NULL)
  }

  saveRDS(result, file = rds_path)
  message(sprintf("  [%s] RDS saved: %s", omics_name, rds_path))

  result
}


# =============================================================================
# 3. Run the fits
# =============================================================================

run_deswan_analysis <- function() {
  ensure_dir(OUTPUT_DIR)

  message("\n", strrep("#", 60))
  message("Step 1: Prepare data and run DEswan")
  message(strrep("#", 60))

  omics_name <- OMICS_KEY
  cat(sprintf("\n[%s]\n", omics_name))

  rds_path <- file.path(
    OUTPUT_DIR,
    windowed_output_filename(
      paste0(omics_name, "_deswan_breakpoint"),
      "rds"
    )
  )
  cache_rds_path <- if (nzchar(DESWAN_RDS_PATH)) {
    DESWAN_RDS_PATH
  } else {
    rds_path
  }

  # A cache hit skips the large raw CSV entirely. This is safe only when the
  # caller explicitly opts in and knows that input data and DE-SWAN settings are
  # unchanged; the default remains a full recomputation.
  if (REUSE_DESWAN_RDS && file.exists(cache_rds_path)) {
    message(sprintf(
      "  [%s] Reusing DEswan RDS: %s",
      omics_name,
      cache_rds_path
    ))
    deswan_results <- readRDS(cache_rds_path)
  } else {
    if (!file.exists(omics_config$input)) {
      stop("Input file not found: ", omics_config$input)
    }
    df_omics <- fread(omics_config$input)
    prepared_data <- prepare_deswan_input(df_omics, omics_name)
    deswan_results <- run_deswan_one(prepared_data, omics_name)
  }

  if (is.null(deswan_results)) {
    stop("DEswan did not produce a result. Check the input and model error above.")
  }
  message("DEswan fitting stage complete. For peak detection and figures, run ",
          "01-plot_deswan_omics_one.R with the same omics and bucket size.")
  invisible(deswan_results)
}

deswan_results <- run_deswan_analysis()
