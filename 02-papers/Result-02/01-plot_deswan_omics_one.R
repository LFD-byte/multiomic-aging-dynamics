# =============================================================================
# DE-SWAN stage 2: detect crests from the raw RDS and plot
#
# Input: *_deswan_breakpoint_window_*.rds saved by 01-de_swan_omics_one.R.
# Output: per-window significant counts, breakpoints, candidate peaks, representative
#   and stable main peaks, stability tables and figures.
# Everything is recomputed from the RDS; DEswan() is never refitted. After changing the
#   FDR, peak rules or plot settings, only this script needs to be rerun.
# The 40 × 40 mm main panel shows the raw counts as coloured lines and points (no
#   smoothing). Peaks are marked by vertical lines (blue, then red, by increasing age)
#   and labelled with age, count and proportion.
# Boundary limitation and parameter stability are recorded separately; stable main peaks
#   also require complete windows and parameter support. No participant resampling.
#
# DESWAN_OMICS / DESWAN_BUCKET_SIZE must match the RDS (defaults: gut_microbiome / 7.5).
# DESWAN_RDS_PATH reads a specific RDS; DESWAN_OUTPUT_DIR / DESWAN_FIGURE_DIR set the
#   table and figure directories. --data-only updates the tables only.
# =============================================================================

local({
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg)) setwd(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
})

library(DEswan)
library(dplyr)
library(ggplot2)


# =============================================================================
# 1. Omics layer and window width
# =============================================================================

OMICS_KEY <- tolower(Sys.getenv("DESWAN_OMICS", unset = "gut_microbiome"))
BUCKET_SIZE <- as.numeric(Sys.getenv("DESWAN_BUCKET_SIZE", unset = "7.5"))
if (!is.finite(BUCKET_SIZE) || BUCKET_SIZE <= 0) {
  stop("BUCKET_SIZE / DESWAN_BUCKET_SIZE must be a positive number.")
}
WINDOW_FILE_TAG <- paste0(
  "window_",
  gsub("\\.", "p", format(BUCKET_SIZE, trim = TRUE, scientific = FALSE))
)


# =============================================================================
# 2. Omics configuration
# =============================================================================

OMICS_CONFIG <- list(

  transcriptomics = list(
    label = "Transcriptomics",
    y_label = "Significant transcripts"
  ),

  metabolomics = list(
    label = "Metabolomics",
    y_label = "Significant metabolites"
  ),

  lipidomics = list(
    label = "Lipidomics",
    y_label = "Significant lipids"
  ),

  gut_microbiome = list(
    label = "Gut Microbiome",
    y_label = "Significant microbial taxa"
  ),

  oral_microbiome = list(
    label = "Oral Microbiome",
    y_label = "Significant microbial taxa"
  )
)


# Validate OMICS_KEY
if (!OMICS_KEY %in% names(OMICS_CONFIG)) {

  stop(
    "Unknown OMICS_KEY: ",
    OMICS_KEY,
    "\nValid values are: ",
    paste(names(OMICS_CONFIG), collapse = ", ")
  )
}


omics_name <- OMICS_KEY

omics_label <- OMICS_CONFIG[[OMICS_KEY]]$label

y_axis_label <- OMICS_CONFIG[[OMICS_KEY]]$y_label


# =============================================================================
# 3. Paths
# =============================================================================

OUTPUT_DIR <- Sys.getenv(
  "DESWAN_OUTPUT_DIR",
  unset = file.path("outputs", paste0("deswan-window_", BUCKET_SIZE))
)
FIGURE_DIR <- Sys.getenv(
  "DESWAN_FIGURE_DIR",
  unset = file.path("figures", paste0("deswan-window_", BUCKET_SIZE))
)
DESWAN_RDS_PATH <- Sys.getenv("DESWAN_RDS_PATH", unset = "")

# Post-processing parameters are set here and do not affect the saved model fits.
FDR_THRESHOLD <- as.numeric(Sys.getenv("DESWAN_FDR_THRESHOLD", unset = "0.05"))
if (!is.finite(FDR_THRESHOLD) || FDR_THRESHOLD <= 0 || FDR_THRESHOLD > 1) {
  stop("FDR_THRESHOLD / DESWAN_FDR_THRESHOLD must be in (0, 1].")
}
FDR_METHOD <- "BH"
GAP_THRESHOLD <- 3
PEAK_MIN_RELATIVE_HEIGHT <- 0.15
PEAK_MIN_DISTANCE <- 4
MAX_VALLEY_RATIO <- 0.85

# Shared rules for candidate and main peaks.
PEAK_METHOD_VERSION <- "peak-v6"
PEAK_MIN_PROMINENCE_FRACTION <- 0.10
PEAK_MIN_RELATIVE_PROMINENCE <- 0.25
# Require a substantial drop relative to this peak's own height. Candidate
# amplitude, distance and valley filters above still exclude small ripples.
MAIN_MIN_RELATIVE_PROMINENCE <- 0.50
# Width measured on raw line segments at 20% of prominence above the baseline.
# At annual age centres, require a wave wider than an isolated single-age spike.
PEAK_WIDTH_RELATIVE_HEIGHT <- 0.80
MAIN_MIN_WIDTH_YEARS <- 2
STABILITY_WINDOWS <- c(5, 7.5, 10, 12.5, 15)
STABILITY_Q_CUTOFFS <- c(0.0001, 0.001, 0.01, 0.05)
STABILITY_CORE_Q <- c(0.01, 0.05)
STABILITY_MATCH_TOLERANCE <- 2
MAIN_MIN_PARAMETER_SUPPORT <- 0.50
MAIN_MIN_SUPPORTING_WINDOWS <- 2L
MAIN_MIN_SUPPORTING_Q <- 2L
DESWAN_RESULTS_ROOT <- Sys.getenv("DESWAN_RESULTS_ROOT", unset = "outputs")

# =============================================================================
# 4. Plot settings
# =============================================================================

AGE_MIN <- 40
AGE_MAX <- 70


# Final size of the main panel: 40 × 40 mm; fonts and line widths are set at actual size.
MAIN_W <- 40 / 25.4
MAIN_H <- 40 / 25.4


# Ridge plot size
RIDGE_W <- 89 / 25.4
RIDGE_H <- 52 / 25.4


# Parameters of the legacy smoothing diagnostics; the main panel shows raw counts only.
DISPLAY_SMOOTHING_BANDWIDTH <- 1
DISPLAY_CURVE_POINTS <- 401L
RAW_LINE_WIDTH <- 0.42
RAW_POINT_SIZE <- 0.65
PEAK_GUIDE_LINE_WIDTH <- 0.30
# By increasing peak age: first peak blue, second red, others neutral grey.
PEAK_GUIDE_COLORS <- c(first = "#3A7FC1", second = "#D64B4B", other = "#A0A0A0")
PLOT_FONT_FAMILY <- "Arial"

# Upward offset of peak labels
PEAK_LABEL_OFFSET <- 6


# Headroom for three-line peak labels.
Y_TOP_EXPANSION <- 1.38


# =============================================================================
# 5. Colours
# =============================================================================

OMICS_COLORS <- c(

  "Transcriptomics" = "#E05A5A",

  "Metabolomics" = "#3A7FC1",

  "Lipidomics" = "#2CA67A",

  "Gut Microbiome" = "#E88C2A",

  "Oral Microbiome" = "#9B59B6"
)


omics_color <- unname(
  OMICS_COLORS[omics_label]
)


# =============================================================================
# 6. Post-processing helpers
# =============================================================================

# Same naming rule as the analysis script, e.g. lipidomics_crest_window_7p5.csv.
windowed_output_filename <- function(stem, extension) {
  extension <- sub("^\\.", "", extension)
  paste0(stem, "_", WINDOW_FILE_TAG, ".", extension)
}

# Prefer the original location, with compatibility for archived endogenous results.
# An explicit DESWAN_RDS_PATH always takes precedence over this lookup.
resolve_deswan_rds <- function(directory, filename) {
  primary <- file.path(directory, filename)
  archived <- file.path(directory, "endogenous", filename)
  if (file.exists(primary) || !file.exists(archived)) return(primary)
  message("Using archived DEswan RDS: ", archived)
  archived
}



#' Detect independent local peaks directly in raw significant-feature counts.
#' No display smoother may change peak age, height, prominence or separation.
#' Self-contained so the Fig. 4 sensitivity script can import this function only.
detect_crests <- function(
    x, y, min_peak_relative_height = 0.15,
    min_peak_distance = 4, max_valley_ratio = 0.85,
    min_prominence_fraction = 0.10, min_relative_prominence = 0.25,
    plateau_tolerance = 1e-10, width_relative_height = 0.80) {
  if (!is.numeric(x) || !is.numeric(y) || length(x) != length(y) || length(x) < 3L ||
      any(!is.finite(x)) || is.unsorted(x) || anyDuplicated(x)) {
    stop("x must be finite and strictly increasing; x and y need at least 3 paired values.")
  }
  if (any(y[is.finite(y)] < 0)) stop("Finite counts must be non-negative.")
  fractions <- c(min_peak_relative_height, max_valley_ratio,
                 min_prominence_fraction, min_relative_prominence)
  if (any(!is.finite(fractions)) || any(fractions < 0 | fractions > 1) ||
      !is.finite(min_peak_distance) || min_peak_distance < 0 ||
      !is.finite(plateau_tolerance) || plateau_tolerance < 0 ||
      length(width_relative_height) != 1L || !is.finite(width_relative_height) ||
      width_relative_height <= 0 || width_relative_height > 1) {
    stop("Invalid peak-detection thresholds.")
  }
  empty_crests <- data.frame(
    crest_id = integer(), peak_age = numeric(), peak_n = numeric(),
    raw_n_at_peak = numeric(), detection_peak_age = numeric(), detection_peak_n = numeric(),
    plateau_start = numeric(), plateau_end = numeric(), plateau_n = integer(),
    prominence = numeric(), prominence_fraction = numeric(), relative_prominence = numeric(),
    relative_height = numeric(), left_base_age = numeric(), right_base_age = numeric(),
    width_relative_height = numeric(), width_height = numeric(),
    width_left_age = numeric(), width_right_age = numeric(), peak_width_years = numeric(),
    n_candidate_peaks = integer(), candidate_peak_ages = character()
  )
  candidates <- data.frame(
    candidate_id = integer(), detection_index = integer(), segment_id = integer(),
    detection_age = numeric(), detection_n = numeric(), raw_n_at_peak = numeric(),
    plateau_left_index = integer(), plateau_right_index = integer(),
    plateau_start = numeric(), plateau_end = numeric(), plateau_n = integer(),
    prominence = numeric(), prominence_fraction = numeric(), relative_prominence = numeric(),
    relative_height = numeric(), left_base_age = numeric(), right_base_age = numeric(),
    width_relative_height = numeric(), width_height = numeric(),
    width_left_age = numeric(), width_right_age = numeric(), peak_width_years = numeric(),
    valley_group_id = integer(), valley_representative_id = integer(),
    crest_id = integer(), decision = character()
  )
  separations <- data.frame(
    comparison_stage = character(), candidate_id = integer(), candidate_age = numeric(), representative_id = integer(),
    representative_age = numeric(), peak_distance = numeric(), valley_age = numeric(),
    valley_n_raw = numeric(), valley_ratio = numeric(),
    distinct_from_representative = logical(), final_decision = character(),
    assigned_crest_id = integer()
  )
  valid <- is.finite(y)
  segment_id <- cumsum(c(TRUE, diff(as.integer(valid)) != 0L))
  segments <- split(which(valid), segment_id[valid])
  raw_y <- ifelse(valid, y, NA_real_)
  curve <- data.frame(age = x, n_raw = raw_y, valid = valid,
                      segment_id = ifelse(valid, segment_id, NA_integer_))
  finish_empty <- function() list(crests = empty_crests, candidates = candidates,
                                  separations = separations, detection_curve = curve)
  scale <- if (any(valid)) max(raw_y[valid]) else 0
  if (scale <= 0) return(finish_empty())
  tol <- plateau_tolerance * max(1, scale)
  rows <- list()
  for (indices in segments) {
    if (length(indices) < 3L) next
    z <- raw_y[indices]
    # Compare to the first value of each run; do not chain nearly equal values.
    starts <- 1L
    for (j in seq.int(2L, length(z))) {
      if (abs(z[j] - z[tail(starts, 1L)]) > tol) starts <- c(starts, j)
    }
    ends <- c(starts[-1L] - 1L, length(z))
    for (r in seq_along(starts)) {
      lo <- starts[r]; hi <- ends[r]
      if (lo == 1L || hi == length(z)) next
      height <- mean(z[lo:hi])
      if (!(height > z[lo - 1L] + tol && height > z[hi + 1L] + tol)) next
      # Use an observed age so the same peak can be queried in downstream RDS tables.
      centre_age <- mean(x[indices[c(lo, hi)]])
      mid <- (lo:hi)[which.min(abs(x[indices[lo:hi]] - centre_age))]
      idx <- indices[mid]
      left <- lo
      while (left > 1L && z[left - 1L] <= height + tol) left <- left - 1L
      right <- hi
      while (right < length(z) && z[right + 1L] <= height + tol) right <- right + 1L
      left_min <- min(z[left:lo]); right_min <- min(z[hi:right])
      # Equal minima: choose the one closest to the plateau, deterministically.
      lb <- tail((left:lo)[abs(z[left:lo] - left_min) <= tol], 1L)
      rb <- (hi:right)[which(abs(z[hi:right] - right_min) <= tol)[1L]]
      prominence <- max(0, height - max(left_min, right_min))
      width_height <- height - prominence * width_relative_height
      wl <- lo; wr <- hi
      while (wl > lb && z[wl] > width_height) wl <- wl - 1L
      while (wr < rb && z[wr] > width_height) wr <- wr + 1L
      # Linear crossings describe the raw polyline; peak age is never interpolated.
      width_left <- x[indices[wl]]
      if (z[wl] < width_height && wl < lo) width_left <- width_left +
        (width_height - z[wl]) / (z[wl + 1L] - z[wl]) *
        (x[indices[wl + 1L]] - x[indices[wl]])
      width_right <- x[indices[wr]]
      if (z[wr] < width_height && wr > hi) width_right <- width_right -
        (width_height - z[wr]) / (z[wr - 1L] - z[wr]) *
        (x[indices[wr]] - x[indices[wr - 1L]])
      rows[[length(rows) + 1L]] <- data.frame(
        candidate_id = length(rows) + 1L, detection_index = idx,
        segment_id = segment_id[idx], detection_age = x[idx], detection_n = raw_y[idx],
        raw_n_at_peak = y[idx], plateau_left_index = indices[lo], plateau_right_index = indices[hi],
        plateau_start = x[indices[lo]], plateau_end = x[indices[hi]], plateau_n = hi - lo + 1L,
        prominence = prominence, prominence_fraction = prominence / scale,
        relative_prominence = prominence / height, relative_height = height / scale,
        left_base_age = x[indices[lb]], right_base_age = x[indices[rb]],
        width_relative_height = width_relative_height, width_height = width_height,
        width_left_age = width_left, width_right_age = width_right,
        peak_width_years = width_right - width_left,
        valley_group_id = NA_integer_, valley_representative_id = NA_integer_,
        crest_id = NA_integer_, decision = "pending"
      )
    }
  }
  if (!length(rows)) return(finish_empty()) # No forced endpoint/global-maximum peak.
  candidates <- do.call(rbind, rows)
  representatives <- integer()
  comparisons <- list()
  processing <- order(-candidates$prominence, -candidates$detection_n, candidates$detection_age)
  compare_pair <- function(i, j, stage) {
    a <- min(candidates$detection_index[c(i, j)])
    b <- max(candidates$detection_index[c(i, j)])
    interior <- if (b - a > 1L) seq.int(a + 1L, b - 1L) else integer()
    same_segment <- candidates$segment_id[i] == candidates$segment_id[j]
    if (length(interior) && same_segment) {
      v <- interior[which.min(raw_y[interior])]
      valley <- raw_y[v]; valley_age <- x[v]
    } else {
      valley <- if (same_segment) min(raw_y[c(a, b)]) else NA_real_
      valley_age <- mean(x[c(a, b)])
    }
    ratio <- valley / min(candidates$detection_n[c(i, j)])
    distance <- abs(candidates$detection_age[i] - candidates$detection_age[j])
    data.frame(comparison_stage = stage, candidate_id = i,
               candidate_age = candidates$detection_age[i], representative_id = j,
               representative_age = candidates$detection_age[j], peak_distance = distance,
               valley_age = valley_age, valley_n_raw = valley, valley_ratio = ratio,
               distinct_from_representative =
                 (stage == "valley_grouping" || distance >= min_peak_distance) &&
                 (!same_segment || ratio <= max_valley_ratio))
  }
  for (i in processing) {
    candidates$decision[i] <- if (candidates$relative_height[i] < min_peak_relative_height) {
      "below_height"
    } else if (candidates$prominence_fraction[i] < min_prominence_fraction ||
               candidates$relative_prominence[i] < min_relative_prominence) {
      "below_prominence"
    } else "pending"
  }
  eligible <- which(candidates$decision == "pending")
  if (!length(eligible)) return(finish_empty())
  # Merge shallow-valley components BEFORE distance suppression. Otherwise a
  # discarded member (oral age 51) can hide a shallow link to a later peak (54).
  component <- seq_len(nrow(candidates))
  if (length(eligible) > 1L) for (k in seq.int(2L, length(eligible))) {
    i <- eligible[k]; j <- eligible[k - 1L]
    pair <- compare_pair(i, j, "valley_grouping")
    if (!pair$distinct_from_representative) {
      component[component == component[i]] <- component[j]
    }
    comparisons[[length(comparisons) + 1L]] <- pair
  }
  groups <- split(eligible, component[eligible])
  for (members in groups) {
    representative <- processing[processing %in% members][1L]
    candidates$valley_group_id[members] <- min(members)
    candidates$valley_representative_id[members] <- representative
  }
  group_representatives <- processing[processing %in% unique(candidates$valley_representative_id[eligible])]
  for (i in group_representatives) {
    pair_rows <- lapply(representatives, function(j) compare_pair(i, j, "representative_filter"))
    pairs <- if (length(pair_rows)) do.call(rbind, pair_rows) else NULL
    if (is.null(pairs) || all(pairs$distinct_from_representative)) {
      representatives <- c(representatives, i)
      candidates$crest_id[i] <- i
      candidates$decision[i] <- "independent"
    } else {
      blocking <- which(!pairs$distinct_from_representative)
      k <- blocking[which.min(pairs$peak_distance[blocking])]
      candidates$crest_id[i] <- pairs$representative_id[k]
      candidates$decision[i] <- if (pairs$peak_distance[k] < min_peak_distance) "merge_distance" else "merge_valley"
    }
    if (!is.null(pairs)) {
      comparisons[[length(comparisons) + 1L]] <- pairs
    }
  }
  for (i in setdiff(eligible, group_representatives)) {
    candidates$crest_id[i] <- candidates$crest_id[candidates$valley_representative_id[i]]
    candidates$decision[i] <- "merge_valley_group"
  }
  if (length(comparisons)) {
    separations <- do.call(rbind, comparisons)
    separations$final_decision <- candidates$decision[separations$candidate_id]
    separations$assigned_crest_id <- candidates$crest_id[separations$candidate_id]
  }
  representatives <- representatives[order(candidates$detection_age[representatives])]
  if (!length(representatives)) return(finish_empty())
  id_map <- setNames(seq_along(representatives), representatives)
  candidates$crest_id <- as.integer(id_map[as.character(candidates$crest_id)])
  separations$assigned_crest_id <- as.integer(id_map[as.character(separations$assigned_crest_id)])
  crests <- do.call(rbind, lapply(seq_along(representatives), function(k) {
    p <- candidates[representatives[k], ]
    members <- which(candidates$crest_id == k)
    data.frame(crest_id = k, peak_age = p$detection_age, peak_n = p$detection_n,
      raw_n_at_peak = p$raw_n_at_peak, detection_peak_age = p$detection_age,
      detection_peak_n = p$detection_n, plateau_start = p$plateau_start,
      plateau_end = p$plateau_end, plateau_n = p$plateau_n,
      prominence = p$prominence, prominence_fraction = p$prominence_fraction,
      relative_prominence = p$relative_prominence, relative_height = p$relative_height,
      left_base_age = p$left_base_age, right_base_age = p$right_base_age,
      width_relative_height = p$width_relative_height, width_height = p$width_height,
      width_left_age = p$width_left_age, width_right_age = p$width_right_age,
      peak_width_years = p$peak_width_years,
      n_candidate_peaks = length(members),
      candidate_peak_ages = paste(candidates$detection_age[members], collapse = ";"))
  }))
  stopifnot(identical(crests$peak_age, crests$detection_peak_age),
            identical(crests$peak_n, crests$detection_peak_n),
            all(crests$peak_n == y[match(crests$peak_age, x)]),
            all(diff(crests$peak_age) >= min_peak_distance))
  list(crests = crests, candidates = candidates, separations = separations, detection_curve = curve)
}


#' Continuous, non-negative Gaussian kernel regression for display only.
#' A normalized positive-weight average cannot overshoot the observed count range.
#' Evaluate each tested segment separately; never bridge gaps or extrapolate.
build_display_curve <- function(detection_curve,
                                bandwidth = DISPLAY_SMOOTHING_BANDWIDTH,
                                n_points = DISPLAY_CURVE_POINTS) {
  if (length(bandwidth) != 1L || !is.finite(bandwidth) || bandwidth <= 0 ||
      length(n_points) != 1L || !is.finite(n_points) || n_points < 2 || n_points %% 1 != 0) {
    stop("Display bandwidth must be positive; n_points must be an integer >= 2.")
  }
  valid <- detection_curve$valid & is.finite(detection_curve$age) &
    is.finite(detection_curve$n_raw)
  segments <- split(which(valid), detection_curve$segment_id[valid])
  if (!length(segments)) return(data.frame(age = numeric(), n_display = numeric(),
                                          segment_id = integer()))
  curves <- lapply(segments, function(indices) {
    x <- detection_curve$age[indices]
    y <- detection_curve$n_raw[indices]
    if (any(y < 0) || is.unsorted(x) || anyDuplicated(x)) {
      stop("Display data must contain non-negative counts at increasing ages.")
    }
    grid <- if (length(x) == 1L) x else sort(unique(c(
      seq(min(x), max(x), length.out = as.integer(n_points)), x)))
    log_weights <- -0.5 * (outer(grid, x, "-") / bandwidth)^2
    row_max <- apply(log_weights, 1L, max)
    if (any(!is.finite(row_max))) stop("Display bandwidth is too small for this age grid.")
    weights <- exp(sweep(log_weights, 1L, row_max, "-"))
    data.frame(age = grid, n_display = as.numeric(weights %*% y) / rowSums(weights),
               segment_id = detection_curve$segment_id[indices[1L]])
  })
  result <- do.call(rbind, curves)
  rownames(result) <- NULL
  result
}


#' Extract breakpoint ages.
#'
#' Results are reshaped with `reshape.DEswan()` and `q.DEswan()` and significant windows
#' selected by the FDR threshold; significant windows separated by a large gap form
#' separate breakpoint regions.
#'
#' @param deswan_result Output of `DEswan()`.
#' @param gap_threshold Age gap between windows that separates breakpoint regions.
#' @return Data frame of breakpoints, or `NULL` if none is detected.
extract_breakpoints <- function(
    deswan_result,
    omics_name,
    fdr_threshold = FDR_THRESHOLD,
    gap_threshold = GAP_THRESHOLD) {
  if (is.null(deswan_result)) {
    return(NULL)
  }

  p_wide <- reshape.DEswan(
    deswan_result,
    parameter = 1,
    factor = "qt"
  )
  q_wide <- q.DEswan(p_wide, method = FDR_METHOD)

  # Strip the X prefix from column names and convert to numeric ages
  window_ages <- as.numeric(sub("^X", "", colnames(q_wide)[-1]))
  features <- q_wide$variable

  breakpoint_list <- lapply(seq_along(features), function(i) {
    feature <- features[i]
    q_values <- as.numeric(q_wide[i, -1])
    significant_indices <- which(
      !is.na(q_values) & q_values < fdr_threshold
    )

    if (length(significant_indices) == 0) {
      return(NULL)
    }

    significant_ages <- window_ages[significant_indices]

    if (length(significant_ages) == 1) {
      breakpoints <- significant_ages
    } else {
      gaps <- diff(significant_ages)
      splits <- c(
        0,
        which(gaps > gap_threshold),
        length(significant_ages)
      )
      breakpoints <- vapply(
        seq_len(length(splits) - 1),
        function(j) {
          segment <- significant_ages[
            (splits[j] + 1):splits[j + 1]
          ]
          median(segment)
        },
        numeric(1)
      )
    }

    data.frame(
      feature = feature,
      omics = omics_name,
      break_age = breakpoints,
      n_sig_wins = length(significant_indices),
      stringsAsFactors = FALSE
    )
  })

  breakpoint_df <- bind_rows(breakpoint_list)

  if (nrow(breakpoint_df) == 0) {
    message(
      sprintf("  [%s] No significant breakpoints.", omics_name)
    )
    return(NULL)
  }

  message(
    sprintf(
      "  [%s] %d breakpoints from %d molecules",
      omics_name,
      nrow(breakpoint_df),
      n_distinct(breakpoint_df$feature)
    )
  )

  breakpoint_df
}


#' Count significant molecules in each DE-SWAN sliding window.
#'
#' BH is applied at each window centre and distinct molecules with q below the
#' threshold are counted; this is the statistic shown in the crest panel.
#'
#' @param deswan_result Output of `DEswan()`.
#' @return Number and proportion of significant molecules per age window.
summarize_significant_windows <- function(
    deswan_result,
    fdr_threshold = FDR_THRESHOLD) {
  p_wide <- reshape.DEswan(
    deswan_result,
    parameter = 1,
    factor = "qt"
  )
  q_wide <- q.DEswan(p_wide, method = FDR_METHOD)
  window_ages <- as.numeric(sub("^X", "", colnames(q_wide)[-1]))
  q_matrix <- as.matrix(q_wide[, -1, drop = FALSE])
  storage.mode(q_matrix) <- "numeric"

  data.frame(
    age = window_ages,
    n_significant = colSums(!is.na(q_matrix) & q_matrix < fdr_threshold),
    n_tested = colSums(!is.na(q_matrix)),
    stringsAsFactors = FALSE
  ) %>%
    mutate(
      proportion_significant = if_else(
        n_tested > 0,
        n_significant / n_tested,
        NA_real_
      )
    ) %>%
    arrange(age)
}




detect_configured_peaks <- function(counts) {
  detect_crests(
    counts$age, ifelse(counts$n_tested > 0, counts$n_significant, NA_real_),
    min_peak_relative_height = PEAK_MIN_RELATIVE_HEIGHT,
    min_peak_distance = PEAK_MIN_DISTANCE, max_valley_ratio = MAX_VALLEY_RATIO,
    min_prominence_fraction = PEAK_MIN_PROMINENCE_FRACTION,
    min_relative_prominence = PEAK_MIN_RELATIVE_PROMINENCE,
    width_relative_height = PEAK_WIDTH_RELATIVE_HEIGHT
  )
}

# Monotone dynamic programming: maximum number of one-to-one matches, then
# minimum total age displacement. A target peak cannot support two reference peaks.
match_peak_ages <- function(reference_ages, target_ages, tolerance = 2) {
  nr <- length(reference_ages); nt <- length(target_ages)
  result <- rep(NA_integer_, nr)
  if (!nr || !nt) return(result)
  ro <- order(reference_ages); to <- order(target_ages)
  a <- reference_ages[ro]; b <- target_ages[to]
  score <- matrix(0L, nr + 1L, nt + 1L)
  cost <- matrix(0, nr + 1L, nt + 1L)
  choice <- matrix(0L, nr, nt)
  for (i in rev(seq_len(nr))) for (j in rev(seq_len(nt))) {
    d <- abs(a[i] - b[j])
    values <- c(score[i + 1L, j], score[i, j + 1L],
                if (d <= tolerance) score[i + 1L, j + 1L] + 1L else -1L)
    costs <- c(cost[i + 1L, j], cost[i, j + 1L], cost[i + 1L, j + 1L] + d)
    options <- which(values == max(values))
    best <- options[which.min(costs[options])]
    choice[i, j] <- best; score[i, j] <- values[best]; cost[i, j] <- costs[best]
  }
  i <- 1L; j <- 1L
  while (i <= nr && j <= nt) {
    k <- choice[i, j]
    if (k == 3L) { result[ro[i]] <- to[j]; i <- i + 1L; j <- j + 1L
    } else if (k == 1L) i <- i + 1L else j <- j + 1L
  }
  result
}

deswan_q_matrix <- function(result) {
  required <- c("variable", "window.center", "factor", "pvalue")
  if (!is.list(result) || !is.data.frame(result$p) || !all(required %in% names(result$p))) {
    stop("Invalid DEswan RDS: missing p-value table or required columns.")
  }
  p <- result$p[result$p$factor == "qt", , drop = FALSE]
  if (!nrow(p) || anyDuplicated(p[, c("variable", "window.center")])) {
    stop("DEswan RDS must contain unique feature/age qt p-values.")
  }
  if (any(!is.na(p$pvalue) & (!is.finite(p$pvalue) | p$pvalue < 0 | p$pvalue > 1))) {
    stop("Invalid qt p-value outside [0, 1].")
  }
  wide <- q.DEswan(reshape.DEswan(result, parameter = 1, factor = "qt"), method = FDR_METHOD)
  ages <- as.numeric(sub("^X", "", names(wide)[-1L]))
  if (any(!is.finite(ages)) || anyDuplicated(ages)) stop("Invalid window ages in RDS.")
  ord <- order(ages)
  values <- as.matrix(wide[, -1L, drop = FALSE])[, ord, drop = FALSE]
  storage.mode(values) <- "numeric"
  list(ages = ages[ord], values = values, features = as.character(wide$variable))
}

counts_from_q_matrix <- function(q_matrix, cutoff) {
  tested <- colSums(!is.na(q_matrix$values))
  significant <- colSums(!is.na(q_matrix$values) & q_matrix$values < cutoff)
  data.frame(age = q_matrix$ages, n_significant = unname(significant),
             n_tested = unname(tested), proportion_significant =
               ifelse(tested > 0, significant / tested, NA_real_))
}

full_window_at <- function(age, window) age - window >= AGE_MIN & age + window <= AGE_MAX

assess_parameter_stability <- function(reference_peaks, reference_rds, reference_q) {
  windows <- sort(unique(c(STABILITY_WINDOWS, BUCKET_SIZE)))
  cutoffs <- sort(unique(c(STABILITY_Q_CUTOFFS, FDR_THRESHOLD)))
  scenarios <- list(); scenario_peaks <- list(); scenario_curves <- list(); evaluations <- list()
  source_hash <- unname(tools::md5sum(reference_rds))
  for (w in windows) {
    tag <- paste0("window_", gsub("\\.", "p", format(w, trim = TRUE, scientific = FALSE)))
    path <- if (w == BUCKET_SIZE) reference_rds else resolve_deswan_rds(
      file.path(DESWAN_RESULTS_ROOT, paste0("deswan-window_", w)),
      paste0(omics_name, "_deswan_breakpoint_", tag, ".rds")
    )
    status <- "available"; detail <- ""
    qmat <- tryCatch({
      if (!file.exists(path)) stop("RDS missing")
      ans <- if (w == BUCKET_SIZE) reference_q else deswan_q_matrix(readRDS(path))
      if (!setequal(ans$features, reference_q$features)) stop("Feature universe differs from reference RDS")
      ans
    }, error = function(e) { status <<- "unavailable"; detail <<- conditionMessage(e); NULL })
    checksum <- if (file.exists(path)) unname(tools::md5sum(path)) else NA_character_
    for (q in cutoffs) {
      id <- length(scenarios) + 1L
      counts <- if (!is.null(qmat)) counts_from_q_matrix(qmat, q) else NULL
      detected <- if (!is.null(counts)) detect_configured_peaks(counts) else NULL
      peaks <- if (!is.null(detected)) detected$crests else reference_peaks[FALSE, ]
      scenarios[[id]] <- data.frame(
        omics = omics_name, window = w, q_cutoff = q, status = status,
        detail = detail, source_rds = normalizePath(path, winslash = "/", mustWork = FALSE),
        source_md5 = checksum, n_features = if (is.null(qmat)) NA_integer_ else length(qmat$features),
        n_peaks = if (is.null(qmat)) NA_integer_ else nrow(peaks),
        core_q = any(abs(q - STABILITY_CORE_Q) < 1e-12),
        reference_scenario = w == BUCKET_SIZE && abs(q - FDR_THRESHOLD) < 1e-12
      )
      key <- scenarios[[id]][, c("omics", "window", "q_cutoff")]
      if (!is.null(counts)) {
        scenario_curves[[id]] <- cbind(key[rep(1L, nrow(counts)), ], counts,
          n_raw = detected$detection_curve$n_raw)
      }
      if (nrow(peaks)) {
        scenario_peaks[[id]] <- cbind(key[rep(1L, nrow(peaks)), ], peaks,
                                     complete_window = full_window_at(peaks$peak_age, w))
      }
      if (!nrow(reference_peaks)) next
      in_span <- full_window_at(reference_peaks$peak_age, w)
      evaluable <- rep(FALSE, nrow(reference_peaks))
      if (!is.null(counts)) {
        for (r in seq_len(nrow(reference_peaks))) {
          centre <- match(reference_peaks$peak_age[r], counts$age)
          evaluable[r] <- in_span[r] && !is.na(centre) && centre > 1L && centre < nrow(counts) &&
            all(counts$n_tested[(centre - 1L):(centre + 1L)] > 0)
        }
      }
      targets <- which(full_window_at(peaks$peak_age, w))
      ref_rows <- which(evaluable)
      matches <- rep(NA_integer_, nrow(reference_peaks))
      if (length(ref_rows) && length(targets)) {
        matched <- match_peak_ages(reference_peaks$peak_age[ref_rows],
                                   peaks$peak_age[targets], STABILITY_MATCH_TOLERANCE)
        matches[ref_rows] <- targets[matched]
      }
      evaluations[[id]] <- data.frame(
        crest_id = reference_peaks$crest_id, reference_age = reference_peaks$peak_age,
        window = w, q_cutoff = q, core_q = scenarios[[id]]$core_q,
        reference_scenario = scenarios[[id]]$reference_scenario,
        complete_span = in_span, source_available = status == "available",
        evaluable = evaluable, matched = evaluable & !is.na(matches),
        matched_age = peaks$peak_age[matches],
        age_displacement = abs(peaks$peak_age[matches] - reference_peaks$peak_age),
        exclusion_reason = ifelse(!in_span, "incomplete_age_span",
          ifelse(status != "available", detail,
                 ifelse(!evaluable, "untested_reference_neighbourhood", "")))
      )
    }
  }
  scenarios <- bind_rows(scenarios)
  evaluations <- bind_rows(evaluations)
  scored <- reference_peaks
  scored$complete_window <- full_window_at(scored$peak_age, BUCKET_SIZE)
  scored$main_prominence_pass <- scored$relative_prominence >= MAIN_MIN_RELATIVE_PROMINENCE
  scored$main_width_pass <- scored$peak_width_years >= MAIN_MIN_WIDTH_YEARS
  scored$main_shape_pass <- scored$main_prominence_pass & scored$main_width_pass
  scored$main_exclusion_reason <- vapply(seq_len(nrow(scored)), function(i) paste(
    c(if (!scored$main_prominence_pass[i]) "below_main_prominence",
      if (!scored$main_width_pass[i]) "below_main_width"), collapse = ";"), character(1))
  scored$n_other_evaluable <- integer(nrow(scored))
  scored$n_other_supported <- integer(nrow(scored))
  scored$parameter_support <- rep(NA_real_, nrow(scored))
  scored$n_supporting_windows <- integer(nrow(scored))
  scored$n_supporting_q <- integer(nrow(scored))
  scored$parameter_grid_complete <- rep(FALSE, nrow(scored))
  # A representative peak describes the current curve. Parameter evidence and
  # incomplete age coverage qualify its interpretation, not its visibility.
  scored$is_main <- scored$main_shape_pass
  scored$peak_class <- rep("minor_candidate", nrow(scored))
  scored$peak_class[scored$is_main] <- "main"
  scored$boundary_limited <- !scored$complete_window
  scored$stability_status <- rep("insufficient_parameter_evidence", nrow(scored))
  scored$is_stable_main <- rep(FALSE, nrow(scored))
  for (r in seq_len(nrow(scored))) {
    e <- evaluations[evaluations$crest_id == scored$crest_id[r], ]
    other <- e[e$core_q & e$evaluable & !e$reference_scenario, ]
    supported <- e[e$core_q & e$evaluable & e$matched, ]
    scored$n_other_evaluable[r] <- nrow(other)
    scored$n_other_supported[r] <- sum(other$matched)
    scored$parameter_support[r] <- if (nrow(other)) mean(other$matched) else NA_real_
    scored$n_supporting_windows[r] <- length(unique(supported$window))
    scored$n_supporting_q[r] <- length(unique(supported$q_cutoff))
    scored$parameter_grid_complete[r] <- !any(e$core_q & e$complete_span & !e$source_available)
    scored$stability_status[r] <- if (!scored$complete_window[r]) "boundary_limited" else
      if (!scored$parameter_grid_complete[r] || !nrow(other)) "insufficient_parameter_evidence" else
      if (scored$parameter_support[r] < MAIN_MIN_PARAMETER_SUPPORT ||
          scored$n_supporting_windows[r] < MAIN_MIN_SUPPORTING_WINDOWS ||
          scored$n_supporting_q[r] < MAIN_MIN_SUPPORTING_Q) "parameter_sensitive" else "supported"
    scored$is_stable_main[r] <- scored$is_main[r] && scored$stability_status[r] == "supported"
  }
  # Always export typed tables, including zero-reference-peak runs.
  if (!nrow(evaluations)) evaluations <- data.frame(
    crest_id = integer(), reference_age = numeric(), window = numeric(), q_cutoff = numeric(),
    core_q = logical(), reference_scenario = logical(), complete_span = logical(),
    source_available = logical(), evaluable = logical(), matched = logical(),
    matched_age = numeric(), age_displacement = numeric(), exclusion_reason = character())
  all_peaks <- bind_rows(scenario_peaks)
  if (!nrow(all_peaks)) all_peaks <- cbind(data.frame(omics = character(), window = numeric(),
    q_cutoff = numeric()), reference_peaks[FALSE, ], complete_window = logical())
  list(crests = scored, scenarios = scenarios, matches = evaluations,
       peaks = all_peaks, curves = bind_rows(scenario_curves), source_md5 = source_hash)
}


# Descriptive feature/window summaries; overlapping windows are not replicates.
summarize_figure_statistics <- function(q_matrix, window_summary, crest_df) {
  tested_features <- rowSums(is.finite(q_matrix$values)) > 0L
  ever_significant <- rowSums(is.finite(q_matrix$values) &
                               q_matrix$values < FDR_THRESHOLD) > 0L
  valid <- window_summary$n_tested > 0 & is.finite(window_summary$n_significant)
  observed <- window_summary$n_significant[valid]
  quartiles <- if (length(observed)) as.numeric(stats::quantile(
    observed, c(0.25, 0.5, 0.75), names = FALSE, type = 7)) else rep(NA_real_, 3)
  n_tested <- sum(tested_features)
  data.frame(
    omics = omics_label, window_half_width_years = BUCKET_SIZE,
    window_total_span_years = 2 * BUCKET_SIZE, q_cutoff = FDR_THRESHOLD,
    fdr_method = FDR_METHOD, n_features_in_rds = length(tested_features),
    n_features_tested_any_window = n_tested,
    n_features_never_tested = sum(!tested_features),
    n_features_significant_any_window = sum(ever_significant),
    proportion_features_significant_any_window = if (n_tested > 0L)
      sum(ever_significant) / n_tested else NA_real_,
    n_age_windows = nrow(window_summary), n_tested_windows = sum(valid),
    n_untested_windows = sum(!valid),
    n_complete_tested_windows = sum(valid & full_window_at(window_summary$age, BUCKET_SIZE)),
    window_count_q25 = quartiles[1], window_count_median = quartiles[2],
    window_count_q75 = quartiles[3],
    window_count_max = if (length(observed)) max(observed) else NA_real_,
    n_main_peaks = sum(crest_df$is_main %in% TRUE),
    n_stable_main_peaks = sum(crest_df$is_stable_main %in% TRUE),
    n_boundary_main_peaks = sum(crest_df$is_main %in% TRUE & crest_df$boundary_limited %in% TRUE),
    summary_unit = "features_and_overlapping_age_windows",
    uncertainty = "not_estimated_no_participant_resampling"
  )
}

# Plot raw observations and their main peaks. Keep display_curve in the calling
# interface for compatibility with existing postprocessing and diagnostic tables.
plot_deswan_results <- function(window_summary, crest_df, all_breakpoints, detection_curve,
                               display_curve = NULL,
                               figure_statistics = NULL) {
  dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)
  raw <- window_summary
  raw$segment_id <- detection_curve$segment_id
  raw$n_significant[raw$n_tested == 0] <- NA_real_
  # Keep candidate diagnostics in the tables; annotate only qualifying main peaks.
  peak_plot <- crest_df[crest_df$is_main %in% TRUE, , drop = FALSE]
  peak_plot <- peak_plot[order(peak_plot$peak_age), , drop = FALSE]
  peak_plot$guide_color <- unname(PEAK_GUIDE_COLORS[
    pmin(seq_len(nrow(peak_plot)), length(PEAK_GUIDE_COLORS))])
  curve_max <- max(c(1, raw$n_significant), na.rm = TRUE)
  count_text <- function(x) ifelse(is.finite(x),
    format(round(x, 1), big.mark = ",", trim = TRUE, scientific = FALSE), "NA")
  percent_text <- function(x) ifelse(is.finite(x), sprintf("%.1f%%", 100 * x), "NA")
  label_offset <- min(max(PEAK_LABEL_OFFSET, 0.09 * curve_max), 0.16 * curve_max)
  peak_plot$n_tested_at_peak <- raw$n_tested[match(peak_plot$peak_age, raw$age)]
  peak_plot$proportion_at_peak <- ifelse(peak_plot$n_tested_at_peak > 0,
    peak_plot$peak_n / peak_plot$n_tested_at_peak, NA_real_)
  peak_plot$label_y <- peak_plot$peak_n + label_offset
  peak_plot$label <- if (nrow(peak_plot)) paste0(format(peak_plot$peak_age, trim = TRUE),
    " y\n",
    count_text(peak_plot$peak_n), "\n(", percent_text(peak_plot$proportion_at_peak), ")") else character()
  x_range <- range(c(AGE_MIN, AGE_MAX, raw$age), finite = TRUE)
  x_span <- diff(x_range)
  # Keep the testing denominator and multiplicity-adjusted cutoff in the legend.
  tested_label <- if (!is.null(figure_statistics)) paste0(
    count_text(figure_statistics$n_features_tested_any_window), " features tested") else {
      tested_range <- range(raw$n_tested)
      paste0(paste(unique(count_text(tested_range)), collapse = "\u2013"), " tested per window")
    }
  midpoint <- mean(x_range)
  # Place labels beside the full-height guides, facing into the panel.
  peak_plot$label_hjust <- ifelse(peak_plot$peak_age <= midpoint, 0, 1)
  peak_plot$label_x <- peak_plot$peak_age +
    ifelse(peak_plot$label_hjust == 0, 1, -1) * 0.032 * x_span
  y_max <- max(c(curve_max, peak_plot$label_y), na.rm = TRUE) * Y_TOP_EXPANSION
  publication_theme <- theme_classic(base_size = 7, base_family = PLOT_FONT_FAMILY) +
    theme(axis.title = element_text(size = 7, colour = "black"),
          axis.text = element_text(size = 6.5, colour = "black"),
          axis.line = element_line(linewidth = 0.28, colour = "black"),
          axis.ticks = element_line(linewidth = 0.25, colour = "black"),
          axis.ticks.length = grid::unit(1.0, "mm"),
          legend.position = "top", legend.justification = "left",
          legend.text = element_text(size = 6.5, colour = "black"),
          legend.key.width = grid::unit(6, "mm"), legend.key.height = grid::unit(3, "mm"),
          legend.margin = margin(0, 0, 0, 0, unit = "mm"),
          legend.box.spacing = grid::unit(1, "mm"),
          plot.margin = margin(2, 2, 2, 2, unit = "mm"))
  compact_theme <- publication_theme + theme(
    axis.title = element_text(size = 5.5),
    axis.title.x = element_text(margin = margin(t = 1.5)),
    axis.title.y = element_text(margin = margin(r = 1.5)),
    axis.text = element_text(size = 5.2),
    axis.text.x = element_text(margin = margin(t = 1)),
    axis.text.y = element_text(margin = margin(r = 1)),
    axis.line = element_line(linewidth = 0.20),
    axis.ticks = element_line(linewidth = 0.18),
    axis.ticks.length = grid::unit(0.65, "mm"),
    plot.margin = margin(1, 1, 1, 1, unit = "mm"))
  p_main <- ggplot(raw, aes(age, n_significant)) +
    geom_vline(data = peak_plot, aes(xintercept = peak_age, colour = guide_color),
               linewidth = PEAK_GUIDE_LINE_WIDTH, linetype = "solid", show.legend = FALSE) +
    scale_colour_identity() +
    geom_line(aes(group = segment_id), colour = omics_color,
              linewidth = RAW_LINE_WIDTH, lineend = "round", na.rm = TRUE) +
    geom_point(colour = omics_color, shape = 16, size = RAW_POINT_SIZE, na.rm = TRUE) +
    geom_point(data = peak_plot, aes(peak_age, peak_n), inherit.aes = FALSE,
               shape = 21, colour = "white", fill = omics_color, size = 1.15, stroke = 0.18) +
    geom_text(data = peak_plot, aes(label_x, label_y, label = label, hjust = label_hjust),
              inherit.aes = FALSE, family = PLOT_FONT_FAMILY,
              size = 1.85, lineheight = 1.0, vjust = 0, colour = "black") +
    scale_x_continuous(breaks = seq(floor(x_range[1] / 10) * 10,
                                   ceiling(x_range[2] / 10) * 10, by = 10),
                       limits = x_range, expand = expansion(mult = 0)) +
    scale_y_continuous(breaks = scales::pretty_breaks(n = 3), labels = scales::label_comma(),
                       limits = c(0, y_max),
                       expand = expansion(mult = c(0.025, 0))) +
    labs(x = "Age (years)", y = y_axis_label) +
    compact_theme
  all_breakpoints$omics <- rep(omics_label, nrow(all_breakpoints))
  if (nrow(all_breakpoints) < 2L) {
    p_ridge <- ggplot() +
      annotate("text", x = mean(c(AGE_MIN, AGE_MAX)), y = 0,
                label = "Insufficient breakpoints\nfor density estimation",
                family = PLOT_FONT_FAMILY, size = 2.1) +
      scale_x_continuous(limits = c(AGE_MIN, AGE_MAX)) +
      labs(x = "Age (years)", y = NULL) + publication_theme +
      theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
  } else {
    density_estimate <- stats::density(all_breakpoints$break_age, bw = 2,
                                       from = x_range[1], to = x_range[2], n = 512)
    density_curve <- data.frame(age = density_estimate$x, density = density_estimate$y)
    p_ridge <- ggplot(density_curve, aes(age, density)) +
      geom_line(colour = omics_color, linewidth = 0.5) +
      geom_vline(data = peak_plot, aes(xintercept = peak_age, colour = guide_color),
                  linewidth = PEAK_GUIDE_LINE_WIDTH, linetype = "solid", show.legend = FALSE) +
      scale_colour_identity() +
      scale_x_continuous(breaks = seq(AGE_MIN, AGE_MAX, by = 5),
                         expand = expansion(mult = 0)) +
      coord_cartesian(xlim = x_range) +
      scale_y_continuous(breaks = scales::pretty_breaks(n = 3),
                         expand = expansion(mult = c(0, 0.08))) +
      labs(x = "Age (years)", y = "Breakpoint density") + publication_theme
  }
  export_plot <- function(plot, suffix, width, height) {
    stem <- file.path(FIGURE_DIR, paste0(omics_name, suffix, "_", WINDOW_FILE_TAG))
    save_vector <- function(extension) {
      if (extension == "pdf") {
        # Cairo embeds the Arial TrueType subsets while preserving editable text.
        grDevices::cairo_pdf(paste0(stem, ".pdf"), width = width, height = height,
                             bg = "white", family = PLOT_FONT_FAMILY)
      } else {
        svglite::svglite(paste0(stem, ".svg"), width = width, height = height, bg = "white")
      }
      on.exit(grDevices::dev.off(), add = TRUE)
      print(plot)
    }
    save_vector("pdf")
    save_vector("svg")
    # Some raster devices only warn when writing fails, leaving a stale PNG.
    # Render a new file first, then verify the bytes copied to the final path.
    png_temp <- tempfile("deswan-png-", tmpdir = FIGURE_DIR, fileext = ".png")
    on.exit(unlink(png_temp), add = TRUE)
    withCallingHandlers(
      ggsave(png_temp, plot, width = width, height = height, dpi = 600,
             device = ragg::agg_png, bg = "white"),
      warning = function(w) stop("PNG rendering failed: ", conditionMessage(w), call. = FALSE)
    )
    if (!file.exists(png_temp) || is.na(file.info(png_temp)$size) || file.info(png_temp)$size <= 0) {
      stop("PNG rendering produced no valid file: ", png_temp)
    }
    expected_hash <- unname(tools::md5sum(png_temp))
    png_path <- paste0(stem, ".png")
    if (!isTRUE(suppressWarnings(file.copy(png_temp, png_path, overwrite = TRUE)))) {
      png_path <- tempfile(paste0(basename(stem), "_", PEAK_METHOD_VERSION, "_"),
                           tmpdir = FIGURE_DIR, fileext = ".png")
      if (!file.rename(png_temp, png_path)) stop("Cannot save rendered PNG: ", png_path)
      warning("Could not replace ", stem, ".png; updated preview saved to ", png_path,
              call. = FALSE)
    }
    if (!identical(expected_hash, unname(tools::md5sum(png_path)))) {
      stop("PNG export verification failed: ", png_path)
    }
    message("Verified PNG: ", normalizePath(png_path, winslash = "/"))
  }
  export_plot(p_main, "_breakpoint_density", MAIN_W, MAIN_H)
  export_plot(p_ridge, "_ridge", RIDGE_W, RIDGE_H)
  # Keep methodological detail in the manuscript legend rather than on the artwork.
  peak_guide_colour_legend <- if (nrow(peak_plot)) paste0(
    "In increasing age order, the first peak guide is blue",
    if (nrow(peak_plot) > 1L) ", the second is red" else "",
    if (nrow(peak_plot) > 2L) ", and subsequent guides are grey" else "",
    ". ") else ""
  main_legend <- paste0(omics_label, " age-window profile. ",
    "Coloured points and straight connecting segments show raw counts of features with ",
    "Benjamini-Hochberg-adjusted q < ", FDR_THRESHOLD,
    " (correction across features within each age window). ", tested_label, "; ",
    sum(raw$n_tested > 0), " tested age centres. Each centre defines two adjacent ",
    BUCKET_SIZE, "-year age parcels (", 2 * BUCKET_SIZE, " years in total). ",
    "Solid vertical lines span the panel at main peaks selected from the raw counts. ",
    peak_guide_colour_legend,
    "Peak labels give age, raw count and the percentage of features tested at that age. ",
    if (any(peak_plot$boundary_limited)) paste0(
      "Age windows centred at ",
      paste(format(peak_plot$peak_age[peak_plot$boundary_limited], trim = TRUE), collapse = ", "),
      " years extend beyond the ", AGE_MIN, "-", AGE_MAX, "-year sampling range. ") else "",
    "Age windows overlap and are not independent replicates. ",
    "Source data accompany the window-count and peak tables.")
  writeLines(main_legend, file.path(FIGURE_DIR, windowed_output_filename(
    paste0(omics_name, "_breakpoint_density_legend"), "txt")), useBytes = TRUE)
  ridge_legend <- if (nrow(all_breakpoints) < 2L) paste0(omics_label,
    " breakpoint ages. Fewer than two breakpoint regions were detected; a density was not estimated.") else
    paste0(omics_label, " breakpoint-age distribution. ",
    "The unfilled curve shows the Gaussian kernel density of ", nrow(all_breakpoints),
    " breakpoint regions from ", length(unique(all_breakpoints$feature)),
    " unique features (bandwidth 2 years). A feature can contribute multiple regions. ",
    "Solid vertical lines indicate main peaks detected from the raw significant-feature counts. ",
    peak_guide_colour_legend)
  writeLines(ridge_legend,
    file.path(FIGURE_DIR, windowed_output_filename(paste0(omics_name, "_ridge_legend"), "txt")),
    useBytes = TRUE)
  invisible(list(main = p_main, ridge = p_ridge))
}

run_deswan_postprocessing <- function(make_plots = TRUE) {
  if (any(!is.finite(STABILITY_WINDOWS) | STABILITY_WINDOWS <= 0) ||
      any(!is.finite(STABILITY_Q_CUTOFFS) | STABILITY_Q_CUTOFFS <= 0 | STABILITY_Q_CUTOFFS > 1) ||
      !length(STABILITY_WINDOWS) || !length(STABILITY_CORE_Q) ||
      !all(STABILITY_CORE_Q %in% STABILITY_Q_CUTOFFS) ||
      !is.finite(STABILITY_MATCH_TOLERANCE) || STABILITY_MATCH_TOLERANCE < 0 ||
      !is.finite(MAIN_MIN_PARAMETER_SUPPORT) || MAIN_MIN_PARAMETER_SUPPORT < 0 || MAIN_MIN_PARAMETER_SUPPORT > 1 ||
      !is.finite(MAIN_MIN_RELATIVE_PROMINENCE) || MAIN_MIN_RELATIVE_PROMINENCE < 0 || MAIN_MIN_RELATIVE_PROMINENCE > 1 ||
      length(MAIN_MIN_WIDTH_YEARS) != 1L || !is.finite(MAIN_MIN_WIDTH_YEARS) || MAIN_MIN_WIDTH_YEARS < 0 ||
      any(c(MAIN_MIN_SUPPORTING_WINDOWS, MAIN_MIN_SUPPORTING_Q) < 1)) {
    stop("Invalid parameter-stability/main-peak configuration; core q cutoffs must be in the comparison grid.")
  }
  rds_path <- if (nzchar(DESWAN_RDS_PATH)) DESWAN_RDS_PATH else resolve_deswan_rds(
    OUTPUT_DIR, windowed_output_filename(paste0(omics_name, "_deswan_breakpoint"), "rds"))
  if (!file.exists(rds_path)) {
    stop("DEswan RDS not found: ", rds_path,
         "\nRun 01-de_swan_omics_one.R for this omics/window first, or set DESWAN_RDS_PATH.")
  }
  message("Loading DEswan RDS (no model fitting): ", rds_path)
  deswan_results <- readRDS(rds_path)
  q_matrix <- deswan_q_matrix(deswan_results)
  window_summary <- counts_from_q_matrix(q_matrix, FDR_THRESHOLD)
  all_breakpoints <- extract_breakpoints(deswan_results, omics_name)
  if (is.null(all_breakpoints)) all_breakpoints <- data.frame(
    feature = character(), omics = character(), break_age = numeric(), n_sig_wins = integer())
  crest_result <- detect_configured_peaks(window_summary)
  display_curve <- build_display_curve(crest_result$detection_curve)
  stability <- assess_parameter_stability(crest_result$crests, rds_path, q_matrix)
  crest_df <- stability$crests
  figure_statistics <- summarize_figure_statistics(q_matrix, window_summary, crest_df)
  main_peaks <- crest_df[crest_df$is_main, , drop = FALSE]
  stable_main_peaks <- crest_df[crest_df$is_stable_main, , drop = FALSE]
  has_main <- nrow(main_peaks) > 0L
  primary <- if (has_main) order(-main_peaks$prominence, -main_peaks$peak_n, main_peaks$peak_age)[1L] else NA_integer_
  summary_table <- data.frame(
    omics = omics_label, window = BUCKET_SIZE, q_cutoff = FDR_THRESHOLD,
    n_molecules = n_distinct(all_breakpoints$feature), n_breakpoints = nrow(all_breakpoints),
    median_age = if (nrow(all_breakpoints)) round(median(all_breakpoints$break_age), 1) else NA_real_,
    n_crests = nrow(crest_df), n_main_peaks = nrow(main_peaks),
    n_stable_main_peaks = nrow(stable_main_peaks),
    primary_crest_age = if (has_main) main_peaks$peak_age[primary] else NA_real_,
    primary_crest_n = if (has_main) main_peaks$peak_n[primary] else NA_real_,
    primary_crest_raw_n = if (has_main) main_peaks$raw_n_at_peak[primary] else NA_real_,
    crest_peak_ages = paste(crest_df$peak_age, collapse = ";"),
    main_peak_ages = paste(main_peaks$peak_age, collapse = ";"),
    stable_main_peak_ages = paste(stable_main_peaks$peak_age, collapse = ";"),
    method_version = PEAK_METHOD_VERSION, peak_basis = "raw_significant_counts",
    evidence_type = "raw_count_shape_and_parameter_sensitivity"
  )
  settings <- data.frame(
    method_version = PEAK_METHOD_VERSION, source_rds = normalizePath(rds_path, winslash = "/"),
    source_md5 = stability$source_md5, omics = omics_name, window = BUCKET_SIZE,
    q_cutoff = FDR_THRESHOLD, fdr_method = FDR_METHOD,
    peak_basis = "raw_significant_counts", smoothing_used_for_detection = FALSE,
    display_smoothing_method = "normalized_gaussian_kernel",
    display_smoothing_bandwidth_years = DISPLAY_SMOOTHING_BANDWIDTH,
    display_curve_points_per_segment = DISPLAY_CURVE_POINTS,
    min_height_fraction = PEAK_MIN_RELATIVE_HEIGHT, min_peak_distance = PEAK_MIN_DISTANCE,
    max_valley_ratio = MAX_VALLEY_RATIO, min_prominence_fraction = PEAK_MIN_PROMINENCE_FRACTION,
    min_relative_prominence = PEAK_MIN_RELATIVE_PROMINENCE,
    main_min_relative_prominence = MAIN_MIN_RELATIVE_PROMINENCE,
    valley_grouping_before_distance = TRUE,
    peak_width_relative_height = PEAK_WIDTH_RELATIVE_HEIGHT,
    main_min_width_years = MAIN_MIN_WIDTH_YEARS,
    main_prominence_denominator = "own_peak_height",
    comparison_windows = paste(STABILITY_WINDOWS, collapse = ";"),
    comparison_q = paste(STABILITY_Q_CUTOFFS, collapse = ";"),
    main_core_q = paste(STABILITY_CORE_Q, collapse = ";"),
    match_tolerance_years = STABILITY_MATCH_TOLERANCE,
    min_parameter_support = MAIN_MIN_PARAMETER_SUPPORT,
    min_supporting_windows = MAIN_MIN_SUPPORTING_WINDOWS, min_supporting_q = MAIN_MIN_SUPPORTING_Q,
    selected_window_requires_complete_span = FALSE, main_requires_parameter_support = FALSE,
    stable_main_requires_complete_span = TRUE, stable_main_requires_parameter_support = TRUE,
    exclude_reference_from_support = TRUE,
    r_version = R.version.string, deswan_version = as.character(packageVersion("DEswan"))
  )
  tables <- list(
    window_counts = window_summary, breakpoints = all_breakpoints, crest = crest_df,
    main_peaks = main_peaks, stable_main_peaks = stable_main_peaks,
    peak_candidates = crest_result$candidates,
    peak_separations = crest_result$separations, peak_detection_curve = crest_result$detection_curve,
    peak_display_curve = display_curve,
    peak_stability_matches = stability$matches, peak_stability_scenarios = stability$scenarios,
    peak_stability_peaks = stability$peaks, peak_stability_curves = stability$curves,
    peak_settings = settings, summary = summary_table, figure_statistics = figure_statistics
  )
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  for (name in names(tables)) {
    write.csv(tables[[name]], file.path(OUTPUT_DIR, windowed_output_filename(
      paste0(omics_name, "_", name), "csv")), row.names = FALSE)
  }
  message("Independent candidate peaks: ", paste(crest_df$peak_age, collapse = ", "),
          "; representative main peaks: ", if (has_main) paste(main_peaks$peak_age, collapse = ", ") else "none",
          "; stable main peaks: ", if (nrow(stable_main_peaks)) paste(stable_main_peaks$peak_age, collapse = ", ") else "none")
  print(summary_table)
  if (make_plots) plot_deswan_results(window_summary, crest_df, all_breakpoints,
                                     crest_result$detection_curve, display_curve, figure_statistics)
  message("Postprocessing complete; source RDS unchanged. Representative peaks and parameter evidence are reported separately.")
  invisible(tables)
}

postprocessing_results <- run_deswan_postprocessing(
  make_plots = !("--data-only" %in% commandArgs(trailingOnly = TRUE))
)
