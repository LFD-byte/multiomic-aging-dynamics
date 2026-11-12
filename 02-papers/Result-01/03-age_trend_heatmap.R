# ============================================================================
# Age-trend heatmap
# Pipeline: read Python outputs -> smooth each feature -> cluster rows -> plot
# Output: 80 x 90 mm panel, Nature-journal font sizes (5-7 pt, sans-serif)
#
# Required package (installed once):
#   install.packages("gtable")
# Optional package for higher-quality PNG output:
#   install.packages("ragg")
# ============================================================================

set.seed(42)

# ----------------------------------------------------------------------------
# 1. Paths and configuration
# ----------------------------------------------------------------------------

find_script_file <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) > 0L) {
    return(sub("^--file=", "", file_arg[[1L]]))
  }

  frames <- sys.frames()
  source_files <- vapply(
    frames,
    function(frame) {
      value <- frame$ofile
      if (is.null(value) || length(value) == 0L) NA_character_ else as.character(value[[1L]])
    },
    character(1)
  )
  source_files <- source_files[!is.na(source_files)]
  if (length(source_files) > 0L) tail(source_files, 1L) else NA_character_
}

find_analysis_dir <- function() {
  script_file <- find_script_file()
  candidates <- c(
    if (!is.na(script_file)) dirname(script_file),
    getwd(),
    file.path(getwd(), "02-papers", "Result-01")
  )
  candidates <- unique(normalizePath(candidates, winslash = "/", mustWork = FALSE))

  required <- file.path(
    "outputs",
    c("age_trend_matrix_dys.csv", "age_trend_row_info_dys.csv")
  )
  found <- vapply(
    candidates,
    function(path) all(file.exists(file.path(path, required))),
    logical(1)
  )

  if (!any(found)) {
    stop(
      paste0(
        "Could not locate the two input CSV files. Searched:\n  ",
        paste(candidates, collapse = "\n  ")
      ),
      call. = FALSE
    )
  }
  candidates[which(found)[[1L]]]
}

if (!requireNamespace("gtable", quietly = TRUE)) {
  stop(
    "Missing package 'gtable'. Install it with install.packages('gtable'), then rerun.",
    call. = FALSE
  )
}

ANALYSIS_DIR <- find_analysis_dir()
INPUT_PREFIX <- file.path(ANALYSIS_DIR, "outputs", "age_trend")
OUTPUT_PREFIX <- file.path(ANALYSIS_DIR, "outputs", "age_trend")  # result tables
FIGURE_PREFIX <- file.path(ANALYSIS_DIR, "figures", "age_trend")  # figures
dir.create(dirname(OUTPUT_PREFIX), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(FIGURE_PREFIX), recursive = TRUE, showWarnings = FALSE)

AGE_MIN <- 40
AGE_MAX <- 70
N_SMOOTH <- 100
SPAR <- 0.5

OMICS_COLORS <- c(
  "Transcriptomics" = "#CB6E70",
  "Metabolomics"    = "#68AADD",
  "Lipidomics"      = "#54AD94",
  "Gut Microbiome"  = "#DDA062",
  "Oral Microbiome" = "#896EAE"
)

cat(sprintf("Analysis directory: %s\n", ANALYSIS_DIR))

# ----------------------------------------------------------------------------
# 2. Read and validate Python outputs
# ----------------------------------------------------------------------------

cat("Step 1/5: Reading and validating data...\n")

matrix_path <- paste0(INPUT_PREFIX, "_matrix_dys.csv")
row_info_path <- paste0(INPUT_PREFIX, "_row_info_dys.csv")

# encoding="UTF-8" marks strings without converting them through the Windows
# native locale. Using fileEncoding="UTF-8" can truncate this file when the
# current locale cannot represent the en dash in age-bin column names.
matrix_table <- read.csv(
  matrix_path,
  row.names = NULL,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  encoding = "UTF-8"
)
row_info <- read.csv(
  row_info_path,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  encoding = "UTF-8"
)

if (ncol(matrix_table) < 4L) {
  stop("The trend matrix must contain a feature column and at least three age bins.", call. = FALSE)
}
if (!all(c("feature", "omics") %in% names(row_info))) {
  stop("The row-info CSV must contain columns named 'feature' and 'omics'.", call. = FALSE)
}
if (nrow(matrix_table) != nrow(row_info)) {
  stop(
    sprintf(
      "Input row-count mismatch: matrix has %d rows but row info has %d rows.",
      nrow(matrix_table), nrow(row_info)
    ),
    call. = FALSE
  )
}

features <- as.character(matrix_table[[1L]])
if (anyNA(features) || any(!nzchar(features))) {
  stop("The matrix contains missing or empty feature names.", call. = FALSE)
}
if (!identical(features, as.character(row_info$feature))) {
  stop("Feature order differs between the matrix and row-info CSV files.", call. = FALSE)
}

unknown_omics <- setdiff(unique(as.character(row_info$omics)), names(OMICS_COLORS))
if (length(unknown_omics) > 0L || anyNA(row_info$omics)) {
  stop(
    sprintf(
      "Unknown or missing omics label(s): %s",
      paste(c(unknown_omics, if (anyNA(row_info$omics)) "<NA>"), collapse = ", ")
    ),
    call. = FALSE
  )
}

value_table <- matrix_table[-1L]
non_numeric <- names(value_table)[!vapply(value_table, is.numeric, logical(1))]
if (length(non_numeric) > 0L) {
  stop(
    sprintf("Non-numeric age-bin column(s): %s", paste(non_numeric, collapse = ", ")),
    call. = FALSE
  )
}

feature_ids <- make.unique(features, sep = "__duplicate_")
mat_raw <- as.matrix(value_table)
storage.mode(mat_raw) <- "double"
rownames(mat_raw) <- feature_ids

bad_values <- !is.finite(mat_raw)
if (any(bad_values)) {
  cat(sprintf("  Treating %d non-finite value(s) as missing.\n", sum(bad_values)))
  mat_raw[bad_values] <- NA_real_
}

cat(sprintf("  Matrix: %d features x %d age bins\n", nrow(mat_raw), ncol(mat_raw)))
cat("  Omics distribution:\n")
print(table(row_info$omics))

# Extract the age-bin midpoints from labels such as "40-41" or "40–41".
parse_age_midpoints <- function(labels) {
  pieces <- regmatches(
    labels,
    gregexpr("[0-9]+(?:\\.[0-9]+)?", labels, perl = TRUE)
  )
  result <- vapply(
    pieces,
    function(numbers) {
      if (length(numbers) < 2L) return(NA_real_)
      mean(as.numeric(numbers[1:2]))
    },
    numeric(1)
  )
  if (anyNA(result) || is.unsorted(result, strictly = TRUE)) NULL else result
}

age_x <- parse_age_midpoints(colnames(mat_raw))
if (is.null(age_x)) {
  warning("Could not parse age-bin labels; using evenly spaced bin midpoints.", call. = FALSE)
  age_breaks <- seq(AGE_MIN, AGE_MAX, length.out = ncol(mat_raw) + 1L)
  age_x <- (head(age_breaks, -1L) + tail(age_breaks, -1L)) / 2
}
age_fine <- seq(AGE_MIN, AGE_MAX, length.out = N_SMOOTH)

# ----------------------------------------------------------------------------
# 3. Smooth every feature independently
# ----------------------------------------------------------------------------

cat("\nStep 2/5: Smoothing feature trajectories...\n")

smooth_rows <- function(mat, x, xout, spar) {
  result <- matrix(
    NA_real_,
    nrow = nrow(mat),
    ncol = length(xout),
    dimnames = list(rownames(mat), sprintf("%.2f", xout))
  )
  errors <- character(nrow(mat))

  for (i in seq_len(nrow(mat))) {
    y <- as.numeric(mat[i, ])
    keep <- is.finite(x) & is.finite(y)
    if (sum(keep) < 3L) {
      errors[[i]] <- "fewer than three finite age bins"
      next
    }

    fit <- tryCatch(
      stats::smooth.spline(x[keep], y[keep], spar = spar),
      error = function(error) error
    )
    if (inherits(fit, "error")) {
      errors[[i]] <- conditionMessage(fit)
      next
    }

    prediction <- tryCatch(
      stats::predict(fit, xout)$y,
      error = function(error) error
    )
    if (inherits(prediction, "error") || any(!is.finite(prediction))) {
      errors[[i]] <- if (inherits(prediction, "error")) {
        conditionMessage(prediction)
      } else {
        "non-finite spline prediction"
      }
      next
    }
    result[i, ] <- prediction
  }

  list(matrix = result, errors = errors)
}

smoothed <- smooth_rows(mat_raw, age_x, age_fine, SPAR)
mat_smooth <- smoothed$matrix
smooth_failed <- nzchar(smoothed$errors)
if (any(smooth_failed)) {
  cat(sprintf("  Dropping %d feature(s) whose spline fit failed.\n", sum(smooth_failed)))
}

row_sd <- apply(mat_smooth, 1L, stats::sd, na.rm = TRUE)
valid_rows <- !smooth_failed & is.finite(row_sd) & row_sd > sqrt(.Machine$double.eps)
if (!any(valid_rows)) {
  stop("No variable feature trajectories remain after smoothing.", call. = FALSE)
}

mat_smooth <- mat_smooth[valid_rows, , drop = FALSE]
all_omics <- as.character(row_info$omics)[valid_rows]
cat(sprintf("  Retained %d variable feature trajectories.\n", nrow(mat_smooth)))

rm(matrix_table, value_table, mat_raw, smoothed, row_sd)
invisible(gc())

# ----------------------------------------------------------------------------
# 4. Correlation-distance hierarchical clustering
# ----------------------------------------------------------------------------

cat("\nStep 3/5: Clustering rows...\n")

if (nrow(mat_smooth) > 1L) {
  row_cor <- stats::cor(t(mat_smooth), use = "pairwise.complete.obs")
  row_cor[row_cor > 1] <- 1
  row_cor[row_cor < -1] <- -1
  diag(row_cor) <- 1

  row_dist <- stats::as.dist(1 - row_cor)
  row_dist[!is.finite(row_dist)] <- 2
  row_clust <- stats::hclust(row_dist, method = "ward.D2")

  rm(row_cor, row_dist)
  invisible(gc())
} else {
  row_clust <- FALSE
}
cat("  Clustering complete.\n")

# ----------------------------------------------------------------------------
# 5. Build and save the heatmap
# ----------------------------------------------------------------------------

cat("\nStep 4/5: Building heatmap...\n")

# ---- Figure specification (Nature Communications style) --------------------
# Final printed size; all font sizes are in points at this size (Nature
# journals: 5-7 pt, sans-serif). "sans" maps to Helvetica in pdf() and to
# Arial/Liberation Sans in cairo_pdf() and ragg.
FIG_WIDTH_MM    <- 80
FIG_HEIGHT_MM   <- 90
FONT_FAMILY     <- "sans"
FONT_AXIS_TITLE <- 7
FONT_TICK       <- 6
FONT_LEGEND     <- 6
LINE_PT         <- 0.5   # tick line width (pt)
DENDRO_LINE_PT  <- 0.25  # dendrogram line width (pt)
pt_to_lwd <- function(pt) pt * 96 / 72  # R lwd = 1/96 inch

Y_LABEL <- "Dysregulated molecules/microbes"
X_LABEL <- "Age (years)"
COLORBAR_TITLE <- "z-score"
OMICS_DISPLAY <- c(
  "Transcriptomics" = "Transcriptomics",
  "Metabolomics"    = "Metabolomics",
  "Lipidomics"      = "Lipidomics",
  "Gut Microbiome"  = "Gut microbiome",
  "Oral Microbiome" = "Oral microbiome"
)

heat_colors <- grDevices::colorRampPalette(
  c("#313695", "#74ADD1", "#FFFFBF", "#F46D43", "#A50026")
)(255)
heat_breaks <- seq(-2, 2, length.out = length(heat_colors) + 1L)

# ---- Data in plotting order -------------------------------------------------
row_order <- if (inherits(row_clust, "hclust")) row_clust$order else seq_len(nrow(mat_smooth))
mat_plot <- mat_smooth[row_order, , drop = FALSE]
omics_plot <- all_omics[row_order]

values_to_colors <- function(values, colors, breaks) {
  clamped <- pmin(pmax(values, min(breaks)), max(breaks))
  index <- findInterval(clamped, breaks, rightmost.closed = TRUE, all.inside = TRUE)
  out <- colors[index]
  out[is.na(values)] <- "#FFFFFF"
  out
}

heat_fill <- matrix(
  values_to_colors(as.vector(mat_plot), heat_colors, heat_breaks),
  nrow = nrow(mat_plot)
)
strip_fill <- matrix(unname(OMICS_COLORS[omics_plot]), ncol = 1L)

# ---- Row dendrogram as line segments (leaves on the right) -----------------
dendrogram_segments <- function(hc) {
  n <- length(hc$order)
  leaf_pos <- integer(n)
  leaf_pos[hc$order] <- seq_len(n)
  merges <- hc$merge
  heights <- hc$height
  node_pos <- numeric(nrow(merges))
  seg <- matrix(NA_real_, nrow = 3L * nrow(merges), ncol = 4L)

  child_info <- function(k) {
    if (k < 0L) c(leaf_pos[-k], 0) else c(node_pos[k], heights[k])
  }
  for (i in seq_len(nrow(merges))) {
    a <- child_info(merges[i, 1L])
    b <- child_info(merges[i, 2L])
    h <- heights[i]
    node_pos[i] <- (a[1L] + b[1L]) / 2
    j <- 3L * (i - 1L)
    seg[j + 1L, ] <- c(a[2L], h, a[1L], a[1L])
    seg[j + 2L, ] <- c(b[2L], h, b[1L], b[1L])
    seg[j + 3L, ] <- c(h, h, a[1L], b[1L])
  }

  max_h <- max(heights)
  to_x <- function(h) 1 - h / max_h
  to_y <- function(p) 1 - (p - 0.5) / n
  grid::segmentsGrob(
    x0 = to_x(seg[, 1L]), x1 = to_x(seg[, 2L]),
    y0 = to_y(seg[, 3L]), y1 = to_y(seg[, 4L]),
    default.units = "npc",
    gp = grid::gpar(col = "black", lwd = pt_to_lwd(DENDRO_LINE_PT), lineend = "butt"),
    name = "row_dendrogram"
  )
}

dendro_grob <- if (inherits(row_clust, "hclust")) {
  dendrogram_segments(row_clust)
} else {
  grid::nullGrob()
}

# ---- Right-side legends: colour bar (z-score) + omics keys -------------------
build_side_legend <- function() {
  title_h <- 2.8; bar_w <- 2.2; bar_h <- 16; gap <- 5; key <- 2; key_step <- 3
  n_keys <- length(OMICS_COLORS)
  block_h <- title_h + 0.8 + bar_h + gap + n_keys * key_step
  # y position for a distance `mm` below the top of the centred legend block
  y_at <- function(mm) grid::unit(0.5, "npc") + grid::unit(block_h / 2 - mm, "mm")
  text_gp <- grid::gpar(fontfamily = FONT_FAMILY, fontsize = FONT_LEGEND, col = "black")

  bar_top <- title_h + 0.8
  tick_values <- c(2, 1, 0, -1, -2)
  tick_mm <- bar_top + (2 - tick_values) / 4 * bar_h

  children <- list(
    grid::textGrob(
      COLORBAR_TITLE, x = grid::unit(0, "npc"), y = y_at(0),
      hjust = 0, vjust = 1, gp = text_gp, name = "colorbar_title"
    ),
    grid::rasterGrob(
      image = matrix(rev(heat_colors), ncol = 1L),
      x = grid::unit(0, "npc"), y = y_at(bar_top),
      width = grid::unit(bar_w, "mm"), height = grid::unit(bar_h, "mm"),
      just = c("left", "top"), interpolate = TRUE, name = "colorbar"
    ),
    grid::segmentsGrob(
      x0 = grid::unit(bar_w, "mm"), x1 = grid::unit(bar_w + 0.6, "mm"),
      y0 = y_at(tick_mm), y1 = y_at(tick_mm),
      gp = grid::gpar(lwd = pt_to_lwd(LINE_PT), lineend = "butt"),
      name = "colorbar_ticks"
    ),
    grid::textGrob(
      format(tick_values), x = grid::unit(bar_w + 1.0, "mm"), y = y_at(tick_mm),
      hjust = 0, vjust = 0.5, gp = text_gp, name = "colorbar_labels"
    )
  )

  keys_top <- bar_top + bar_h + gap
  key_centres <- keys_top + (seq_len(n_keys) - 0.5) * key_step
  children <- c(children, list(
    grid::rectGrob(
      x = grid::unit(0, "npc"), y = y_at(key_centres),
      width = grid::unit(key, "mm"), height = grid::unit(key, "mm"),
      just = c("left", "centre"),
      gp = grid::gpar(fill = unname(OMICS_COLORS), col = NA),
      name = "omics_keys"
    ),
    grid::textGrob(
      unname(OMICS_DISPLAY[names(OMICS_COLORS)]),
      x = grid::unit(key + 0.8, "mm"), y = y_at(key_centres),
      hjust = 0, vjust = 0.5, gp = text_gp, name = "omics_labels"
    )
  ))
  grid::gTree(children = do.call(grid::gList, children), name = "side_legend")
}

# ---- Age axis ---------------------------------------------------------------
tick_ages <- seq(AGE_MIN, AGE_MAX, by = 5)
tick_ages <- tick_ages[tick_ages >= AGE_MIN & tick_ages <= AGE_MAX]
# Cell centres span AGE_MIN (first column) to AGE_MAX (last column).
age_to_npc <- function(age) {
  (0.5 + (age - AGE_MIN) / (AGE_MAX - AGE_MIN) * (N_SMOOTH - 1)) / N_SMOOTH
}
tick_x <- age_to_npc(tick_ages)

# ---- Assemble the panel with an explicit mm layout -------------------------
# Columns: margin | y-label | dendrogram | gap | omics strip | gap | heatmap |
#          gap | legends | margin
col_mm <- c(0.8, 3.2, 7.0, 0.3, 1.6, 0.4, NA, 2.0, 18.5, 0.5)
col_mm[is.na(col_mm)] <- FIG_WIDTH_MM - sum(col_mm, na.rm = TRUE)
# Rows: top margin | heatmap | tick | tick labels | x title | bottom margin
row_mm <- c(1.5, NA, 0.9, 2.8, 3.4, 0.8)
row_mm[is.na(row_mm)] <- FIG_HEIGHT_MM - sum(row_mm, na.rm = TRUE)
if (any(col_mm <= 0) || any(row_mm <= 0)) {
  stop("Figure size is too small for the fixed layout margins.", call. = FALSE)
}
COL <- list(ylab = 2L, dendro = 3L, strip = 5L, heat = 7L, legend = 9L)
ROW <- list(heat = 2L, tick = 3L, ticklab = 4L, xtitle = 5L)

figure <- gtable::gtable(
  widths = grid::unit(col_mm, "mm"),
  heights = grid::unit(row_mm, "mm"),
  name = "age_trend_heatmap"
)
add <- function(g, grob, t, l, name) {
  gtable::gtable_add_grob(g, grob, t = t, l = l, name = name, clip = "off")
}

figure <- add(figure, build_side_legend(), ROW$heat, COL$legend, "legend")
figure <- add(
  figure,
  grid::textGrob(
    Y_LABEL, rot = 90,
    gp = grid::gpar(fontfamily = FONT_FAMILY, fontsize = FONT_AXIS_TITLE)
  ),
  ROW$heat, COL$ylab, "y_label"
)
figure <- add(figure, dendro_grob, ROW$heat, COL$dendro, "dendrogram")
figure <- add(
  figure,
  grid::rasterGrob(strip_fill, width = grid::unit(1, "npc"), height = grid::unit(1, "npc"),
                   interpolate = FALSE),
  ROW$heat, COL$strip, "omics_strip"
)
figure <- add(
  figure,
  grid::rasterGrob(heat_fill, width = grid::unit(1, "npc"), height = grid::unit(1, "npc"),
                   interpolate = FALSE),
  ROW$heat, COL$heat, "heatmap_body"
)
figure <- add(
  figure,
  grid::segmentsGrob(
    x0 = tick_x, x1 = tick_x, y0 = 1, y1 = 0, default.units = "npc",
    gp = grid::gpar(lwd = pt_to_lwd(LINE_PT), lineend = "butt")
  ),
  ROW$tick, COL$heat, "x_ticks"
)
figure <- add(
  figure,
  grid::textGrob(
    as.character(tick_ages), x = tick_x, y = grid::unit(1, "npc"),
    default.units = "npc", vjust = 1,
    gp = grid::gpar(fontfamily = FONT_FAMILY, fontsize = FONT_TICK)
  ),
  ROW$ticklab, COL$heat, "x_tick_labels"
)
figure <- add(
  figure,
  grid::textGrob(
    X_LABEL, y = grid::unit(1, "npc"), vjust = 1,
    gp = grid::gpar(fontfamily = FONT_FAMILY, fontsize = FONT_AXIS_TITLE)
  ),
  ROW$xtitle, COL$heat, "x_title"
)

cat(sprintf(
  "  Panel: %g x %g mm; heatmap body %.1f x %.1f mm.\n",
  FIG_WIDTH_MM, FIG_HEIGHT_MM, col_mm[COL$heat], row_mm[ROW$heat]
))

describe_temp_file <- function(path) {
  if (!file.exists(path)) return("file was not created")
  sprintf("file exists but is %s bytes", format(file.info(path)$size))
}

disk_report <- function(path) {
  out <- tryCatch(
    system2("df", c("-h", shQuote(dirname(path)), shQuote(tempdir())), stdout = TRUE, stderr = TRUE),
    error = function(e) "df not available"
  )
  paste(out, collapse = "\n    ")
}

# The figures/ folder lives on an S3 bucket mounted as a folder (FUSE). Such
# mounts only accept a new file written once, start to finish: graphics
# devices, file.copy() (create-then-append) and rename all fail there. So every
# output is first written to the local disk (tempdir()), then streamed into the
# destination with a single open-write-close.
publish_file <- function(local_path, final_path) {
  size <- if (file.exists(local_path)) file.info(local_path)$size else NA_real_
  if (is.na(size) || size <= 0) {
    stop(
      sprintf(
        "Plot device did not create a valid file: %s (%s).\n  Disk usage:\n    %s",
        local_path, describe_temp_file(local_path), disk_report(local_path)
      ),
      call. = FALSE
    )
  }
  bytes <- readBin(local_path, what = "raw", n = size)

  if (file.exists(final_path) && !isTRUE(suppressWarnings(unlink(final_path) == 0L))) {
    warning(sprintf("Could not delete existing %s; trying to overwrite it.", final_path),
            call. = FALSE)
  }
  write_error <- tryCatch(
    {
      con <- file(final_path, open = "wb")
      on.exit(try(close(con), silent = TRUE), add = TRUE)
      writeBin(bytes, con)
      close(con)
      on.exit()
      NULL
    },
    error = function(e) conditionMessage(e),
    warning = function(w) conditionMessage(w)
  )
  written <- if (file.exists(final_path)) file.info(final_path)$size else NA_real_
  if (!is.null(write_error) || is.na(written) || written != size) {
    stop(
      sprintf(
        paste0(
          "Could not write %s (%s; wrote %s of %s bytes).\n",
          "  The finished file is kept at: %s\n",
          "  If the folder is an S3 mount, the existing file may need deleting first,\n",
          "  or the mount may need --allow-overwrite / --allow-delete."
        ),
        final_path, if (is.null(write_error)) "size mismatch" else write_error,
        format(written), format(size), local_path
      ),
      call. = FALSE
    )
  }
  unlink(local_path)
  invisible(final_path)
}

# Draw `grob` to a temp file with one device. Returns NULL on success, or a
# character message describing why the device did not produce a usable file.
try_device <- function(temp_path, open_device, grob) {
  if (file.exists(temp_path)) unlink(temp_path)
  devices_before <- grDevices::dev.list()
  device_id <- NA_integer_
  plot_error <- NULL

  tryCatch(
    {
      open_device(temp_path)
      device_id <- grDevices::dev.cur()
      if (device_id %in% devices_before) {
        stop("device did not open a new graphics device", call. = FALSE)
      }
      grid::grid.newpage()
      grid::grid.draw(grob)
    },
    error = function(error) {
      plot_error <<- error
    },
    finally = {
      if (!is.na(device_id) && !(device_id %in% devices_before) &&
          device_id %in% grDevices::dev.list()) {
        grDevices::dev.off(which = device_id)
      }
    }
  )

  if (!is.null(plot_error)) {
    if (file.exists(temp_path)) unlink(temp_path)
    return(conditionMessage(plot_error))
  }
  if (!file.exists(temp_path) || file.info(temp_path)$size <= 0) {
    msg <- describe_temp_file(temp_path)
    if (file.exists(temp_path)) unlink(temp_path)
    return(msg)
  }
  NULL
}

# `devices` is a named list of device-opening functions, tried in order.
save_grid_plot <- function(final_path, devices, grob) {
  if (is.function(devices)) devices <- list(device = devices)
  ext <- tools::file_ext(final_path)
  # Render on the local disk, never directly into the (possibly S3) output dir.
  temp_path <- file.path(
    tempdir(),
    sprintf("%s.tmp-%d.%s", tools::file_path_sans_ext(basename(final_path)), Sys.getpid(), ext)
  )

  failures <- character(0)
  for (device_name in names(devices)) {
    problem <- try_device(temp_path, devices[[device_name]], grob)
    if (is.null(problem)) {
      if (length(failures) > 0L) {
        cat(sprintf("  Note: used %s after: %s\n", device_name, paste(failures, collapse = "; ")))
      }
      return(publish_file(temp_path, final_path))
    }
    failures <- c(failures, sprintf("%s -> %s", device_name, problem))
  }

  stop(
    sprintf(
      "Failed to write %s. Tried:\n  %s\n  Disk usage:\n    %s",
      final_path, paste(failures, collapse = "\n  "), disk_report(final_path)
    ),
    call. = FALSE
  )
}

png_path <- paste0(FIGURE_PREFIX, "_R_dysregulated.png")
pdf_path <- paste0(FIGURE_PREFIX, "_R_dysregulated.pdf")
smooth_path <- paste0(OUTPUT_PREFIX, "_smooth_matrix_dysregulated.csv")

cat("\nStep 5/5: Saving outputs...\n")

open_png <- if (requireNamespace("ragg", quietly = TRUE)) {
  function(path) {
    ragg::agg_png(
      path,
      width = FIG_WIDTH_MM,
      height = FIG_HEIGHT_MM,
      units = "mm",
      res = 600,
      background = "white"
    )
  }
} else {
  function(path) {
    grDevices::png(
      path, width = FIG_WIDTH_MM, height = FIG_HEIGHT_MM,
      units = "mm", res = 600, bg = "white"
    )
  }
}

save_grid_plot(png_path, open_png, figure)
cat(sprintf("  Saved: %s\n", png_path))

pdf_devices <- list(
  "grDevices::pdf" = function(path) {
    grDevices::pdf(
      path, width = FIG_WIDTH_MM / 25.4, height = FIG_HEIGHT_MM / 25.4,
      useDingbats = FALSE, family = "Helvetica"
    )
  }
)
if (isTRUE(capabilities("cairo"))) {
  pdf_devices[["grDevices::cairo_pdf"]] <- function(path) {
    grDevices::cairo_pdf(
      path, width = FIG_WIDTH_MM / 25.4, height = FIG_HEIGHT_MM / 25.4,
      family = "Arial"
    )
  }
}
save_grid_plot(pdf_path, pdf_devices, figure)
cat(sprintf("  Saved: %s\n", pdf_path))

smooth_temp <- file.path(tempdir(), paste0(basename(smooth_path), ".tmp-", Sys.getpid()))
write.csv(mat_smooth, smooth_temp, row.names = TRUE)
publish_file(smooth_temp, smooth_path)
cat(sprintf("  Saved: %s\n", smooth_path))

cat("\nCompleted successfully.\n")
