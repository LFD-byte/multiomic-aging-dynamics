# Shared lipid nomenclature for pathway and LION/web exports.
# Sum compositions are never expanded into unmeasured isomers.
# Main-peak selection belongs to the peak helpers/callers. Annotation is
# independent of peak status: keep the full measured lipid background even
# when only one main peak is analyzed.
# LION submission eligibility is applied separately by lion_web_mapping().
# For example, CoQ10 remains a valid local common-name annotation even though
# the user's LION/web matching report did not recognize it. Do not relabel PCE
# as a specific ether/plasmalogen structure merely to obtain a website match.

parse_sum_composition <- function(feature, prefix_pattern) {
  pattern <- paste0("^", prefix_pattern, "_([0-9]+)[.:]([0-9]+)")
  match <- regexec(pattern, feature, ignore.case = TRUE)
  values <- regmatches(feature, match)[[1]]
  if (length(values) < 3) return(c(NA_real_, NA_real_))
  c(as.numeric(values[[2]]), as.numeric(values[[3]]))
}

parse_sphingolipid <- function(feature) {
  pattern <- "_d([0-9]+)[.:]([0-9]+)[./_]c?([0-9]+)[.:]([0-9]+)"
  match <- regexec(pattern, feature, ignore.case = TRUE)
  values <- regmatches(feature, match)[[1]]
  if (length(values) < 5) return(rep(NA_real_, 4))
  as.numeric(values[2:5])
}

annotate_one_lipid <- function(feature) {
  original <- feature
  feature_lower <- tolower(feature)
  feature_lower <- gsub("-", "_", feature_lower, fixed = TRUE)

  lipid_class <- NA_character_
  category <- NA_character_
  standard_name <- NA_character_
  mapping_status <- "unsupported"
  total_c <- NA_real_
  total_db <- NA_real_
  acyl_c <- NA_real_
  acyl_db <- NA_real_

  class_map <- c(
    "tag" = "TG", "dag" = "DG", "pc" = "PC", "pce" = "PCE",
    "pe" = "PE", "pg" = "PG", "pi" = "PI", "ps" = "PS",
    "lysopc" = "LPC", "lysope" = "LPE", "sm" = "SM",
    "cer" = "Cer", "glccer" = "GlcCer", "laccer" = "LacCer",
    "gsl" = "GSL", "fa" = "FA", "acca" = "AcylCarnitine"
  )

  matched_prefix <- names(class_map)[
    vapply(
      names(class_map),
      function(prefix) startsWith(feature_lower, paste0(prefix, "_")),
      logical(1)
    )
  ]

  if (length(matched_prefix) > 0) {
    prefix <- matched_prefix[[which.max(nchar(matched_prefix))]]
    lipid_class <- unname(class_map[[prefix]])
  }

  if (startsWith(feature_lower, "x22.6_cholesteryl_ester") ||
      startsWith(feature_lower, "22.6_cholesteryl_ester")) {
    lipid_class <- "CE"
    category <- "Sterol lipids"
    total_c <- 22
    total_db <- 6
    acyl_c <- 22
    acyl_db <- 6
    standard_name <- "CE(22:6)"
    mapping_status <- "sum_composition"
  } else if (feature_lower == "coenzyme_q10") {
    lipid_class <- "CoQ"
    category <- "Prenol lipids"
    standard_name <- "CoQ10"
    mapping_status <- "common_name"
  } else if (lipid_class %in% c("Cer", "GlcCer", "LacCer", "GSL", "SM")) {
    sph <- parse_sphingolipid(feature_lower)
    if (all(is.finite(sph))) {
      total_c <- sph[[1]] + sph[[3]]
      total_db <- sph[[2]] + sph[[4]]
      acyl_c <- sph[[3]]
      acyl_db <- sph[[4]]
      standard_name <- paste0(
        lipid_class, "(d", sph[[1]], ":", sph[[2]], "/",
        sph[[3]], ":", sph[[4]], ")"
      )
      mapping_status <- "molecular_species"
    }
    category <- "Sphingolipids"
  } else if (!is.na(lipid_class)) {
    prefix <- matched_prefix[[which.max(nchar(matched_prefix))]]
    composition <- parse_sum_composition(feature_lower, prefix)
    if (all(is.finite(composition))) {
      total_c <- composition[[1]]
      total_db <- composition[[2]]
      acyl_c <- total_c
      acyl_db <- total_db
      standard_name <- paste0(lipid_class, "(", total_c, ":", total_db, ")")
      mapping_status <- "sum_composition"
    }

    if (lipid_class %in% c("TG", "DG")) category <- "Glycerolipids"
    if (lipid_class %in% c("PC", "PCE", "PE", "PG", "PI", "PS", "LPC", "LPE")) {
      category <- "Glycerophospholipids"
    }
    if (lipid_class %in% c("FA", "AcylCarnitine")) category <- "Fatty acyls"
    if (lipid_class == "PCE") mapping_status <- "ambiguous_subclass"
  }

  if (is.na(standard_name)) standard_name <- original

  unsaturation <- if (!is.finite(total_db)) {
    NA_character_
  } else if (total_db == 0) {
    "Saturated"
  } else if (total_db == 1) {
    "Monounsaturated"
  } else {
    "Polyunsaturated"
  }

  chain_length <- if (!is.finite(total_c)) {
    NA_character_
  } else if (total_c <= 34) {
    "Total carbon <=34"
  } else if (total_c <= 44) {
    "Total carbon 35-44"
  } else if (total_c <= 54) {
    "Total carbon 45-54"
  } else {
    "Total carbon >=55"
  }

  data.frame(
    feature = original,
    standard_name = standard_name,
    lipid_class = lipid_class,
    category = category,
    total_carbons = total_c,
    total_double_bonds = total_db,
    acyl_carbons = acyl_c,
    acyl_double_bonds = acyl_db,
    unsaturation = unsaturation,
    chain_length = chain_length,
    mapping_status = mapping_status,
    stringsAsFactors = FALSE
  )
}

annotate_lipids <- function(features) {
  annotations <- do.call(rbind, lapply(unique(features), annotate_one_lipid))
  rownames(annotations) <- NULL
  annotations
}

