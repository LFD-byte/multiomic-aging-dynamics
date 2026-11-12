#!/usr/bin/env Rscript
# Descriptive window selection from saved DEswan tests and current metadata.
# No significance-count maximization or independent-validation claim is made.
options(stringsAsFactors = FALSE, warn = 1)
suppressPackageStartupMessages({library(data.table); library(dplyr)})
args <- commandArgs(trailingOnly = TRUE)
file_arg <- grep('^--file=', commandArgs(), value=TRUE)
base_dir <- dirname(normalizePath(sub('^--file=', '', file_arg[1]), winslash='/'))
source_dir <- file.path(base_dir, 'outputs/deswan-window-sensitivity')
data_dir <- file.path(base_dir, 'outputs/deswan-window-selection')
figure_dir <- file.path(base_dir, 'figures/deswan-window-selection')
invisible(lapply(c(data_dir,figure_dir), dir.create, recursive=TRUE, showWarnings=FALSE))
assert <- function(ok, message) if (!isTRUE(ok)) stop(message, call.=FALSE)
WINDOWS <- c(5,7.5,10,12.5,15)
SELECTED_WINDOW <- 7.5
assert(SELECTED_WINDOW %in% WINDOWS, 'The primary window must be among the sensitivity windows.')
OMICS <- c('transcriptomics','metabolomics','lipidomics','gut_microbiome','oral_microbiome')
Q <- c(.0001,.001,.01,.05)
pair_index <- t(combn(WINDOWS,2))
counts <- fread(file.path(source_dir,'source_counts_all.csv'))
crests <- fread(file.path(source_dir,'detected_crests.csv'))
assert(setequal(unique(counts$window), WINDOWS), 'Rebuild the five-window counts first.')
counts[, full_support := age >= 40+window & age <= 70-window]
fwrite(counts, file.path(data_dir,'counts_with_boundary_status.csv'))
coverage <- data.table(window=WINDOWS, total_width=2*WINDOWS,
  complete_center_min=40+WINDOWS, complete_center_max=70-WINDOWS)
coverage[, complete_integer_centers := floor(complete_center_max)-ceiling(complete_center_min)+1L]
coverage[, complete_fraction_of_29 := complete_integer_centers/29]
coverage[, age64_complete := 64 >= complete_center_min & 64 <= complete_center_max]
fwrite(coverage, file.path(data_dir,'window_coverage.csv'))

if (!'--plots-only' %in% args) {
  support_parts <- list(); metadata_parts <- list(); overlap_parts <- list()
  feature_parts <- list(); raw_input_parts <- list()
  for (om in OMICS) {
    raw_file <- file.path(dirname(base_dir),'01-interim_data/omics_data',paste0(om,'_with_metadata.csv'))
    meta <- fread(raw_file, select=c('participant_id','age','sex'))
    raw_n <- nrow(meta)
    meta <- meta[is.finite(age) & age >= 40 & age <= 70 & !is.na(sex)]
    assert(!anyNA(meta$participant_id),'Missing participant IDs require review.')
    metadata_parts[[om]] <- data.table(omics=om,raw_rows=raw_n,complete_rows=nrow(meta),
      unique_participants=uniqueN(meta$participant_id),
      duplicate_ids=meta[, .N, by=participant_id][N>1,.N],
      observed_min=min(meta$age),observed_max=max(meta$age))
    raw_input_parts[[om]] <- data.table(omics=om, file=basename(raw_file),
      bytes=file.info(raw_file)$size,modified=as.character(file.info(raw_file)$mtime))
    support_parts[[om]] <- rbindlist(lapply(WINDOWS, function(w) rbindlist(lapply(41:69, function(a) {
      le <- meta[age >= a-w & age < a]; ri <- meta[age > a & age <= a+w]
      nl <- uniqueN(le$participant_id); nr <- uniqueN(ri$participant_id)
      data.table(omics=om,window=w,age=a,full_support=a-w>=40 & a+w<=70,
        actual_left_width=a-max(40,a-w),actual_right_width=min(70,a+w)-a,
        n_left=nrow(le),n_right=nrow(ri),participants_left=nl,participants_right=nr,
        shared_ids_between_sides=length(intersect(le$participant_id,ri$participant_id)),
        smaller_larger_n_ratio=if(max(nl,nr)>0) min(nl,nr)/max(nl,nr) else NA_real_,
        n_at_center=sum(meta$age==a))
    }))))
    tests <- list()
    for (w in WINDOWS) {
      tag <- gsub('\\.','p',as.character(w))
      path <- file.path(base_dir,'outputs',paste0('deswan-window_',w),
        paste0(om,'_deswan_breakpoint_window_',tag,'.rds'))
      result <- readRDS(path)
      p <- as.data.table(result$p)[factor=='qt',.(variable,age=window.center,p=pvalue)]
      b <- as.data.table(result$coeff)[factor=='qt',.(variable,age=window.center,beta=coefficient)]
      assert(!anyDuplicated(p[,.(variable,age)]) && !anyDuplicated(b[,.(variable,age)]),
        'Duplicate feature-age keys in saved model output.')
      p[,q:=p.adjust(p,'BH'),by=age]
      p <- merge(p,b,by=c('variable','age'),all=TRUE)
      assert(!anyNA(p), 'Missing p/q/coefficient values require review.')
      p[,`:=`(omics=om,window=w)]
      for (cut in Q) {
        chk <- p[,.(recomputed=sum(q<cut)),by=age]
        saved <- counts[omics==om & window==w & q_cutoff==cut,.(age,n_significant)]
        chk <- merge(chk,saved,by='age')
        assert(nrow(chk)==29 && all(chk$recomputed==chk$n_significant),
          'Independent BH recomputation differs from the figure source.')
      }
      tests[[as.character(w)]] <- p
      feature_parts[[paste(om,w)]] <- p[age %in% c(52,64)]
    }
    for (i in seq_len(nrow(pair_index))) {
      w1 <- pair_index[i,1]; w2 <- pair_index[i,2]
      for (a in c(52,64)) {
        z <- merge(tests[[as.character(w1)]][age==a,.(variable,q1=q,b1=beta)],
                   tests[[as.character(w2)]][age==a,.(variable,q2=q,b2=beta)],by='variable')
        assert(nrow(z)==counts[omics==om & window==w1 & age==a & q_cutoff==.05,n_tested],
          'Different feature universe in paired coefficient comparison.')
        for (cut in Q) {
          s1 <- z$q1<cut; s2 <- z$q2<cut; both <- s1 & s2
          overlap_parts[[length(overlap_parts)+1L]] <- data.table(
            omics=om,age=a,window1=w1,window2=w2,q_cutoff=cut,n1=sum(s1),n2=sum(s2),
            intersection=sum(both),union=sum(s1|s2),
            fraction_window1_retained=if(sum(s1))sum(both)/sum(s1) else NA_real_,
            fraction_window2_retained=if(sum(s2))sum(both)/sum(s2) else NA_real_,
            jaccard=if(sum(s1|s2))sum(both)/sum(s1|s2) else NA_real_,
            shared_sign_agreement=if(sum(both))mean(sign(z$b1[both])==sign(z$b2[both])) else NA_real_,
            all_beta_pearson=cor(z$b1,z$b2),all_beta_spearman=cor(z$b1,z$b2,method='spearman'))
        }
      }
    }
    message('Verified q values, coefficients and participant support: ',om)
  }
  support <- rbindlist(support_parts)
  fwrite(support,file.path(data_dir,'participant_support.csv'))
  fwrite(rbindlist(metadata_parts),file.path(data_dir,'metadata_audit.csv'))
  fwrite(rbindlist(raw_input_parts),file.path(data_dir,'metadata_input_inventory.csv'))
  fwrite(rbindlist(overlap_parts),file.path(data_dir,'feature_overlap.csv'))
  fwrite(rbindlist(feature_parts),file.path(data_dir,'feature_evidence_52_64.csv.gz'))
} else support <- fread(file.path(data_dir,'participant_support.csv'))

min_support <- support[full_support==TRUE,.(min_left_ids=min(participants_left),
  min_right_ids=min(participants_right),max_shared_ids=max(shared_ids_between_sides)),by=.(omics,window)]
fwrite(min_support,file.path(data_dir,'minimum_complete_window_support.csv'))
fwrite(support[age==64],file.path(data_dir,'age64_support.csv'))
peak_rows <- counts[,{
  a <- if (max(n_significant)>0) as.numeric(age[which.max(n_significant)]) else NA_real_
  .(global_max_age=a,global_max_n=max(n_significant),
    global_max_tied_ages=if(max(n_significant)>0)paste(age[n_significant==max(n_significant)],collapse=';') else '',
    global_max_complete=a>=40+window[1]&a<=70-window[1],
    complete_max_age=if(max(n_significant[full_support])>0)as.numeric(age[full_support][which.max(n_significant[full_support])]) else NA_real_,
    complete_max_n=max(n_significant[full_support]),
    n_nonzero_complete=sum(n_significant[full_support]>0),
    mean_count_complete=mean(n_significant[full_support]))
},by=.(omics,window,q_cutoff)]
pk <- crests[,.(detected_ages=paste(peak_age,collapse=';')),by=.(omics,window,q_cutoff)]
peak_rows <- merge(peak_rows,pk,by=c('omics','window','q_cutoff'),all.x=TRUE)
fwrite(peak_rows,file.path(data_dir,'peak_and_threshold_summary.csv'))
fwrite(counts[age %in% c(52,64)],file.path(data_dir,'age52_age64_counts.csv'))

profiles <- list()
for (om in c('All',OMICS)) for (cut in Q) for (i in seq_len(nrow(pair_index))) {
  w1 <- pair_index[i,1]; w2 <- pair_index[i,2]
  d <- merge(counts[omics==om & window==w1 & q_cutoff==cut,.(age,y1=n_significant)],
             counts[omics==om & window==w2 & q_cutoff==cut,.(age,y2=n_significant)],by='age')
  for (scope in c('all_29_centers','common_complete_centers')) {
    z <- if(scope=='all_29_centers')d else d[age>=40+max(w1,w2) & age<=70-max(w1,w2)]
    can_cor <- nrow(z)>=3 && sd(z$y1)>0 && sd(z$y2)>0
    profiles[[length(profiles)+1L]] <- data.table(omics=om,q_cutoff=cut,window1=w1,window2=w2,
      scope=scope,n_centers=nrow(z),pearson=if(can_cor)cor(z$y1,z$y2) else NA_real_,
      spearman=if(can_cor)cor(z$y1,z$y2,method='spearman') else NA_real_)
  }
}
fwrite(rbindlist(profiles),file.path(data_dir,'profile_similarity.csv'))
decision <- data.table(window=WINDOWS,total_width=2*WINDOWS,
  role=c('Boundary sensitivity','Primary analysis','Broader sensitivity','Broad-window sensitivity','Broad-window sensitivity'),
  rationale=c('Complete age64 comparison; weaker threshold and multi-omics detection',
    'Practical balance of age coverage and molecular/threshold stability',
    'Greater broad-trend detection; fewer complete centers and more age64 imbalance',
    'Only five complete integer centers; earlier crest merges',
    'Only center55 complete; earlier crest moves to55'))
fwrite(merge(decision,coverage,by=c('window','total_width')),file.path(data_dir,'window_recommendation.csv'))
hashes <- fread(file.path(source_dir,'input_checksums.csv'))
current_hash <- unname(tools::md5sum(file.path(base_dir,hashes$file)))
assert(!anyNA(current_hash) && identical(current_hash,hashes$md5),'An input RDS/CSV changed during the analysis.')
fwrite(data.table(file=hashes$file,expected_md5=hashes$md5,current_md5=current_hash,
  unchanged=current_hash==hashes$md5),file.path(data_dir,'input_immutability_check.csv'))
manifest <- list(generated_at=format(Sys.time(),tz='UTC',usetz=TRUE),windows=WINDOWS,
  selected_window=SELECTED_WINDOW,
  recommended_primary_window=SELECTED_WINDOW,recommended_total_width=2*SELECTED_WINDOW,
  boundary_sensitivity_window=5,broad_comparison_window=10,
  recommendation_scope='Five-omics exploratory aging profiles; age64 requires the complete Window5 sensitivity check',
  q_cutoffs=Q,age_support=c(40,70),boundary_rule='Complete if 40+b <= center <= 70-b',
  unit='Participants counted by unique ID separately per side; repeated IDs can occur in both sides',
  estimation='Saved sex-adjusted Gaussian DEswan GLM coefficients and p values; BH within omics and center',
  comparison='Descriptive, correlated analyses on the same participants; no independent replication',
  primary_q=.05,primary_complete_center_range=c(40+SELECTED_WINDOW,70-SELECTED_WINDOW),
  mathematical_optimum_established=FALSE,
  selection_rule='Balance complete age coverage, local resolution, sample support and molecular/threshold stability; do not maximize discovery counts',
  bootstrap_performed=FALSE,independent_validation_performed=FALSE,
  source_rds_modified=FALSE,feature_q_recomputed_and_matched=TRUE,
  limitations=c('Per-feature missingness can reduce metadata-based sample counts',
    'Available saved fits do not model participant-level repetition explicitly',
    'Count peaks do not themselves establish a nonlinear breakpoint',
    'All-omics raw counts are dominated by transcriptomics',
    'Post-hoc practical parameter recommendation; not a mathematical optimum'))
jsonlite::write_json(manifest,file.path(data_dir,'selection_manifest.json'),pretty=TRUE,auto_unbox=TRUE)
writeLines(capture.output(sessionInfo()),file.path(data_dir,'sessionInfo.txt'))
message('Wrote comparison tables; all ',nrow(hashes),' original RDS/CSV checksums match.')
print(decision)
if ('--data-only' %in% args) quit(save='no',status=0)

# Figure contract: compare local evidence at two FDR cutoffs and show the
# exact age coverage lost with broader windows. R draws all three panels.
# Quantitative grid, 183 x 103 mm, editable PDF/SVG, 600 dpi TIFF; no CIs are
# invented for these deterministic summaries. All 29 supplied ages are shown.
suppressPackageStartupMessages({library(ggplot2); library(patchwork)})
FONT <- 'Arial'
# Reuse base_theme from the window-sensitivity script without running its analysis.
plot_source <- parse(file.path(base_dir,'01-de_swan_window_sensitivity.R'))
helpers <- c('base_theme')
for (expr in plot_source) if (is.call(expr) && identical(expr[[1]],as.name('<-')) &&
  as.character(expr[[2]]) %in% helpers) eval(expr)
assert(exists('base_theme'),'base_theme not found in 01-de_swan_window_sensitivity.R.')
export_figure <- function(plot,name,width_mm,height_mm,n_panels) {
  w <- width_mm/25.4; h <- height_mm/25.4
  out <- file.path(figure_dir,name)
  grDevices::cairo_pdf(paste0(out,'.pdf'),width=w,height=h,family=FONT,bg='white')
  print(plot); grDevices::dev.off()
  svglite::svglite(paste0(out,'.svg'),width=w,height=h,bg='white')
  print(plot); grDevices::dev.off()
  ggsave(paste0(out,'.png'),plot,width=w,height=h,device=ragg::agg_png,dpi=300,bg='white')
  ggsave(paste0(out,'.tiff'),plot,width=w,height=h,device=ragg::agg_tiff,dpi=600,compression='lzw',bg='white')
  message('Exported ',name,': PDF, SVG, 300 dpi PNG, 600 dpi TIFF.')
}
WCOL <- c('5'='#65519A','7.5'='#CD7557','10'='#0077B5','12.5'='#B3903C','15'='#648D70')
plot_windows <- c(5,7.5,10)
plot_counts <- counts[omics=='All' & window %in% plot_windows & q_cutoff %in% c(.01,.05)]
plot_counts[,Window:=factor(as.character(window),levels=as.character(plot_windows))]
fwrite(plot_counts,file.path(data_dir,'source_selection_figure_curves.csv'))
comparison_plot <- function(cut,tag) {
  d <- plot_counts[q_cutoff==cut]
  top <- if(cut==.05)4000 else 1750
  breaks <- if(cut==.05)seq(0,4000,1000) else seq(0,1500,500)
  ggplot(d[window!=SELECTED_WINDOW],aes(age,n_significant,colour=Window,group=Window))+
    geom_line(linewidth=.38)+geom_point(size=.6)+
    geom_line(data=d[window==SELECTED_WINDOW],linewidth=.75)+
    geom_point(data=d[window==SELECTED_WINDOW],size=.85)+
    scale_x_continuous(limits=c(40,70),breaks=seq(40,70,5),expand=c(0,0))+
    scale_y_continuous(limits=c(0,top),breaks=breaks,labels=scales::label_comma(),expand=c(0,0))+
    scale_colour_manual(values=WCOL,limits=as.character(plot_windows),name='Years per side')+
    labs(x='Age (years)',y='Significant features',title=if(cut==.05)'Window comparison' else 'Stricter FDR comparison',
      subtitle=paste0('All five omics; q < ',cut),tag=tag)+base_theme+
    guides(colour=guide_legend(nrow=1,override.aes=list(linewidth=.55,size=1)))
}
cv <- copy(coverage)
cv[,Window:=factor(as.character(window),levels=rev(as.character(WINDOWS)))]
cv[,range_label:=paste0(complete_center_min,' - ',complete_center_max)]
cv[window==15,range_label:='55 only']
fwrite(cv,file.path(data_dir,'source_selection_figure_coverage.csv'))
p_c <- ggplot(cv,aes(y=Window,colour=as.character(window)))+
  geom_segment(aes(x=40,xend=70,yend=Window),colour='#E5E8EB',linewidth=1.3)+
  geom_vline(xintercept=64,linetype='dashed',linewidth=.3,colour='#929AA2')+
  geom_segment(aes(x=complete_center_min,xend=complete_center_max,yend=Window),linewidth=1.3)+
  geom_point(aes(x=complete_center_min),size=1.2)+
  geom_point(aes(x=complete_center_max),size=1.2)+
  geom_text(aes(x=55,label=range_label),nudge_y=.25,colour='#3A4249',family=FONT,size=2.15)+
  scale_colour_manual(values=WCOL,guide='none')+
  scale_x_continuous(limits=c(40,70),breaks=seq(40,70,5),expand=c(0,0))+
  scale_y_discrete(expand=expansion(add=c(.35,.65)))+
  labs(x='Window center age (years)',y='Window (years per side)',title='Complete age coverage',
    subtitle='Colored spans: both sides complete',tag='c')+base_theme+
  theme(plot.margin=margin(t=4,r=5,b=4,l=8.1))
selection_plot <- wrap_plots(comparison_plot(.05,'a'),comparison_plot(.01,'b'),p_c,ncol=3,widths=c(1,1,1))+
  plot_annotation(title='Window selection for participants aged 40-70',
    subtitle=paste0('Primary window: ',SELECTED_WINDOW,
      ' years per side. Window 5 checks age-64 boundary sensitivity.'),
    caption='Observed counts; BH within each omics and age. Dashed line: center age 64. Gray spans: incomplete windows.',
    theme=theme(plot.title=element_text(family=FONT,size=10,face='bold'),
      plot.subtitle=element_text(family=FONT,size=7,margin=margin(b=5)),
      plot.caption=element_text(family=FONT,size=6.2,hjust=0,margin=margin(t=5)),
      plot.margin=margin(5,5,5,5)))
export_figure(selection_plot,'window_selection_evidence',width_mm=183,height_mm=103,n_panels=3)
