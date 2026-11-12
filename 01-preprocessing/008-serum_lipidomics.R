setwd("/home/ec2-user/studies/HUIFA-RNA-SEQ-new/Project/2026-nc-final/01-preprocessing")

library(imputeLCMD)
set.seed(123)

offset <- 1e-6            # 或者 1、0.5，看你数据单位
x_raw <- read.csv("interim_data/008-serum_lipidomics_feature_filtered.csv", header = TRUE, stringsAsFactors = FALSE)
x_log <- log2(x_raw + offset)

x_log_imp <- impute.QRILC(x_log)[[1]]  # 使用 QRILC 方法进行缺失值填补
x_imp_raw <- 2^x_log_imp - offset  # 如果一开始加了 offset，这里减回去

rownames(x_imp_raw) <- rownames(x_raw)
colnames(x_imp_raw) <- colnames(x_raw)

write.csv(x_imp_raw, "interim_data/008-serum_lipidomics_feature_imputed.csv", row.names = FALSE)