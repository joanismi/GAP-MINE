
BiocManager::install("ropls")

library(foreach)
library(doParallel)


####CUSTOM FUNCTIONS####

oplsda.fs <- function(metrics, labels){
  myCluster <- makeCluster(4, # number of cores to use
                           type = "PSOCK") # type of cluster
  registerDoParallel(myCluster)
  fs <- foreach(column = 1:ncol(labels), .combine = 'rbind') %dopar% {
    library(ropls)
    library(caret)
    column_labels <- factor(labels[,column], levels=c(1,0))
    train.index <- unlist(createDataPartition(column_labels, p = .8, list = TRUE))
    all_labels  <- seq(1, nrow(labels))
    test.index <- setdiff(all_labels, train.index)
    if(length(test.index) == 3275){
      test.index <- append(test.index, NA)
    }
    process.oplsda <- try(opls(metrics, column_labels, orthoI = NA, predI=1, crossvalI = 10, subset=train.index, fig.pdfC = "none", info.txtC='none'), silent=TRUE)
    if("try-error" %in% class(process.oplsda)) {
      process.oplsda <- try(opls(metrics, column_labels, orthoI = 9, predI=1, crossvalI = 10, subset=train.index, fig.pdfC = "none", info.txtC='none'), silent=TRUE)
    } 
    if("try-error" %in% class(process.oplsda)) {
      process.oplsda <- opls(metrics, column_labels, orthoI = 1, predI=1, crossvalI = 10, subset=train.index, fig.pdfC = "none", info.txtC='none')}
    VIP <- getVipVn(process.oplsda)
    
    return(list(VIP, as.vector(test.index)))
  }
  stopCluster(myCluster)
  fs
}

####DATA LOAD####

#####PROCESS#####
####-------------Normal modules---------------------------------------####
reactome_labels <- read.csv('../python/data/processed/reactome_labels_string.csv', header=FALSE)
rwr <- read.csv('../python/data/processed/metrics/process_rwr_string.csv', row.names = 1)

#genePANDA <- read.csv('../python/data/processed/metrics/process_genePANDA_string.csv', row.names = 1)
#maxlink <- read.csv('../python/data/processed/metrics/process_maxlink_string.csv', row.names = 1)
#raw <- read.csv('../python/data/processed/metrics/process_raw_string.csv', row.names = 1)

row.names(reactome_labels) <- row.names(rwr)
colnames(reactome_labels) <- colnames(rwr)

####-------------False annotation modules---------------------------------------####
reactome_labels_fp <- read.csv('../python/data/processed/string_reactome_labels_fp.csv', header=FALSE)
rwr_fp <- read.csv('../python/data/processed/metrics/string_process_rwr_fp.csv', row.names = 1)

#raw_fp <- read.csv('../python/data/processed/metrics/process_raw_fp_string.csv', row.names = 1)

row.names(reactome_labels_fp) <- row.names(rwr_fp)
colnames(reactome_labels_fp) <- colnames(rwr_fp)

#####DISEASE#####
######SCA######

####-------------Normal modules---------------------------------------####
disgenet_labels <- read.csv('../python/data/processed/disgenet_sca_labels_string.csv', header=FALSE)
disease_rwr <- read.csv('../python/data/processed/metrics/disease_rwr_string.csv', row.names = 1)

#disease_genePANDA <- read.csv('../python/data/processed/metrics/disease_genePANDA_string.csv', row.names = 1)
#disease_maxlink <- read.csv('../python/data/processed/metrics/disease_maxlink_string.csv', row.names = 1)
#disease_raw <- read.csv('../python/data/processed/metrics/disease_raw_string.csv', row.names = 1)

row.names(disgenet_labels) <- row.names(disease_rwr)
colnames(disgenet_labels) <- colnames(disease_rwr)

####-------------False annotation modules---------------------------------------####
disgenet_labels_fp <- read.csv('../python/data/processed/string_disgenet_labels_fp.csv', header=FALSE)
disease_rwr_fp <- read.csv('../python/data/processed/metrics/string_disease_rwr_fp.csv', row.names = 1)

#disease_raw_fp <- read.csv('../python/data/processed/metrics/disease_raw_fp_string.csv', row.names = 1)

row.names(disgenet_labels_fp) <- row.names(disease_rwr_fp)
colnames(disgenet_labels_fp) <- colnames(disease_rwr_fp)

######CONSERVATIVE######

####-------------Normal modules---------------------------------------####
disgenet_labels_conservative <- read.csv('../python/data/processed/disgenet_conservative_labels_string.csv', header=FALSE)
disease_rwr_conservative <- read.csv('../python/data/processed/metrics/disease_rwr_conservative_string.csv', row.names = 1)

#disease_genePANDA_conservative <- read.csv('../python/data/processed/metrics/disease_conservative_genePANDA_string.csv', row.names = 1)
#disease_maxlink_conservative <- read.csv('../python/data/processed/metrics/disease_conservative_maxlink_string.csv', row.names = 1)
#disease_raw_conservative <- read.csv('../python/data/processed/metrics/disease_conservative_raw_string.csv', row.names = 1)

row.names(disgenet_labels_conservative) <- row.names(disease_rwr_conservative)
colnames(disgenet_labels_conservative) <- colnames(disease_rwr_conservative)

####-------------False annotation modules---------------------------------------####
disgenet_labels_conservative_fp <- read.csv('../python/data/processed/string_disgenet_conservative_labels_fp.csv', header=FALSE)
disease_rwr_conservative_fp <- read.csv('../python/data/processed/metrics/string_disease_rwr_conservative_fp.csv', row.names = 1)

#disease_raw_conservative_fp <- read.csv('../python/data/processed/metrics/disease_raw_fp_string.csv', row.names = 1)

row.names(disgenet_labels_conservative_fp) <- row.names(disease_rwr_conservative_fp)
colnames(disgenet_labels_conservative_fp) <- colnames(disease_rwr_conservative_fp)


####FEATURE SELECTION PROCESS####

####-------------Normal modules---------------------------------------####
rwr_fs <- oplsda.fs(rwr, reactome_labels)
write.csv(as.data.frame(rwr_fs[,1]),"../python/data/processed/fs/reactome_rwr_fs_string.csv", row.names = FALSE)
write.csv(as.data.frame(rwr_fs[,2]),"../python/data/processed/fs/reactome_rwr_test_string.csv", row.names = FALSE)

####-------------False annotation modules---------------------------------------####
rwr_fp_fs <- oplsda.fs(rwr_fp, reactome_labels_fp)
write.csv(as.data.frame(rwr_fp_fs[,1]),"../python/data/processed/fs/reactome_rwr_fp_fs_string.csv", row.names = FALSE)
write.csv(as.data.frame(rwr_fp_fs[,2]),"../python/data/processed/fs/reactome_rwr_fp_test_string.csv", row.names = FALSE)


####FEATURE SELECTION DISEASE####
#####SCA#####

####-------------Normal modules---------------------------------------####
disease_rwr_fs <- oplsda.fs(disease_rwr, disgenet_labels)
write.csv(as.data.frame(disease_rwr_fs[,1]),"../python/data/processed/fs/disease/disease_rwr_fs_string.csv", row.names = FALSE)
write.csv(as.data.frame(disease_rwr_fs[,2]),"../python/data/processed/fs/disease/disease_rwr_test_string.csv", row.names = FALSE)

####-------------False annotation modules---------------------------------------####
disease_rwr_fp_fs <- oplsda.fs(disease_rwr_fp, disgenet_labels_fp)
write.csv(as.data.frame(disease_rwr_fp_fs[,1]),"../python/data/processed/fs/disease/disease_rwr_fp_fs_string.csv", row.names = FALSE)
write.csv(as.data.frame(disease_rwr_fp_fs[,2]),"../python/data/processed/fs/disease/disease_rwr_fp_test_string.csv", row.names = FALSE)


#####CONSERVATIVE MODULE#####

####-------------Normal modules---------------------------------------####
disease_rwr_fs_conservative <- oplsda.fs(disease_rwr_conservative, disgenet_labels_conservative)
write.csv(as.data.frame(disease_rwr_fs_conservative[,1]),"../python/data/processed/fs/disease/disease_rwr_fs_conservative_string.csv", row.names = FALSE)
write.csv(as.data.frame(disease_rwr_fs_conservative[,2]),"../python/data/processed/fs/disease/disease_rwr_test_conservative_string.csv", row.names = FALSE)

####-------------False annotation modules---------------------------------------####
disease_rwr_fp_fs_conservative <- oplsda.fs(disease_rwr_conservative_fp, disgenet_labels_conservative_fp)
write.csv(as.data.frame(disease_rwr_fp_fs_conservative[,1]),"../python/data/processed/fs/disease/disease_rwr_fp_fs_conservative_string.csv", row.names = FALSE)
write.csv(as.data.frame(disease_rwr_fp_fs_conservative[,2]),"../python/data/processed/fs/disease/disease_rwr_fp_test_conservative_string.csv", row.names = FALSE)


#####Raw####
#raw_fs <- oplsda.fs(raw, reactome_labels)
#write.csv(as.data.frame(raw_fs[,1]),"../python/data/processed/fs/reactome_raw_fs_string.csv", row.names = FALSE)
#write.csv(as.data.frame(raw_fs[,2]),"../python/data/processed/fs/reactome_raw_test_string.csv", row.names = FALSE)

#raw_fp_fs <- oplsda.fs(raw_fp, reactome_labels_fp)
#write.csv(as.data.frame(raw_fp_fs[,1]),"../python/data/processed/fs/reactome_raw_fp_fs_string.csv", row.names = FALSE)
#write.csv(as.data.frame(raw_fp_fs[,2]),"../python/data/processed/fs/reactome_raw_fp_test_string.csv", row.names = FALSE)


#disease_raw_fs <- oplsda.fs(disease_raw, disgenet_labels)
#write.csv(as.data.frame(disease_raw_fs[,1]),"../python/data/processed/fs/disease/disease_raw_fs_string.csv", row.names = FALSE)
#write.csv(as.data.frame(disease_raw_fs[,2]),"../python/data/processed/fs/disease/disease_raw_test_string.csv", row.names = FALSE)

#disease_raw_fp_fs <- oplsda.fs(disease_raw_fp, disgenet_labels_fp)
#write.csv(as.data.frame(disease_raw_fp_fs[,1]),"../python/data/processed/fs/disease/disease_raw_fp_fs_string.csv", row.names = FALSE)
#write.csv(as.data.frame(disease_raw_fp_fs[,2]),"../python/data/processed/fs/disease/disease_raw_fp_test_string.csv", row.names = FALSE)


#disease_raw_fs_conservative <- oplsda.fs(disease_raw_conservative, disgenet_labels_conservative)
#write.csv(as.data.frame(disease_raw_fs_conservative[,1]),"../python/data/processed/fs/disease/disease_raw_fs_conservative_string.csv", row.names = FALSE)
#write.csv(as.data.frame(disease_raw_fs_conservative[,2]),"../python/data/processed/fs/disease/disease_raw_test_conservative_string.csv", row.names = FALSE)

#disease_raw_fp_fs_conservative <- oplsda.fs(disease_raw_conservative_fp, disgenet_labels_conservative_fp)
#write.csv(as.data.frame(disease_raw_fp_fs_conservative[,1]),"../python/data/processed/fs/disease/disease_raw_fp_fs_conservative_string.csv", row.names = FALSE)
#write.csv(as.data.frame(disease_raw_fp_fs_conservative[,2]),"../python/data/processed/fs/disease/disease_raw_fp_test_conservative_string.csv", row.names = FALSE)
