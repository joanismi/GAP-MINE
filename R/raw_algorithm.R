if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("diffuStats")


library(diffuStats)
library(ropls)
library(igraph)
library(foreach)
library(doPararel)
library(caret)
library(stringr)
library(reticulate)
source("oplsda_fs.R")
np <- import('numpy')
library(tidyr)


reactome_proteins_indexes <- read.csv('../python/data/processed/reactome_proteins_indexes_apid_huri.csv')
ppi_graph <- read_graph("../python/data/processed/graph_apid_huri", format='gml')
protein80 <- np$load("../python/data/processed/ppis/ppis_red_protein80_apid_huri.npy", allow_pickle = TRUE)
ppi_80 <- np$load("../python/data/processed/ppis/ppis_red80_apid_huri.npy", allow_pickle = TRUE)

reactome_labels <- read.csv('../python/data/processed/reactome_labels_apid_huri.csv', header=FALSE)
reactome_labels_fp <- read.csv('../python/data/processed/reactome_labels_fp.csv', header=FALSE)
disgenet_labels <- read.csv('../python/data/processed/disgenet_filtered_labels_apid_huri.csv', header=FALSE)
disgenet_conservative_labels <- read.csv('../python/data/processed/disgenet_conservative_labels_apid_huri.csv', header=FALSE)
disgenet_labels_fp <- read.csv('../python/data/processed/disgenet_labels_fp.csv', header=FALSE)
disgenet_conservative_labels_fp <- read.csv('../python/data/processed/disgenet_conservative_labels_fp.csv', header=FALSE)
colnames(reactome_labels) <- reactome_proteins_indexes[['process']]
rownames(reactome_labels) <- V(ppi_graph)$name
rownames(reactome_labels_fp) <- V(ppi_graph)$name
rownames(disgenet_labels) <- V(ppi_graph)$name
rownames(disgenet_conservative_labels) <- V(ppi_graph)$name
rownames(disgenet_labels_fp) <- V(ppi_graph)$name
rownames(disgenet_conservative_labels_fp) <- V(ppi_graph)$name

kernel_scores <- regularisedLaplacianKernel(ppi_graph)

####PROCESS####

process_raw_scores <- matrix(0,length(reactome_proteins_indexes[['process']]) , length(V(ppi_graph)$name))
rownames(process_raw_scores) <- reactome_proteins_indexes[['process']]
colnames(process_raw_scores) <- V(ppi_graph)$name

for(i in 1:length(reactome_proteins_indexes[['process']])){
  module_scores <- reactome_labels[,i]
  names(module_scores) <- rownames(reactome_labels)
  df_diff <- diffuse_grid(graph=ppi_graph,
                          K = kernel_scores,
                          scores = reactome_labels[,i],
                          grid_param = expand.grid(method = c('raw')),
                          n.perm = 1000)
  process_raw_scores[i,] = df_diff$node_score
}
write.csv(t(process_raw_scores),"../python/data/processed/metrics/process_raw.csv")

process_raw_scores <- read.csv('../python/data/processed/metrics/process_raw.csv', row.names = 1)

process_raw_fs <- oplsda.fs(process_raw_scores, reactome_labels)
write.csv(as.data.frame(process_raw_fs[,1]),"../python/data/processed/fs/process_raw_fs.csv", row.names = FALSE)
write.csv(as.data.frame(process_raw_fs[,2]),"../python/data/processed/fs/process_raw_test.csv", row.names = FALSE)

####DISEASE####

disease_raw_scores <- matrix(0, dim(disgenet_labels)[2] , length(V(ppi_graph)$name))
colnames(disease_raw_scores) <- V(ppi_graph)$name


for(i in 1:dim(disgenet_labels)[2]){
  module_scores <- disgenet_labels[,i]
  names(module_scores) <- rownames(disgenet_labels)
  df_diff <- diffuse_grid(graph=ppi_graph,
                          K = kernel_scores,
                          scores = module_scores,
                          grid_param = expand.grid(method = c('raw')),
                          n.perm = 1000)
  disease_raw_scores[i,] = df_diff$node_score
}
write.csv(t(disease_raw_scores),"../python/data/processed/metrics/disease_raw.csv", row.names = FALSE)

disease_raw_fs <- oplsda.fs(t(disease_raw_scores), disgenet_labels)
write.csv(as.data.frame(disease_raw_fs[,1]),"../python/data/processed/fs/disease_raw_fs.csv", row.names = FALSE)
write.csv(as.data.frame(disease_raw_fs[,2]),"../python/data/processed/fs/disease_raw_test.csv", row.names = FALSE)


####DISEASE CONSERVATIVE####

disease_conservative_raw_scores <- matrix(0, dim(disgenet_conservative_labels)[2] , length(V(ppi_graph)$name))
colnames(disease_conservative_raw_scores) <- V(ppi_graph)$name

for(i in 1:dim(disgenet_conservative_labels)[2]){
  module_scores <- disgenet_conservative_labels[,i]
  names(module_scores) <- rownames(disgenet_conservative_labels)
  df_diff <- diffuse_grid(graph=ppi_graph,
                          K = kernel_scores,
                          scores = module_scores,
                          grid_param = expand.grid(method = c('raw')),
                          n.perm = 1000)
  disease_conservative_raw_scores[i,] = df_diff$node_score
}
write.csv(t(disease_conservative_raw_scores),"../python/data/processed/metrics/disease_conservative_raw.csv", row.names = FALSE)

disease_conservative_raw_fs <- oplsda.fs(t(disease_conservative_raw_scores), disgenet_conservative_labels)
write.csv(as.data.frame(disease_conservative_raw_fs[,1]),"../python/data/processed/fs/disease_conservative_raw_fs.csv", row.names = FALSE)
write.csv(as.data.frame(disease_conservative_raw_fs[,2]),"../python/data/processed/fs/disease_conservative_raw_test.csv", row.names = FALSE)


####PROCESS FP####
process_fp_raw_scores <- matrix(0,dim(reactome_labels_fp)[2] , length(V(ppi_graph)$name))
colnames(process_fp_raw_scores) <- V(ppi_graph)$name

for(i in 1:dim(reactome_labels_fp)[2]){
  module_scores <- reactome_labels_fp[,i]
  names(module_scores) <- rownames(reactome_labels_fp)
  df_diff <- diffuse_grid(graph=ppi_graph,
                          K = kernel_scores,
                          scores = module_scores,
                          grid_param = expand.grid(method = c('raw')),
                          n.perm = 1000)
  process_fp_raw_scores[i,] = df_diff$node_score
}
write.csv(t(process_fp_raw_scores),"../python/data/processed/metrics/process_fp_raw.csv", row.names = FALSE)
process_fp_raw_scores <- read.csv('../python/data/processed/metrics/process_fp_raw.csv')

process_fp_raw_fs <- oplsda.fs(process_fp_raw_scores, reactome_labels_fp)
write.csv(as.data.frame(process_fp_raw_fs[,1]),"../python/data/processed/fs/process_fp_raw_fs.csv", row.names = FALSE)
write.csv(as.data.frame(process_fp_raw_fs[,2]),"../python/data/processed/fs/process_fp_raw_test.csv", row.names = FALSE)

####DISEASE####
disease_fp_raw_scores <- matrix(0,dim(disgenet_labels_fp)[2] , length(V(ppi_graph)$name))
colnames(disease_fp_raw_scores) <- V(ppi_graph)$name

for(i in 1:dim(disgenet_labels_fp)[2]){
  module_scores <- disgenet_labels_fp[,i]
  names(module_scores) <- rownames(disgenet_labels_fp)
  df_diff <- diffuse_grid(graph=ppi_graph,
                          K = kernel_scores,
                          scores = module_scores,
                          grid_param = expand.grid(method = c('raw')),
                          n.perm = 1000)
  disease_fp_raw_scores[i,] = df_diff$node_score
}
write.csv(t(disease_fp_raw_scores),"../python/data/processed/metrics/disease_fp_raw.csv", row.names = FALSE)
disease_fp_raw_scores <- read.csv('../python/data/processed/metrics/disease_fp_raw.csv')

disease_fp_raw_fs <- oplsda.fs(disease_fp_raw_scores, disgenet_labels_fp)
write.csv(as.data.frame(disease_fp_raw_fs[,1]),"../python/data/processed/fs/disease_fp_raw_fs.csv", row.names = FALSE)
write.csv(as.data.frame(disease_fp_raw_fs[,2]),"../python/data/processed/fs/disease_fp_raw_test.csv", row.names = FALSE)

####DISEASE CONSERVATIVE####
disease_conservative_fp_raw_scores <- matrix(0,dim(disgenet_conservative_labels_fp)[2] , length(V(ppi_graph)$name))
colnames(disease_conservative_fp_raw_scores) <- V(ppi_graph)$name

for(i in 1:dim(disgenet_conservative_labels_fp)[2]){
  module_scores <- disgenet_conservative_labels_fp[,i]
  names(module_scores) <- rownames(disgenet_conservative_labels_fp)
  df_diff <- diffuse_grid(graph=ppi_graph,
                          K = kernel_scores,
                          scores = module_scores,
                          grid_param = expand.grid(method = c('raw')),
                          n.perm = 1000)
  disease_conservative_fp_raw_scores[i,] = df_diff$node_score
}
write.csv(t(disease_conservative_fp_raw_scores),"../python/data/processed/metrics/disease_conservative_fp_raw.csv", row.names = FALSE)
disease_conservative_fp_raw_scores <- read.csv('../python/data/processed/metrics/disease_conservative_fp_raw.csv')

disease_conservative_fp_raw_fs <- oplsda.fs(disease_conservative_fp_raw_scores, disgenet_conservative_labels_fp)
write.csv(as.data.frame(disease_conservative_fp_raw_fs[,1]),"../python/data/processed/fs/disease_conservative_fp_raw_fs.csv", row.names = FALSE)
write.csv(as.data.frame(disease_conservative_fp_raw_fs[,2]),"../python/data/processed/fs/disease_conservative_fp_raw_test.csv", row.names = FALSE)



####PROTEIN 80####

for(i in 1:dim(protein80)[1]){
  protein80_df <- as.data.frame(protein80[i,,1:2])
  protein80_df[protein80_df==0]<-NA
  protein80_df <- protein80_df[!is.na(protein80_df$V1) & !is.na(protein80_df$V2),]
  protein80_df <- protein80_df[(protein80_df$V1 %in% rownames(reactome_labels)) & (protein80_df$V2 %in% rownames(reactome_labels)),]
  protein80_graph <- graph_from_data_frame(protein80_df)
  kernel_scores_protein80 <- regularisedLaplacianKernel(protein80_graph)
  save(kernel_scores_protein80 ,paste("kernel", i,'.Rda', sep=""))
  }   


for (i in 1:10){
  file_ <- paste("kernel", i,'.Rda', sep="")
  kernel_1 <- load(file_)
  protein80_df <- as.data.frame(protein80[i,,1:2])
  protein80_df[protein80_df==0]<-NA
  protein80_df <- protein80_df[!is.na(protein80_df$V1) & !is.na(protein80_df$V2),]
  protein80_df <- protein80_df[(protein80_df$V1 %in% rownames(reactome_labels)) & (protein80_df$V2 %in% rownames(reactome_labels)),]
  protein80_graph <- graph_from_data_frame(protein80_df)
  
  reactome_protein80_scores <- matrix(0,dim(reactome_labels)[2] , length(V(protein80_graph)$name))
  colnames(reactome_protein80_scores) <- V(protein80_graph)$name
  
  
  for(j in 1:length(reactome_proteins_indexes[['process']])){
    module_scores <- reactome_labels[rownames(kernel_scores_protein80),j]
    names(module_scores) <- rownames(reactome_labels[rownames(kernel_scores_protein80),])
    df_diff <- diffuse_grid(graph=protein80_graph,
                            K = kernel_scores_protein80,
                            scores = module_scores,
                            grid_param = expand.grid(method = c('raw')),
                            n.perm = 1000)
    reactome_protein80_scores[j,] = df_diff$node_score
    cat(sprintf("\"%f\" \"%f\"\n", i, j))
  }
  file_destination <- paste("../python/data/processed/metrics/reactome_protein80_", i, ".csv", sep="")
  write.csv(t(reactome_protein80_scores),file_destination, row.names = FALSE)
}


for (i in 1:10){
  file_ <- paste("kernel", i,'.Rda', sep="")
  kernel_1 <- load(file_)
  protein80_df <- as.data.frame(protein80[i,,1:2])
  protein80_df[protein80_df==0]<-NA
  protein80_df <- protein80_df[!is.na(protein80_df$V1) & !is.na(protein80_df$V2),]
  protein80_df <- protein80_df[(protein80_df$V1 %in% rownames(disgenet_labels)) & (protein80_df$V2 %in% rownames(disgenet_labels)),]
  protein80_graph <- graph_from_data_frame(protein80_df)
  
  disgenet_protein80_scores <- matrix(0,dim(disgenet_labels)[2] , length(V(protein80_graph)$name))
  colnames(disgenet_protein80_scores) <- V(protein80_graph)$name
  
  
  for(j in 1:dim(disgenet_labels)[2]){
    module_scores <- disgenet_labels[rownames(kernel_scores_protein80),j]
    names(module_scores) <- rownames(disgenet_labels[rownames(kernel_scores_protein80),])
    df_diff <- diffuse_grid(graph=protein80_graph,
                            K = kernel_scores_protein80,
                            scores = module_scores,
                            grid_param = expand.grid(method = c('raw')),
                            n.perm = 1000)
    disgenet_protein80_scores[j,] = df_diff$node_score
    cat(sprintf("\"%f\" \"%f\"\n", i, j))
  }
  file_destination <- paste("../python/data/processed/metrics/disease_protein80_", i, ".csv", sep="")
  write.csv(t(disgenet_protein80_scores),file_destination, row.names = FALSE)
}


for (i in 1:10){
  file_ <- paste("kernel", i,'.Rda', sep="")
  kernel_1 <- load(file_)
  protein80_df <- as.data.frame(protein80[i,,1:2])
  protein80_df[protein80_df==0]<-NA
  protein80_df <- protein80_df[!is.na(protein80_df$V1) & !is.na(protein80_df$V2),]
  protein80_df <- protein80_df[(protein80_df$V1 %in% rownames(disgenet_conservative_labels)) & (protein80_df$V2 %in% rownames(disgenet_conservative_labels)),]
  protein80_graph <- graph_from_data_frame(protein80_df)
  
  disgenet_conservative_protein80_scores <- matrix(0,dim(disgenet_conservative_labels)[2] , length(V(protein80_graph)$name))
  colnames(disgenet_conservative_protein80_scores) <- V(protein80_graph)$name
  
  
  for(j in 1:dim(disgenet_conservative_labels)[2]){
    module_scores <- disgenet_conservative_labels[rownames(kernel_scores_protein80),j]
    names(module_scores) <- rownames(disgenet_conservative_labels[rownames(kernel_scores_protein80),])
    df_diff <- diffuse_grid(graph=protein80_graph,
                            K = kernel_scores_protein80,
                            scores = module_scores,
                            grid_param = expand.grid(method = c('raw')),
                            n.perm = 1000)
    disgenet_conservative_protein80_scores[j,] = df_diff$node_score
    cat(sprintf("\"%f\" \"%f\"\n", i, j))
  }
  file_destination <- paste("../python/data/processed/metrics/disease_conservative_protein80_", i, ".csv", sep="")
  write.csv(t(disgenet_conservative_protein80_scores),file_destination, row.names = FALSE)
}



#####PPI 80

for(i in 1:dim(ppi_80)[1]){
  ppi80_df <- as.data.frame(ppi_80[i,,1:2])
  ppi80_df[ppi80_df==0]<-NA
  ppi80_df <- ppi80_df[!is.na(ppi80_df$V1) & !is.na(ppi80_df$V2),]
  ppi80_df <- ppi80_df[(ppi80_df$V1 %in% rownames(reactome_labels)) & (ppi80_df$V2 %in% rownames(reactome_labels)),]
  ppi80_graph <- graph_from_data_frame(ppi80_df, directed = FALSE)
  kernel_scores_ppi80 <- regularisedLaplacianKernel(ppi80_graph)
  save(kernel_scores_ppi80 ,file=paste("kernel_ppi80_", i,'.Rda', sep=""))
}   


for (i in 1:10){
  file_ <- paste("kernel_ppi80_", i,'.Rda', sep="")
  kernel_1 <- load(file_)
  ppi80_df <- as.data.frame(ppi_80[i,,1:2])
  ppi80_df[ppi80_df==0]<-NA
  ppi80_df <- ppi80_df[!is.na(ppi80_df$V1) & !is.na(ppi80_df$V2),]
  ppi80_df <- ppi80_df[(ppi80_df$V1 %in% rownames(reactome_labels)) & (ppi80_df$V2 %in% rownames(reactome_labels)),]
  ppi80_graph <- graph_from_data_frame(ppi80_df)
  
  reactome_ppi80_scores <- matrix(0,dim(reactome_labels)[2] , length(V(ppi80_graph)$name))
  colnames(reactome_ppi80_scores) <- V(ppi80_graph)$name
  
  
  for(j in 1:length(reactome_proteins_indexes[['process']])){
    module_scores <- reactome_labels[rownames(kernel_scores_ppi80),j]
    names(module_scores) <- rownames(reactome_labels[rownames(kernel_scores_ppi80),])
    df_diff <- diffuse_grid(graph=ppi80_graph,
                            K = kernel_scores_ppi80,
                            scores = module_scores,
                            grid_param = expand.grid(method = c('raw')),
                            n.perm = 1000)
    reactome_ppi80_scores[j,] = df_diff$node_score
    cat(sprintf("\"%f\" \"%f\"\n", i, j))
  }
  file_destination <- paste("../python/data/processed/metrics/reactome_ppi80_", i, ".csv", sep="")
  write.csv(t(reactome_ppi80_scores),file_destination, row.names = FALSE)
}


for (i in 1:10){
  file_ <- paste("kernel_ppi80_", i,'.Rda', sep="")
  kernel_1 <- load(file_)
  ppi80_df <- as.data.frame(ppi_80[i,,1:2])
  ppi80_df[ppi80_df==0]<-NA
  ppi80_df <- ppi80_df[!is.na(ppi80_df$V1) & !is.na(ppi80_df$V2),]
  ppi80_df <- ppi80_df[(ppi80_df$V1 %in% rownames(disgenet_labels)) & (ppi80_df$V2 %in% rownames(disgenet_labels)),]
  ppi80_graph <- graph_from_data_frame(ppi80_df)
  
  disgenet_ppi80_scores <- matrix(0,dim(disgenet_labels)[2] , length(V(ppi80_graph)$name))
  colnames(disgenet_ppi80_scores) <- V(ppi80_graph)$name
  
  
  for(j in 1:dim(disgenet_labels)[2]){
    module_scores <- disgenet_labels[rownames(kernel_scores_ppi80),j]
    names(module_scores) <- rownames(disgenet_labels[rownames(kernel_scores_ppi80),])
    df_diff <- diffuse_grid(graph=ppi80_graph,
                            K = kernel_scores_ppi80,
                            scores = module_scores,
                            grid_param = expand.grid(method = c('raw')),
                            n.perm = 1000)
    disgenet_ppi80_scores[j,] = df_diff$node_score
    cat(sprintf("\"%f\" \"%f\"\n", i, j))
  }
  file_destination <- paste("../python/data/processed/metrics/disease_ppi80_", i, ".csv", sep="")
  write.csv(t(disgenet_ppi80_scores),file_destination, row.names = FALSE)
}


for (i in 1:10){
  file_ <- paste("kernel_ppi80_", i,'.Rda', sep="")
  kernel_1 <- load(file_)
  ppi80_df <- as.data.frame(ppi_80[i,,1:2])
  ppi80_df[ppi80_df==0]<-NA
  ppi80_df <- ppi80_df[!is.na(ppi80_df$V1) & !is.na(ppi80_df$V2),]
  ppi80_df <- ppi80_df[(ppi80_df$V1 %in% rownames(disgenet_conservative_labels)) & (ppi80_df$V2 %in% rownames(disgenet_conservative_labels)),]
  ppi80_graph <- graph_from_data_frame(ppi80_df)
  
  disgenet_conservative_ppi80_scores <- matrix(0,dim(disgenet_conservative_labels)[2] , length(V(ppi80_graph)$name))
  colnames(disgenet_conservative_ppi80_scores) <- V(ppi80_graph)$name
  
  
  for(j in 1:dim(disgenet_conservative_labels)[2]){
    module_scores <- disgenet_conservative_labels[rownames(kernel_scores_ppi80),j]
    names(module_scores) <- rownames(disgenet_conservative_labels[rownames(kernel_scores_ppi80),])
    df_diff <- diffuse_grid(graph=ppi80_graph,
                            K = kernel_scores_ppi80,
                            scores = module_scores,
                            grid_param = expand.grid(method = c('raw')),
                            n.perm = 1000)
    disgenet_conservative_ppi80_scores[j,] = df_diff$node_score
    cat(sprintf("\"%f\" \"%f\"\n", i, j))
  }
  file_destination <- paste("../python/data/processed/metrics/disease_conservative_ppi80_", i, ".csv", sep="")
  write.csv(t(disgenet_conservative_ppi80_scores),file_destination, row.names = FALSE)
}
