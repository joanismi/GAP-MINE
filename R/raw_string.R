library(igraph)
library(diffuStats)
library(foreach)
library(doParallel)

process_modules <- read.csv('../python/data/processed/string_reactome_modules.csv')
disease_modules <- read.csv('../python/data/processed/string_disgenet_modules.csv')
process_modules_fp <- read.csv('../python/data/processed/string_reactome_modules_fp.csv')
disease_modules_fp <- read.csv('../python/data/processed/string_disgenet_sca_modules_fp.csv')
disease_conservative_modules_fp <- read.csv('../python/data/processed/string_disgenet_conservative_modules_fp.csv')
ppi_graph <- read_graph("../python/data/processed/graph_string", format='gml')

reactome_labels <- read.csv('../python/data/processed/reactome_labels_string.csv', header=FALSE)
disgenet_labels <- read.csv('../python/data/processed/disgenet_sca_labels_string.csv', header=FALSE)
disgenet_conservative_labels <- read.csv('../python/data/processed/disgenet_conservative_labels_string.csv', header=FALSE)
reactome_labels_fp <- read.csv('../python/data/processed/string_reactome_labels_fp.csv', header=FALSE)
disgenet_labels_fp <- read.csv('../python/data/processed/string_disgenet_labels_fp.csv', header=FALSE)
disgenet_conservative_labels_fp <- read.csv('../python/data/processed/string_disgenet_conservative_labels_fp.csv', header=FALSE)
colnames(reactome_labels) <- process_modules[['process']]
rownames(reactome_labels) <- V(ppi_graph)$name
rownames(disgenet_labels) <- V(ppi_graph)$name
rownames(disgenet_conservative_labels) <- V(ppi_graph)$name

rownames(reactome_labels_fp) <- V(ppi_graph)$name
rownames(disgenet_labels_fp) <- V(ppi_graph)$name
rownames(disgenet_conservative_labels_fp) <- V(ppi_graph)$name

kernel_scores <- regularisedLaplacianKernel(ppi_graph)

####PROCESS####

process_raw_scores <- matrix(0,length(process_modules[['process']]) , length(V(ppi_graph)$name))
rownames(process_raw_scores) <- process_modules[['process']]
colnames(process_raw_scores) <- V(ppi_graph)$name


for(i in 1:length(process_modules[['process']])){
  module_scores <- reactome_labels[,i]
  names(module_scores) <- rownames(reactome_labels)
  seleted_scores <- reactome_labels[,i]
  names(seleted_scores) <- rownames(reactome_labels)
  df_diff <- diffuse_grid(graph=ppi_graph,
                          K = kernel_scores,
                          scores = seleted_scores,
                          grid_param = expand.grid(method = c('raw')),
                          n.perm = 1000)
  process_raw_scores[i,] = df_diff$node_score
}
write.csv(t(process_raw_scores),"../python/data/processed/metrics/process_raw_string.csv")

process_raw_scores <- read.csv('../python/data/processed/metrics/process_raw_string.csv', row.names = 1)

process_raw_fs <- oplsda.fs(process_raw_scores, reactome_labels)
write.csv(as.data.frame(process_raw_fs[,1]),"../python/data/processed/fs/process_raw_fs_string.csv", row.names = FALSE)
write.csv(as.data.frame(process_raw_fs[,2]),"../python/data/processed/fs/process_raw_test_string.csv", row.names = FALSE)


process_raw_scores_fp <- matrix(0,length(process_modules_fp[['process']]) , length(V(ppi_graph)$name))
rownames(process_raw_scores_fp) <- process_modules_fp[['process']]
colnames(process_raw_scores_fp) <- V(ppi_graph)$name


for(i in 1:length(process_modules_fp[['process']])){
  module_scores <- reactome_labels_fp[,i]
  names(module_scores) <- rownames(reactome_labels_fp)
  seleted_scores <- reactome_labels_fp[,i]
  names(seleted_scores) <- rownames(reactome_labels_fp)
  df_diff <- diffuse_grid(graph=ppi_graph,
                          K = kernel_scores,
                          scores = seleted_scores,
                          grid_param = expand.grid(method = c('raw')),
                          n.perm = 1000)
  process_raw_scores_fp[i,] = df_diff$node_score
}
write.csv(t(process_raw_scores_fp),"../python/data/processed/metrics/process_raw_fp_string.csv")

####DISEASE SCA####

disease_raw_scores_fp <- matrix(0,length(disease_modules_fp[['process']]) , length(V(ppi_graph)$name))
rownames(disease_raw_scores_fp) <- disease_modules_fp[['process']]
colnames(disease_raw_scores_fp) <- V(ppi_graph)$name

for(i in 1:length(disease_modules_fp[['process']])){
  module_scores <- disgenet_labels_fp[,i]
  names(module_scores) <- rownames(disgenet_labels_fp)
  seleted_scores <- disgenet_labels_fp[,i]
  names(seleted_scores) <- rownames(disgenet_labels_fp)
  df_diff <- diffuse_grid(graph=ppi_graph,
                          K = kernel_scores,
                          scores = seleted_scores,
                          grid_param = expand.grid(method = c('raw')),
                          n.perm = 1000)
  disease_raw_scores_fp[i,] = df_diff$node_score
}
write.csv(t(disease_raw_scores_fp),"../python/data/processed/metrics/disease_raw_fp_string.csv")


####DISEASE CONSERVATIVE####

disease_conservative_raw_scores_fp <- matrix(0,length(disease_conservative_modules_fp[['process']]) , length(V(ppi_graph)$name))
rownames(disease_conservative_raw_scores_fp) <- disease_conservative_modules_fp[['process']]
colnames(disease_conservative_raw_scores_fp) <- V(ppi_graph)$name


for(i in 1:length(disease_conservative_modules_fp[['process']])){
  module_scores <- disgenet_conservative_labels_fp[,i]
  names(module_scores) <- rownames(disgenet_conservative_labels_fp)
  seleted_scores <- disgenet_conservative_labels_fp[,i]
  names(seleted_scores) <- rownames(disgenet_conservative_labels_fp)
  df_diff <- diffuse_grid(graph=ppi_graph,
                          K = kernel_scores,
                          scores = seleted_scores,
                          grid_param = expand.grid(method = c('raw')),
                          n.perm = 1000)
  disease_conservative_raw_scores_fp[i,] = df_diff$node_score
}
write.csv(t(disease_conservative_raw_scores_fp),"../python/data/processed/metrics/disease_conservative_raw_fp_string.csv")
