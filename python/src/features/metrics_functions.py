from multiprocessing import process
from os import close
import pandas as pd
import numpy as np
import math
from tqdm.notebook import tqdm
from scipy.stats import hypergeom
from collections import Counter
from itertools import combinations
from igraph import Graph
from itertools import chain
from ast import literal_eval
import sys


def hypergeometric_test(graph, proteins_by_process_df, adjacency_matrix):
    # Computation of the negative log10 p-value given by an hypergeometric test.
    #
    # INPUT:
    #   - graph with the PPI network
    #   - dataframe with proteins of each process/disease
    #   - adjacency matrix of the graph
    #
    # RETURNS: dataframe with -log10(p-value) values for every protein in every process/disease.
    proteins_neglogp_byprocess = {}
    protein_indices = graph.vs.indices
    protein_names = graph.vs['name']
    n_proteins = len(protein_names)
    proteins_by_process_dict = proteins_by_process_df.to_dict('index')
    for index in tqdm(protein_indices):
        protein = protein_names[index]
        processes_neglogp = protein_hypertests(
            index, proteins_by_process_dict, adjacency_matrix, n_proteins)
        proteins_neglogp_byprocess[protein] = processes_neglogp
    return proteins_neglogp_byprocess


def protein_hypertests(protein_index, proteins_by_process_dict, adjacency_matrix, n_proteins):
    # Computation of the negative log10 p-value given by an hypergeometric test for a given protein in all processes/diseases.
    #
    # INPUT:
    #   - index of the query protein
    #   - dict with proteins present in each process/disease
    #   - adjacency matrix of the graph
    #   - number of proteins in the network
    #
    # RETURNS: dict with -log10(p-value) values for the target protein in every process/disease.
    protein_process_neglogpvalue = {}
    protein_row = adjacency_matrix[protein_index]
    n_ppi_target_protein = sum(protein_row)
    for index, values in proteins_by_process_dict.items():
        num_neighbors = 0
        process_protein_indexes = list(values['protein_index'])
        process_n_proteins = values['n_proteins']
        for process_protein_index in process_protein_indexes:
            if process_protein_index > adjacency_matrix.shape[0]:
                continue
            if protein_row[process_protein_index] == 1:
                num_neighbors += 1
        neg_logp = -math.log(hypergeom.pmf(num_neighbors,
                             n_proteins-1, process_n_proteins, n_ppi_target_protein))
        protein_process_neglogpvalue[values['process']
                                     ] = neg_logp
    return protein_process_neglogpvalue


def shortest_paths(graph, processes_df):
    # Computation of the shortest paths betweeen every protein and a process/disease protein.
    #
    # INPUT:
    #   - graph with the PPI network
    #   - dataframe with processes/diseases and their respective proteins.
    #
    # RETURNS: dataframe with shortest paths between any protein (rows) and a process/disease proteins (columns).
    proteins_shortest_paths = {}
    all_process_proteins = list(set(
        [protein for protein_list in processes_df['proteins_ids'].values for protein in protein_list if protein in graph.vs['name']]))
    target_proteins = graph.vs['name']
    for protein in all_process_proteins:
        if protein not in target_proteins:
            all_process_proteins.remove(protein)
    shortest_paths = graph.shortest_paths(
        source=all_process_proteins, target=target_proteins, mode='all')
    for source_protein_index in tqdm(range(len(all_process_proteins))):
        source_protein = all_process_proteins[source_protein_index]
        proteins_shortest_paths[source_protein] = {}
        for target_protein_index in range(len(target_proteins)):
            target_protein = target_proteins[target_protein_index]
            short_path_len = shortest_paths[source_protein_index][target_protein_index]
            proteins_shortest_paths[source_protein][target_protein] = short_path_len
    process_shortest_paths_df = pd.DataFrame.from_dict(proteins_shortest_paths)
    process_shortest_paths_df = process_shortest_paths_df.rename(
        index=dict(zip(list(process_shortest_paths_df.index), graph.vs['name'])))
    return process_shortest_paths_df


def closeness(shortest_paths_df, process_df):
    # Computation of the closeness of every protein to the proteins of each process/disease.
    #
    # INPUT:
    #   - dataframe of shortest paths
    #   - dataframe with processes/diseases and their respective proteins.
    #
    # RETURNS: dataframe with closeness of a protein in each process/disease.
    shortest_paths = shortest_paths_df.to_dict('dict')
    process_proteins_closeness = {}
    for index, process in tqdm(process_df.iterrows(), total=process_df.shape[0]):
        n_process_proteins = len(process['proteins_ids'])
        process_id = process['process']
        process_proteins = {
            proteins: shortest_paths[proteins] for proteins in process['proteins_ids'] if proteins in shortest_paths_df.index}
        process_proteins_values = process_proteins.values()
        shortest_sum = Counter()
        for protein in process_proteins_values:
            shortest_sum.update(protein)
        shortest_sum = dict(shortest_sum)
        #closeness = {protein: short_sum / n_process_proteins for protein, short_sum in shortest_sum.items()}
        closeness = {protein: n_process_proteins /
                     short_sum for protein, short_sum in shortest_sum.items()}
        process_proteins_closeness[process_id] = closeness
    return process_proteins_closeness


def betweenness(shortest_paths, process_df, graph):
    # Computation of the binary betweenness of every protein to the proteins of each process/disease.
    # For every shortest paths between two proteins of the same process, checks what are the middle proteins (connecting both proteins) and keeps track of the amount of times they connect to process proteins.
    #
    # INPUT:
    #   - dataframe of shortest paths
    #   - dataframe with processes/diseases and their respective proteins.
    #   - graph with PPIs
    #
    # RETURNS: dataframe with betweenness of a protein in each process/disease.
    process_proteins_betweenness = {}
    for index, process in tqdm(process_df.iterrows(), total=process_df.shape[0]):
        process_proteins = [
            protein for protein in process['proteins_ids'] if protein in shortest_paths.index]
        process_id = process['process']
        process_proteins_pairs = combinations(process_proteins, 2)
        n_possible_paths = int(
            (len(process_proteins)*(len(process_proteins)-1))/2)
        process_proteins_betweenness[process_id] = {}
        shortest_sum = Counter()
        for pair in process_proteins_pairs:
            pair_shortest_path = shortest_paths.at[pair[0], pair[1]]
            intermediate_paths = shortest_paths.loc[:, [
                pair[0], pair[1]]].sum(axis=1, skipna=False)
            middle_proteins = list(
                intermediate_paths[intermediate_paths == pair_shortest_path].index)
            shortest_sum.update(middle_proteins)
        process_proteins_betweenness[process_id] = {protein: short_sum /
                                                    n_possible_paths for protein, short_sum in shortest_sum.items()}
        for protein in graph.vs['name']:
            if protein not in process_proteins_betweenness[process_id].keys():
                process_proteins_betweenness[process_id][protein] = 0
    return process_proteins_betweenness


def fraction_betweenness(processes_df, graph):
    # Computation of the binary betweenness of every protein to the proteins of each process/disease.
    # For every shortest paths between two proteins of the same process, checks what are the middle proteins (connecting both proteins) and keeps track of the amount of times they connect to process proteins.
    #
    # INPUT:
    #   - dataframe of shortest paths
    #   - dataframe with processes/diseases and their respective proteins.
    #   - graph with PPIs
    #
    # RETURNS: dataframe with betweenness of a protein in each process/disease.
    try:
        processes_df['protein_index'] = processes_df['protein_index'].apply(
            literal_eval)
        processes_df['proteins_ids'] = processes_df['proteins_ids'].apply(
            literal_eval)
    except ValueError:
        processes_df['protein_index'] = processes_df['protein_index']
        processes_df['proteins_ids'] = processes_df['proteins_ids']
    #processes_df = processes_df.iloc[:3,:]
    process_proteins = list(set(
        [protein for protein_list in processes_df['proteins_ids'].values for protein in protein_list if protein in graph.vs['name']]))
    n_combinations = int((len(process_proteins)*(len(process_proteins)-1))/2)
    processes_by_proteins = {}
    for k, v in processes_df[['process', 'proteins_ids']].set_index('process').to_dict('dict')['proteins_ids'].items():
        for x in v:
            if x in graph.vs['name']:
                processes_by_proteins.setdefault(
                    int(graph.vs.find(name=x)['id']), []).append(k)

    process_proteins_pairs = combinations(process_proteins, 2)
    protein_pairs = {}
    for pair in process_proteins_pairs:
        if pair[0] in protein_pairs:
            protein_pairs[pair[0]].append(pair[1])
        else:
            protein_pairs[pair[0]] = [pair[1]]

    process_proteins_betweenness = {}
    for process in processes_df['process']:
        process_proteins_betweenness[process] = {}
    for protein, protein_list in tqdm(protein_pairs.items()):
        shortest_path_dict = {}
        shortest_paths = graph.get_all_shortest_paths(protein, protein_list, mode='in')
        protein_a = shortest_paths[0][0]
        for shortest_path in shortest_paths:
            protein_b = shortest_path[-1]
            if protein_b not in shortest_path_dict:
                shortest_path_dict[protein_b] = [shortest_path[1:-1]]
            else:
                shortest_path_dict[protein_b].append(shortest_path[1:-1])

        shortest_path_count = {protein_B : Counter(chain.from_iterable(sp)) for protein_B, sp in shortest_path_dict.items()}
        shortest_path_ratio = {}
        for protein_A, counts in shortest_path_count.items():
            shortest_path_ratio[protein_A] = {protein_B: n/len(shortest_path_dict[protein_A]) for protein_B, n in counts.items()}
        for protein_b in shortest_path_ratio.keys():
            processes_in_pair = list(set(processes_by_proteins[protein_a]) & set(
                    processes_by_proteins[protein_b]))
            for process in processes_in_pair:
                process_proteins_betweenness[process] = dict(
                            Counter(process_proteins_betweenness[process])+Counter(shortest_path_ratio[protein_b]))


    for protein in range(len(graph.vs['name'])):
        for process in process_proteins_betweenness.keys():
            if protein not in process_proteins_betweenness[process]:
                process_proteins_betweenness[process][protein] = 0

    id_dict = {}
    for i in range(len(graph.vs['name'])):
        id_dict[i] = graph.vs['name'][i]

    betweenness = pd.DataFrame.from_dict(
        process_proteins_betweenness, orient='columns')
    betweenness['index'] = betweenness.index.to_series().map(id_dict)
    betweenness.set_index('index', inplace=True)
    betweenness.sort_index(inplace=True)
    return betweenness


def random_walk_restart(graph, process_df):
    # Computation of the random walks with restart of every protein to the proteins of each process/disease with igraph's pagerank algorithm.
    #
    # INPUT:
    #   - dataframe of shortest paths
    #   - dataframe with processes/diseases and their respective proteins.
    #
    # RETURNS: dataframe with random walks with restart scores of every protein in each process/disease.
    process_proteins_rwr = {}
    for index, process in tqdm(process_df.iterrows(), total=process_df.shape[0]):
        process_proteins = [
            protein for protein in process['proteins_ids'] if protein in graph.vs['name']]
        process_id = process['process']
        rwr_values = graph.personalized_pagerank(
            reset_vertices=process_proteins)
        process_proteins_rwr[process_id] = rwr_values
    return process_proteins_rwr


def multiple_metrics(ppis, process_df):
    # Allows for the computation of the hypergeometric test, shortest paths, closeness, betweenness, and random walks with restart in the reduced networks.
    #
    # INPUT:
    #   - numpy array with PPIs
    #   - dataframe with processes/diseases and their respective proteins.
    #
    # RETURNS: collection of dataframes with hypergeometric, closeness, betweenneess and random walk with restart scores.
    hyper_scores = []
    closeness_scores = []
    betweenness_scores = []
    rwr_scores = []
    fraction_betweenness_scores = []
    df_keys = []
    keys = 0
    for array in tqdm(ppis, total=len(ppis)):
        df_keys.append(keys)
        keys += 1
        print('Generating Graph')
        ppi_df = pd.DataFrame(array, columns=[
                              'HGNC_A', 'HGNC_B', 'apid', 'dorothea', 'huri', 'omnipath', 'is_directed'])
        ppi_df.replace(0, np.nan, inplace=True)
        ppi_df.dropna(axis=0, how='all')
        reduced_graph = Graph.DataFrame(ppi_df, directed=False)
        
        graph = reduced_graph.simplify()
        if not graph.is_connected():
            cluster = graph.clusters()
            graph = graph.induced_subgraph(cluster[0])
        graph.write_gml('../../data/processed/graph_apid_huri_60')
        graph = Graph.Read_GML("../../data/processed/graph_apid_huri_60")
        print('Generating Confusion Matrix')
        adj_matrix = graph.get_adjacency()
        adj_matrix = np.array(adj_matrix.data)
        print('Hypergeometric Test Computation:')
        hyper_dict = hypergeometric_test(graph, process_df, adj_matrix)
        hyper_df = pd.DataFrame.from_dict(hyper_dict).transpose()
        hyper_scores.append(hyper_df)
        print('Shortest Paths Computation:')
        shortest_paths_df = shortest_paths(graph, process_df)
        print('Closeness Computation:')
        closeness_dict = closeness(shortest_paths_df, process_df)
        closeness_df = pd.DataFrame.from_dict(closeness_dict)
        closeness_scores.append(closeness_df)
        print('Betweenness Computation:')
        betweenness_dict = betweenness(shortest_paths_df, process_df, graph)
        betweenness_df = pd.DataFrame.from_dict(betweenness_dict)
        betweenness_df.fillna(value=0, inplace=True)
        betweenness_scores.append(betweenness_df)
        print('Fraction Betweenness Computation:')
        fraction_betweenness_df = fraction_betweenness(process_df, graph)
        fraction_betweenness_df.fillna(value=0, inplace=True)
        fraction_betweenness_scores.append(fraction_betweenness_df)
        print('Random-Walks with Restart Computation:')
        rwr_dict = random_walk_restart(graph, process_df)
        rwr_df = pd.DataFrame.from_dict(rwr_dict)
        rwr_df.rename(index=dict(
            zip(list(rwr_df.index), graph.vs['name'])), inplace=True)
        rwr_scores.append(rwr_df)
    hyper_scores_df = pd.concat(
        hyper_scores, keys=df_keys, axis=0).reset_index(level=1)
    closeness_df = pd.concat(
        closeness_scores, keys=df_keys, axis=0).reset_index(level=1)
    betweenness_df = pd.concat(
        betweenness_scores, keys=df_keys, axis=0).reset_index(level=1)
    fraction_betweenness_df = pd.concat(
        fraction_betweenness_scores, keys=df_keys, axis=0).reset_index(level=1)
    rwr_df = pd.concat(rwr_scores, keys=df_keys, axis=0).reset_index(level=1)
    return hyper_scores_df, closeness_df, betweenness_df, rwr_df, fraction_betweenness_df

def MaxLink(labels, adj):
    proteins = labels.index
    labels.reset_index(inplace=True, drop=True)
    maxlink = {}
    for module_name in tqdm(labels.columns):
        maxlink[module_name] = []
        module = labels[labels[module_name]==1].index.values
        protein_degree_module = np.sum(adj[:,module], axis=1)
        protein_degree = np.sum(adj, axis=1)
        for i in range(len(protein_degree)):
            connectivity = hypergeom.pmf(protein_degree_module[i], adj.shape[0], len(module), protein_degree[i])
            if connectivity >= 0.5:
                maxlink[module_name].append(0)
            else:
                maxlink[module_name].append(protein_degree_module[i])
    maxlink_df = pd.DataFrame.from_dict(maxlink)
    return maxlink_df


def genePANDA(graph, labels, sp, weight_adj=np.zeros((2,2))):
    print('Running...')
    proteins = graph.vs['name']
    average_distance = np.sum(sp, axis=1)/len(graph.vs['name'])
    average_distance_sqrt = np.sqrt(np.dot(average_distance[:,None],average_distance[None,:]))
    if weight_adj.shape != (2,2):
        sp = np.divide(sp,weight_adj)
    raw_distance = np.divide(sp, average_distance_sqrt)
    labels.reset_index(drop=True, inplace=True)
    genePANDA_proba = {}
    for module_name in tqdm(labels.columns):
        module = labels[labels[module_name]==1].index.values
        module_distance = raw_distance[:,module]
        weights = (np.sum(raw_distance, axis=1)/len(graph.vs['name'])) - (np.sum(module_distance, axis=1)/len(module))
        weights_labels_df = pd.DataFrame(labels[module_name])
        weights_labels_df['weights'] = weights
        weights_labels_df.sort_values(by='weights', inplace=True, ascending=False)
        weights_labels_df.reset_index(inplace=True)
        weights_labels_df.columns = ['true_index', 'label', 'weight']
        weights_labels_df.reset_index(inplace=True)
        weights_labels_df.set_index('true_index', inplace=True)
        weights_labels_df['P'] = weights_labels_df.apply(lambda row: row['index']+1, axis=1)
        weights_labels_df['TP'] = np.cumsum(weights_labels_df['label'])
        weights_labels_df['probability'] = weights_labels_df['TP']/weights_labels_df['P']
        weights_labels_df.sort_index(inplace=True)
        genePANDA_proba[module_name] = weights_labels_df['probability'].values
    genePANDA_df = pd.DataFrame.from_dict(genePANDA_proba, orient='columns')    
    genePANDA_df.index = graph.vs['name']
    return genePANDA_df