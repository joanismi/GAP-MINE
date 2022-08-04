import random
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from igraph import Graph
from ast import literal_eval


def get_protein_index(dataframe, data_origin, g_simple):
    # INPUT:
    #   -Dataframe of the process/disease,
    #   -whether the dataframe is from reactome or disgenet
    #   -protein graph
    #
    # RETURNS: set of proteins and respective IDs grouped by each process/disease
    if data_origin == 'disgenet':
        group_column = 'diseaseId'
        protein_column = 'geneSymbol'
    if data_origin == 'reactome':
        group_column = 'Reactome ID'
        protein_column = 'HGNC ID'
    proteins_by_disease_df = dataframe.groupby(
        group_column)[protein_column].apply(list).reset_index()
    proteins_by_disease_df.columns = ['process', 'proteins_ids']
    protein_indexes_list = []
    for protein_list in proteins_by_disease_df['proteins_ids'].values:
        protein_indexes = []
        for protein in protein_list:
            protein_indexes.append(g_simple.vs.find(protein).index)
        protein_indexes_list.append(protein_indexes)
    proteins_by_disease_df['protein_index'] = protein_indexes_list
    return proteins_by_disease_df


def process_proteins_selection(df_, proteins_df, multiple_df=True):
    # From a metric dataframe (where there are rows with n proteins and columns with n processes/diseases),
    # retrieves two dataframes with metrics for process processes and non-process proteins.
    #
    # INPUT:
    #   - metric dataframe
    #   - dataframe with proteins of each process/disease
    #   - option multiple_df allows for the selection to occur when dealing with the reduction dtaframes.
    #
    # RETURNS: two metric dataframes with process/disease related and non-related proteins
    process_proteins_list = []
    remain_proteins_list = []
    df_keys = []
    if multiple_df == True:
        for key, df in tqdm(df_.groupby(level=0)):
            df_keys.append(key)
            df = df.set_index('level_1')
            process_proteins_dict = {}
            remain_proteins_dict = {}
            for process in proteins_df.index:
                process_proteins = [
                    protein for protein in proteins_df.loc[process, 'proteins_ids'] if protein in df.index]
                remain_proteins = [
                    protein for protein in df.index if protein not in process_proteins]
                process_proteins_df = df.loc[process_proteins, process]
                remain_proteins_df = df.loc[remain_proteins, process]
                process_proteins_dict[process] = process_proteins_df.to_dict()
                remain_proteins_dict[process] = remain_proteins_df.to_dict()
            process_proteins_list.append(
                pd.DataFrame.from_dict(process_proteins_dict))
            remain_proteins_list.append(
                pd.DataFrame.from_dict(remain_proteins_dict))
            process_proteins_df = pd.concat(
                process_proteins_list, keys=df_keys, axis=0).reset_index(level=1)
            remain_proteins_df = pd.concat(
                remain_proteins_list, keys=df_keys, axis=0).reset_index(level=1)
    else:
        df = df_
        process_proteins_dict = {}
        remain_proteins_dict = {}
        for process in tqdm(proteins_df.index):
            process_proteins = [
                protein for protein in proteins_df.loc[process, 'proteins_ids'] if protein in df.index]
            remain_proteins = [
                protein for protein in df.index if protein not in process_proteins]
            process_proteins_df = df.loc[process_proteins, process]
            remain_proteins_df = df.loc[remain_proteins, process]
            process_proteins_dict[process] = process_proteins_df.to_dict()
            remain_proteins_dict[process] = remain_proteins_df.to_dict()
        process_proteins_df = pd.DataFrame.from_dict(process_proteins_dict)
        remain_proteins_df = pd.DataFrame.from_dict(remain_proteins_dict)
    return process_proteins_df, remain_proteins_df


def random_reduction(ppis, percentage, n_reps):
    # From the protein-protein interaction data, create n_reps random reductions of the interactions, with a specified percentage.
    #
    # INPUT:
    #   - PPI Data
    #   - percentage of PPIs wanted
    #   - number of random reductions
    #
    # RETURNS: numpy array with the several random reductions.
    ppis_array = ppis.to_numpy()
    n_ppis = int(round(len(ppis_array) * percentage, 0))
    rng = np.random.default_rng()
    for n in range(n_reps):
        if n == 0:
            random_ppis = np.array(
                [rng.choice(ppis_array, n_ppis, replace=False)])
            print(random_ppis.shape)
        else:
            red_ppis = np.array(
                [rng.choice(ppis_array, n_ppis, replace=False)])
            print(red_ppis.shape)
            random_ppis = np.concatenate((random_ppis, red_ppis))
    return random_ppis


def protein_ppi_by_process(proteins_by_process_df, adjacency_matrix, reduced=False):
    if reduced:
        try:
            proteins_by_process_df['protein_index'] = proteins_by_process_df['protein_index'].apply(
                literal_eval)
            proteins_by_process_df['proteins_ids'] = proteins_by_process_df['proteins_ids'].apply(
                literal_eval)
        except ValueError:
            proteins_by_process_df['protein_index'] = proteins_by_process_df['protein_index']
            proteins_by_process_df['proteins_ids'] = proteins_by_process_df['proteins_ids']
        adjacency_matrix = pd.DataFrame(adjacency_matrix)
        reduced_graph = Graph.DataFrame(
            adjacency_matrix[['V1', 'V2']], directed=False)
        graph = reduced_graph.simplify()
        if not graph.is_connected():
            cluster = graph.clusters()
            graph = graph.induced_subgraph(cluster[0])
            graph.write_gml("../python/data/graph_apid_huri_80")
            graph_ = Graph.Read_GML("../python/data/graph_apid_huri_80")
        adjacency_matrix = graph.get_adjacency()
        adjacency_matrix = np.array(adjacency_matrix.data)

    proteins_by_process_dict = proteins_by_process_df.to_dict('index')
    protein_ppi_by_process = {}
    for index in tqdm(proteins_by_process_dict.keys(), desc='PPI by process computation'):
        process = proteins_by_process_dict[index]['process']
        if reduced:
            process_proteins = [int(graph_.vs.find(name=x)[
                                    'id']) for x in proteins_by_process_dict[index]['proteins_ids'] if x in graph.vs['name']]
        else:
            process_proteins = proteins_by_process_dict[index]['protein_index']
        process_proteins_adjmatrix = adjacency_matrix[:, process_proteins]
        protein_process_ppi = list(
            np.sum(process_proteins_adjmatrix, axis=1))
        protein_ppi_by_process[process] = protein_process_ppi
    protein_ppi_by_process_df = pd.DataFrame.from_dict(protein_ppi_by_process)
    if reduced:
        return protein_ppi_by_process_df, adjacency_matrix, graph_.vs['name']
    else:
        return protein_ppi_by_process_df


def graph_reduction(ppis):
    # Creates graph for the randomçy reduced networks.
    # INPUT:
    #   - PPI array
    #
    # RETURNS: Saves graph and gives list of graph proteins.
    ppis = pd.DataFrame(ppis)
    reduced_graph = Graph.DataFrame(ppis[['V1', 'V2']], directed=False)
    graph = reduced_graph.simplify()
    if not graph.is_connected():
        cluster = graph.clusters()
        graph = graph.induced_subgraph(cluster[0])
        graph.write_gml("../python/data/graph_apid_huri_80")
    return graph.vs['name']


def random_reduction_protein(ppis, percentage, n_reps):
    # From the protein-protein interaction data, create n_reps random reductions of the interactions given the protein degree, with a specified percentage.
    #
    # INPUT:
    #   - PPI Data
    #   - percentage of proteins wanted
    #   - number of random reductions
    #
    # RETURNS: numpy array with the several random reductions.

    ppis_array = ppis.to_numpy()
    degree = {}
    for ppi in ppis_array:
        if ppi[0] != ppi[1]:
            if ppi[0] in degree:
                if ppi[1] not in degree[ppi[0]]:
                    degree[ppi[0]].append(ppi[1])
            else:
                degree[ppi[0]] = [ppi[1]]
            if ppi[1] in degree:
                if ppi[0] not in degree[ppi[1]]:
                    degree[ppi[1]].append(ppi[0])
            else:
                degree[ppi[1]] = [ppi[0]]
    degree_count = {protein: len(list_protein)
                    for protein, list_protein in degree.items()}

    degree_count_df = pd.DataFrame.from_dict(
        degree_count, orient='index').sort_index()
    degree_count_values = list(degree_count_df.to_dict()[0].values())
    normalizer = round(1/float(sum(degree_count_values)), 64)
    degree_count_proba = np.array(
        [round(count*normalizer, 64) for count in degree_count_values]).astype('float64')
    degree_count_proba /= degree_count_proba.sum()
    degree_count_df['proba'] = degree_count_proba

    rng = np.random.default_rng()
    proteins = list(degree_count.keys())
    n_proteins_red = int(len(proteins)*percentage)
    random_ppis = np.zeros((n_reps, 185000, 7), dtype='object')
    for n in range(n_reps):
        random_proteins = rng.choice(
            proteins, n_proteins_red, replace=False, p=degree_count_proba)
        for ppis in ppis_array:
            if ppis[0] not in random_proteins and ppis[1] in random_proteins:
                np.append(random_proteins, np.array(ppis[0]))
            if ppis[1] not in random_proteins and ppis[0] in random_proteins:
                np.append(random_proteins, np.array(ppis[1]))
        red_ppis = np.array(
            [ppi for ppi in ppis_array if ppi[0] in random_proteins and ppi[1] in random_proteins])
        random_ppis[n, :len(red_ppis), :] = red_ppis

    return random_ppis


def process_selector(fs_df, column, method):
    # Selects Modules to be used in the classification task given their VIP score and method of choice.
    #
    # INPUT:
    #   - VIP scores
    #   - Module
    #   - method of selection
    #
    # RETURNS: array with selected modules

    fs = []
    process = fs_df.columns[column]
    if method == 'outlier':
        Q3 = fs_df[process].quantile(0.75)
        Q1 = fs_df[process].quantile(0.25)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5*IQR
        if len(fs_df[fs_df[process] > threshold].index) > 0:
            fs.append(np.array(fs_df[fs_df[process] > threshold].index))
        else:
            fs.append(np.array(fs_df[fs_df[process] >= 1].index))
    if method == 'outlier10':
        Q3 = fs_df[process].quantile(0.75)
        Q1 = fs_df[process].quantile(0.25)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5*IQR
        if len(fs_df[fs_df[process] > threshold].index) > 0:
            fs.append(np.array(fs_df[fs_df[process] > threshold].index))
        else:
            fs_list = list(fs_df[process].sort_values(
                ascending=False)[:11].index)
            if column not in fs_list:
                fs_list = fs_list[:10]
                fs_list.append(column)
            fs.append(fs_list)
    if method == 'outlier/Q3':
        Q3 = fs_df[process].quantile(0.75)
        Q1 = fs_df[process].quantile(0.25)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5*IQR
        if len(fs_df[fs_df[process] > threshold].index) > 0:
            fs.append(np.array(fs_df[fs_df[process] > threshold].index))
        else:
            fs.append(np.array(fs_df[fs_df[process] > Q3].index))
    if method == '10':
        fs_list = list(fs_df[process].sort_values(
            ascending=False)[:11].index)
        if column not in fs_list:
            fs_list = fs_list[:10]
            fs_list.append(column)
        fs.append(fs_list)
    if method == 'middle':
        Q3 = fs_df[process].quantile(0.75)
        Q1 = fs_df[process].quantile(0.25)
        IQR = Q3 - Q1
        threshold = Q3 + 1.5*IQR
        len_threshold = int(
            (len(fs_df[fs_df[process] > threshold].index) - 11)/2)+11
        fs_list = list(fs_df[process].sort_values(
            ascending=False)[:len_threshold].index)
        if column not in fs_list:
            fs_list = fs_list[:len_threshold-1]
            fs_list.append(column)
        fs.append(fs_list)

    return np.array(fs)