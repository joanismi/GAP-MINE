import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from igraph import Graph
from ast import literal_eval
from tqdm import trange
from joblib import Parallel, delayed

def get_protein_index(dataframe, graph, module_id_col='Reactome_ID', protein_id_col='protein_id'):
    # INPUT:
    #   -Dataframe of the process/disease,
    #   -whether the dataframe is from reactome or disgenet
    #   -protein graph
    #
    # RETURNS: set of proteins and respective IDs grouped by each process/disease
   
    new_dataframe = dataframe.copy()
    new_dataframe['protein_index'] = [graph.vs.find(protein).index for protein in new_dataframe[protein_id_col]]
    
    new_dataframe = new_dataframe.groupby(module_id_col, as_index=False).aggregate(list)
    new_dataframe['module_size'] = new_dataframe['protein_index'].transform(len)
    return new_dataframe


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
    
    return np.array(fs[0])


def add_false_annotations(modules_df, sp_df, graph, modules_col='module_id', n_jobs=-1):
    """
    Adds false annotations to proteins.

    modules_df: array like
    """
    rng = np.random.default_rng(42)
    
    def f(module, modules_df, graph, modules_col):
        protein_indices = modules_df.iloc[module]['protein_index']
        
        min_sp = sp_df.loc[~sp_df.index.isin(protein_indices), protein_indices].min(axis=1)
        min_sp_index = min_sp.index
        min_sp = min_sp.to_numpy()
        degree_values = np.array(graph.degree(sp_df[~sp_df.index.isin(protein_indices)].index))
        log_degree_values = np.log10(degree_values)

        weight = log_degree_values/(10**min_sp)
        
        normalized_weight = weight/np.sum(weight)

        new_proteins = list(rng.choice(min_sp_index, int(len(protein_indices)*0.1), p=normalized_weight))

        indices = protein_indices+new_proteins
        ids = graph.vs[indices]['name']
        
        return {modules_col: modules_df.iloc[module][modules_col], 'protein_id': ids, 'protein_index': indices}
    
    fa_df = Parallel(n_jobs=n_jobs)(delayed(f)(module, modules_df, graph, modules_col) for module in trange(modules_df.shape[0]))
        
    return pd.DataFrame(fa_df)