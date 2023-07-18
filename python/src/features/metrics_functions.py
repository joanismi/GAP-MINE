from multiprocessing import process
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from tqdm import trange
from igraph import Graph
from joblib import Parallel, delayed


def random_walk_restart(graph, process_df, id_col='protein_id', n_jobs=-1):
    # Computation of the random walks with restart of every protein to the proteins of each process/disease with igraph's pagerank algorithm.
    #
    # INPUT:
    #   - dataframe of shortest paths
    #   - dataframe with processes/diseases and their respective proteins.
    #
    # RETURNS: dataframe with random walks with restart scores of every protein in each process/disease.
    
    def rwr(process):
        
        process_proteins = [
            protein for protein in process[id_col] if protein in graph.vs['name']
            ]
        
        rwr_values = graph.personalized_pagerank(
            reset_vertices=process_proteins)
        
        return rwr_values
    
    process_list = process_df.to_dict('records')
    process_proteins_rwr = Parallel(n_jobs=n_jobs)(
        delayed(rwr)(process) for process in tqdm(process_list)
        )
    

    process_proteins_rwr = pd.DataFrame(
        process_proteins_rwr,
        index=process_df['module_id'],
        columns=graph.vs['name']
        ).T

    return process_proteins_rwr
