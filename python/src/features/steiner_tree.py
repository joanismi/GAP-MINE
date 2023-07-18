from igraph import Graph
import pandas as pd
import numpy as np

def sca(idlist, graph, adj):
    added_nodes = []
    increase = True
    original_nodes = idlist[:]
    while increase:
        subgraph = graph.induced_subgraph(idlist)
        components = Graph.clusters(subgraph, mode='strong')
        # print(list(components))
        comp_adj_matrix = adj[:, sorted(idlist)]
        lcc = max([len(component) for component in components])
        # print(lcc)
        ind_comp_adj_matrices = [comp_adj_matrix[:, comp]
                                 for comp in components]
        
        max_addition = []
        for ind_comp in ind_comp_adj_matrices:
            n_int = np.sum(ind_comp, axis=1)
            n_int[n_int >= 1] = ind_comp.shape[1]
            max_addition.append(n_int)
        
        max_addition = np.array(max_addition).transpose()
        max_addition_total = np.array(max_addition).sum(axis=1)
        if max(max_addition_total) > lcc:
            increase = True
            candidates = np.argwhere(max_addition_total == np.amax(
                max_addition_total)).flatten().tolist()
            if len(candidates) > 1:
                cand_dict = {}
                for cand in candidates:
                    cand_dict[cand] = len([id_ for id_ in idlist if id_ in np.argwhere(
                        adj[:, cand] == 1).flatten().tolist()])/len(np.argwhere(adj[:, cand] == 1).flatten().tolist())
                candidates = [key for key, value in cand_dict.items() if value == max(cand_dict.values())]

            idlist.append(candidates[0])
            added_nodes.append(candidates[0])

        else:
            increase = False
    subgraph = graph.induced_subgraph(idlist)
    final_components = list(Graph.clusters(subgraph, mode='strong'))
    final_component = max(final_components, key=len)
    main_component = [idlist[node] for node in final_component]
    conservative_module = list(set(original_nodes) & set(main_component))
    return pd.Series([main_component, conservative_module, added_nodes])