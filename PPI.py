from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import scipy.sparse as sp
import networkx as nx
from tqdm import tqdm


def PPI(relation_df):
    relation_list = []
    for index, row in relation_df.iterrows():
        relation_list.append((row[0], row[1]))
    graph = nx.Graph(relation_list)
    return graph



def remove_isolated_nodes(matrix):
    isolated_nodes = [i for i, row in enumerate(matrix) if all(cell == 0 for cell in row)]
    for node in isolated_nodes:
        for i in range(len(matrix)):
            matrix[i].pop(node)
        matrix.pop(node)
    return matrix


def intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return list(set1.intersection(set2))

def get_connected_components(graph_structure):
    adjacency_matrix = nx.from_pandas_adjacency(graph_structure)
    graph = nx.Graph(adjacency_matrix)
    #print(graph)
    df = pd.read_table('data/HomoSapiens.txt', sep='\t', header=0)
    relation_df = pd.DataFrame(df)
    PPI_graph = PPI(relation_df)
    set = intersection(graph.nodes(),PPI_graph.nodes())
    sub_graph = PPI_graph.subgraph(set)
    #print(sub_graph)
    edges = sub_graph.edges()
    graph.add_edges_from(edges)
    #print(graph)
    nodes = list(graph.nodes())
    adjacency_matrix = nx.to_pandas_adjacency(graph, nodelist=nodes)
    # print(adjacency_matrix)
    return  adjacency_matrix


