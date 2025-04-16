import numpy as np
from tqdm import *
import networkx as nx
from networkx.algorithms import cuts
import math
from itertools import chain
import matplotlib as plt

class SE:
    def __init__(self, graph: nx.Graph):
        self.graph = graph.copy()
        self.vol = self.get_vol()
        self.struc_data = {}

    def get_vol(self):
        '''
        get the volume of the graph
        '''
        return cuts.volume(self.graph, self.graph.nodes, weight='weight')

    def calc_1dSE(self):
        '''
        get the 1D SE of the graph
        '''
        SE = 0
        for n in self.graph.nodes:
            d = cuts.volume(self.graph, [n], weight='weight')
            SE += - (d / self.vol) * math.log2(d / self.vol)
        return SE

    def update_1dSE(self, original_1dSE, new_edges):
        '''
        get the updated 1D SE after new edges are inserted into the graph
        '''

        affected_nodes = []
        for edge in new_edges:
            affected_nodes += [edge[0], edge[1]]
        affected_nodes = set(affected_nodes)

        original_vol = self.vol
        original_degree_dict = {node: 0 for node in affected_nodes}
        for node in affected_nodes.intersection(set(self.graph.nodes)):
            original_degree_dict[node] = self.graph.degree(node, weight='weight')

        # insert new edges into the graph
        self.graph.add_weighted_edges_from(new_edges)

        self.vol = self.get_vol()
        updated_vol = self.vol
        updated_degree_dict = {}
        for node in affected_nodes:
            updated_degree_dict[node] = self.graph.degree(node, weight='weight')

        updated_1dSE = (original_vol / updated_vol) * (original_1dSE - math.log2(original_vol / updated_vol))
        for node in affected_nodes:
            d_original = original_degree_dict[node]
            d_updated = updated_degree_dict[node]
            if d_original != d_updated:
                if d_original != 0:
                    updated_1dSE += (d_original / updated_vol) * math.log2(d_original / updated_vol)
                updated_1dSE -= (d_updated / updated_vol) * math.log2(d_updated / updated_vol)

        return updated_1dSE


def find_optimal_threshold(matrix, min_threshold, max_threshold, num_partitions):
    step = (max_threshold - min_threshold) / num_partitions
    thresholds = np.arange(max_threshold, min_threshold, -step)
    edges_weights = []
    structural_entropies = []
    seg = None

    for i in range(num_partitions):
        if i == 101:
            print()
        threshold = thresholds[i]
        edges = np.where(matrix > threshold)
        weights = matrix[edges]
        edges_weights.append(list(zip(edges[0], edges[1], weights)))
        if i == 0:
            g = nx.Graph()
            g.add_weighted_edges_from(edges_weights[i])
            seg = SE(g)
            structural_entropies.append(seg.calc_1dSE())
        else:
            structural_entropies.append(seg.update_1dSE(structural_entropies[-1], edges_weights[i]))
    average = sum(structural_entropies) / len(structural_entropies)
    nearest = min(structural_entropies, key=lambda x: abs(x - average))
    optimal_threshold = thresholds[structural_entropies.index(nearest)]

    print("optimal_threshold:", optimal_threshold)
    return optimal_threshold


