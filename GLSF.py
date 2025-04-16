# -*- coding: utf-8 -*-
import snf
import torch
import pandas as pd
import numpy as np
import argparse
import seaborn as sns
import warnings
from PPI import get_connected_components
from SE import find_optimal_threshold


warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rdata', '-rd', type=str, nargs=3, required=True,
                        help='Location of input files, must be 3 files')
    parser.add_argument('--featuredata', '-fd', type=str, nargs=3, required=True,
                        help='Location of input files, must be 3 files')
    parser.add_argument('--metric', '-m', type=str, choices=['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
                        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'], default='cosine',
                        help='Distance metric to compute. Must be one of available metrics in :py:func scipy.spatial.distance.pdist.')
    parser.add_argument('--K', '-k', type=int, default=20,
                        help='(0, N) int, number of neighbors to consider when creating affinity matrix. See Notes of :py:func snf.compute.affinity_matrix for more details. Default: 20.')
    parser.add_argument('--mu', '-mu', type=int, default=0.5,
                        help='(0, 1) float, Normalization factor to scale similarity kernel when constructing affinity matrix. See Notes of :py:func snf.compute.affinity_matrix for more details. Default: 0.5.')
    args = parser.parse_args(['-rd', 'data/BRCA/DEG-results/Met.txt', 'data/BRCA/DEG-results/SCNV.txt', 'data/BRCA/DEG-results/Seq_RNA.txt',
                              '-fd', 'data/BRCA/Met.csv', 'data/BRCA/SCNV.csv', 'data/BRCA/Seq_RNA.csv',
                              '-m', 'cosine'])

    print('Load data files...')
    omics_data_1 = pd.read_csv(args.featuredata[0], header=0, index_col=0)
    omics_data_2 = pd.read_csv(args.featuredata[1], header=0, index_col=0)
    omics_data_3 = pd.read_csv(args.featuredata[2], header=0, index_col=0)
    print(omics_data_1.shape, omics_data_2.shape, omics_data_3.shape)

    if omics_data_1.shape[0] != omics_data_2.shape[0] or omics_data_1.shape[0] != omics_data_3.shape[0]:
        print('Input files must have same samples.')
        exit(1)


    print('Load DEG files...')
    DEG_data_1 = pd.read_csv(args.rdata[0], header=None,names=['DEGname'])
    DEG_data_2 = pd.read_csv(args.rdata[1], header=None,names=['DEGname'])
    DEG_data_3 = pd.read_csv(args.rdata[2], header=None,names=['DEGname'])
    common_data = pd.merge(DEG_data_1, DEG_data_2, on='DEGname').merge(DEG_data_3, on='DEGname')

    common_list = common_data['DEGname'].tolist()
    omics_data1 = omics_data_1[common_list].transpose()
    omics_data2 = omics_data_2[common_list].transpose()
    omics_data3 = omics_data_3[common_list].transpose()

    print('Start similarity network fusion...')
    affinity_nets = snf.make_affinity([omics_data1.values.astype(np.float64), omics_data2.values.astype(np.float64), omics_data3.values.astype(np.float64)],
                                                                                                   metric=args.metric, K=args.K, mu=args.mu)
    fused_net = snf.snf(affinity_nets, K=args.K)

    print('Save fused adjacency matrix...')
    fused_df = pd.DataFrame(fused_net)
    fused_df.columns = omics_data1.index.tolist()
    fused_df.index = omics_data1.index.tolist()
    np.fill_diagonal(fused_df.values, 0)

    tensor = torch.tensor(fused_df.values).cuda()
    values_max, _ = tensor.view(-1).topk(1000)
    values_min, _ = tensor.view(-1).topk(5000)

    mean_min = values_min.min().item()
    mean_max = values_max.min().item()

    print('min:', mean_min)
    print('max:', mean_max)

    mean = find_optimal_threshold(fused_df.values, mean_min, mean_max, 200)
    #print(mean)
    df_processed = fused_df > mean
    df_processed = df_processed.astype(int)

    np.fill_diagonal(df_processed.values, 1)
    #df_processed.to_csv('results-ablation/BRCA_PPI.csv', header=True, index=True)

    df_processed = get_connected_components(df_processed)
    df_processed.to_csv('results/BRCA_SNF.csv', header=True, index=True)



    print('Success! Results can be seen in result file')
