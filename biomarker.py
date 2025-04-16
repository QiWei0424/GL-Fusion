import numpy as np
import pandas as pd
import argparse
import glob
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import torch.nn as nn
from LGCN_model import GCN
from utils import load_data
from utils import accuracy

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def test(features, fadj, labels, idx_test):

    features = features[idx_test]
    labels =  labels[idx_test]
    GCN_model.eval()
    output = GCN_model(features, fadj)

    #output is the one-hot label
    ot = output.detach().cpu().numpy()
    #change one-hot label to digit label
    ot = np.argmax(ot, axis=1)
    #original label
    lb = labels.detach().cpu().numpy()

    f_macro = f1_score(ot, lb, average='macro')
    f_weighted = f1_score(ot, lb, average='weighted')


    #return accuracy and f1 score
    return f_macro




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--featuredata', '-fd', type=str, nargs=3, required=True,
                        help='Location of input files, must be 3 files')
    parser.add_argument('--rdata', '-rd', type=str, nargs=3, required=True,
                        help='Location of input files, must be 3 files')
    parser.add_argument('--fadjdata', '-fad', type=str, required=True, help='The adjacency matrix file.')
    parser.add_argument('--labeldata', '-ld', type=str, required=True, help='The sample label file.')
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed, default=0.')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='cpu',
                        help='Training on cpu or gpu, default: cpu.')
    parser.add_argument('--latentfactor', '-lf', type=int, default=10, help='Integration of multi-omics latent factor embeddings, default: 10.')
    parser.add_argument('--dropout', '-dp', type=float, default=0.5, help='Dropout rate, methods to avoid overfitting, default: 0.5.')
    parser.add_argument('--nclass', '-nc', type=int, default=5, help='Number of classes, default: 5')
    args = parser.parse_args(['-rd', 'data/BRCA/DEG-results/Met.txt', 'data/BRCA/DEG-results/SCNV.txt', 'data/BRCA/DEG-results/Seq_RNA.txt',
                              '-fd', 'data/BRCA/Met.csv', 'data/BRCA/SCNV.csv', 'data/BRCA/Seq_RNA.csv',
                              '-fad', 'results/BRCA_SNF.csv',
                              '-ld', 'data/BRCA/label.csv',
                              '-m', 'model/BRCA/116.pkl',
                              '-d', 'gpu'])

    # Check whether GPUs are available
    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set random seed
    setup_seed(args.seed)

    # load input files
    fadj, features, label = load_data(args.fadjdata,  args.featuredata, args.rdata, args.labeldata, device)
    labels = torch.tensor(label.iloc[:, 1].values, dtype=torch.long, device=device)

    DEG_data_1 = pd.read_csv(args.rdata[0], header=None, names=['DEGname'])
    DEG_data_2 = pd.read_csv(args.rdata[1], header=None, names=['DEGname'])
    DEG_data_3 = pd.read_csv(args.rdata[2], header=None, names=['DEGname'])
    common_data = pd.merge(DEG_data_1, DEG_data_2, on='DEGname').merge(DEG_data_3, on='DEGname')
    common_list = common_data['DEGname'].tolist()

    indices = np.arange(label.shape[0])
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
    idx_train, idx_test = torch.tensor(idx_train, dtype=torch.long, device=device), torch.tensor(idx_test,dtype=torch.long,device=device)

    GCN_model = GCN(nfea=features.shape[1], n_in=features.shape[2], n_lf=args.latent, n_class=args.nclass,
                    dropout=args.dropout)
    GCN_model.to(device)
    GCN_model.load_state_dict(torch.load(args.model))
    f1 = test(features, fadj, labels, idx_test)


    feat_imp_list = []
    for i in range(features.shape[2]):
        feat_imp = {"feat_name": common_list}
        feat_imp['imp'] = np.zeros(len(common_list))
        for j in range(features.shape[1]):
            feat = features[:, j:j+1, i:i+1].clone()
            features[:, j:j+1, i:i+1] = 0
            f1_tmp = test(features, fadj, labels, idx_test)
            feat_imp['imp'][j] = (f1 - f1_tmp) * len(common_list)
            features[:, j:j+1, i:i+1] = feat.clone()
        feat_imp_list.append(pd.DataFrame(data=feat_imp))

    file_names = ['feature-imp/BRCA/Met_top.csv', 'feature-imp/BRCA/SCNV_top.csv', 'feature-imp/BRCA/RNA_top.csv']

    for idx, df in enumerate(feat_imp_list):
        df_sorted = df.sort_values(by='imp', ascending=False)
        feature_top = df_sorted[['feat_name', 'imp']].iloc[:10]
        feature_top.to_csv(file_names[idx], index=False)