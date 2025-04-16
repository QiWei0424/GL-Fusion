import pandas as pd
import numpy as np
import sys
import torch
from sklearn.preprocessing import MinMaxScaler

def load_data(f_adj,  fea, rdata, lab, device):
    '''
    :param adj: the similarity matrix filename
    :param fea: the omics vector features filename
    :param lab: sample labels  filename
    '''
    print('loading data...')
    omics_data1 = pd.read_csv(fea[0], header=0, index_col=0)
    omics_data2 = pd.read_csv(fea[1], header=0, index_col=0)
    omics_data3 = pd.read_csv(fea[2], header=0, index_col=0)

    DEG_data_1 = pd.read_csv(rdata[0], header=None, names=['DEGname'])
    DEG_data_2 = pd.read_csv(rdata[1], header=None, names=['DEGname'])
    DEG_data_3 = pd.read_csv(rdata[2], header=None, names=['DEGname'])
    common_data = pd.merge(DEG_data_1, DEG_data_2, on='DEGname').merge(DEG_data_3, on='DEGname')

    common_list = common_data['DEGname'].tolist()


    omics_data_1 = omics_data1[common_list]
    omics_data_2 = omics_data2[common_list]
    omics_data_3 = omics_data3[common_list]
    scaler = MinMaxScaler(feature_range=(-1, 1))

    omics_data_1 = pd.DataFrame(scaler.fit_transform(omics_data_1), columns=omics_data_1.columns, index=omics_data_1.index).transpose()
    omics_data_2 = pd.DataFrame(scaler.fit_transform(omics_data_2), columns=omics_data_2.columns, index=omics_data_2.index).transpose()
    omics_data_3 = pd.DataFrame(scaler.fit_transform(omics_data_3), columns=omics_data_3.columns, index=omics_data_3.index).transpose()

    data_combined = np.array([omics_data_1.values, omics_data_2.values, omics_data_3.values])
    data = torch.tensor(data_combined, dtype=torch.float,
                        device=device)
    features = data.permute(2, 1, 0)

    label = pd.read_csv(lab, header=0, index_col=0)


    f_adj = pd.read_csv(f_adj, header=0, index_col=0)
    factor = np.ones(f_adj.shape[1])
    res = np.dot(f_adj, factor)  # degree of each node
    diag_matrix = np.diag(res)  # degree matrix
    d_inv = np.linalg.inv(diag_matrix)
    f_adj = d_inv.dot(f_adj)
    f_adj = torch.tensor(f_adj, dtype=torch.float, device=device)

    return f_adj, features, label



def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def read_(file):
    # read file
    if file.endswith('.csv'):
        df = pd.read_csv(file, index_col=0)
    elif file.endswith('.csv.gz'):
        df = pd.read_csv(file, compression='gzip', index_col=0)
    else:
        print('\n[Error]: The program cannot infer the format of {} . Currently, only the csv format is supported, please ensure that the file name suffix is .csv or .csv.gz.'.format(file))
        sys.exit(0)
    return df


def read_omics(args):
    omics = []
    for file in args.omic_file:
        df = read_(file)
        df = df.fillna(0)  # fill nan with 0
        omics.append(df)
    return omics


def read_label(args):
    file = args.label_file
    df = read_(file)
    df = df.rename(
        columns={df.columns.values[0]: 'label'})
    return df


def read_clin(args):
    file = args.clin_file
    df = None
    if not file is None:
        df = read_(file)
        # fill na
        df = df.fillna(0)
    return df

def process(df_omics, df_label, df_clin):
    # extract patient id
    patients = [df_tmp.index.to_list() for df_tmp in df_omics]
    patients.append(df_label.index.to_list())
    if not df_clin is None:
        patients.append(df_clin.index.to_list())

    # get shared patients between different data
    patients_shared = patients[0]
    for i in range(1, len(patients)):
        patients_shared = list(set(patients_shared).intersection(patients[i]))

    # extract shared patients' data
    for i in range(len(df_omics)):
        df_omics[i] = df_omics[i].loc[patients_shared, :].sort_index()
    df_label = df_label.loc[patients_shared, :].sort_index()
    if not df_clin is None:
        df_clin = df_clin.loc[patients_shared, :].sort_index()
    return df_omics, df_label, df_clin


# api
def read_dataset(args):
    # 1. read raw dataset
    # (1) read omics dataset
    df_omics = read_omics(args)
    # (2) read label
    df_label = read_label(args)
    # (3) read clinical feature
    df_clin = read_clin(args)

    # 2. process
    df_omics, df_label, df_clin = process(df_omics, df_label, df_clin)

    # 3. return clean dataset
    return df_omics, df_label, df_clin
