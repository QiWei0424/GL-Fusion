import numpy as np
import pandas as pd
import argparse
import glob
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torch.nn as nn
from LGCN_model import GCN
from utils import load_data
from utils import accuracy

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def train(epoch, optimizer, features, fadj, labels, idx_train):
    labels.to(device)
    features = features[idx_train]
    labels = labels[idx_train]

    GCN_model.train()
    optimizer.zero_grad()
    output = GCN_model(features, fadj)
    loss_train = F.cross_entropy(output, labels)
    acc_train = accuracy(output, labels)
    loss_train.backward()
    optimizer.step()
    if (epoch+1) % 50 ==0:
        print('Epoch: %.2f | loss train: %.4f | acc train: %.4f' %(epoch+1, loss_train.item(), acc_train.item()))
    return loss_train.data.item()

def test(epoch, features, fadj, labels, idx_test):
    features = features[idx_test]
    labels =  labels[idx_test]
    GCN_model.eval()
    output = GCN_model(features, fadj)
    loss_test = F.cross_entropy(output, labels)

    #calculate the accuracy
    acc_test = accuracy(output, labels)

    #output is the one-hot label
    ot = output.detach().cpu().numpy()
    #change one-hot label to digit label
    ot = np.argmax(ot, axis=1)
    #original label
    lb = labels.detach().cpu().numpy()

    #calculate the f1 score
    f_weighted = f1_score(ot, lb, average='weighted')
    f_macro = f1_score(ot, lb, average='macro')


    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    #return accuracy and f1 score
    return acc_test.item(), f_weighted, f_macro




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--featuredata', '-fd', type=str, nargs=3, required=True,
                        help='Location of input files, must be 3 files')
    parser.add_argument('--rdata', '-rd', type=str, nargs=3, required=True,
                        help='Location of input files, must be 3 files')
    parser.add_argument('--fadjdata', '-fad', type=str, required=True, help='The adjacency matrix file.')
    parser.add_argument('--labeldata', '-ld', type=str, required=True, help='The sample label file.')
    parser.add_argument('--mode', '-m', type=int, choices=[0,1], default=0,
                        help='mode 0: 5-fold cross validation; mode 1: train and test a model.')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed, default=0.')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='cpu',
                        help='Training on cpu or gpu, default: cpu.')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='Training epochs, default: 200.')
    parser.add_argument('--learningrate', '-lr', type=float, default=0.01, help='Learning rate, default: 0.01.')
    parser.add_argument('--weight_decay', '-w', type=float, default=0.01,
                        help='Weight decay (L2 loss on parameters), methods to avoid overfitting, default: 0.01')
    parser.add_argument('--latentfactor', '-lf', type=int, default=10, help='Integration of multi-omics latent factor embeddings, default: 10.')
    parser.add_argument('--dropout', '-dp', type=float, default=0.5, help='Dropout rate, methods to avoid overfitting, default: 0.5.')
    parser.add_argument('--nclass', '-nc', type=int, default=5, help='Number of classes, default: 5')
    parser.add_argument('--patience', '-p', type=int, default=100, help='Patience')
    args = parser.parse_args(['-rd', 'data/BRCA/DEG-results/Met.txt', 'data/BRCA/DEG-results/SCNV.txt', 'data/BRCA/DEG-results/Seq_RNA.txt',
                              '-fd', 'data/BRCA/Met.csv', 'data/BRCA/SCNV.csv', 'data/BRCA/Seq_RNA.csv',
                              '-fad', 'results/BRCA_SNF.csv',
                              '-ld', 'data/BRCA/label.csv',
                              '-m', '0', '-d', 'gpu', '-p', '10'])

    # Check whether GPUs are available(0.8159, 0.0646)
    device = torch.device('cpu')
    if args.device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set random seed
    setup_seed(args.seed)

    # load input files
    fadj, features, label = load_data(args.fadjdata,  args.featuredata, args.rdata, args.labeldata, device)
    labels = torch.tensor(label.iloc[:, 1].values, dtype=torch.long, device=device)

    print('Begin training model...')

    # 5-fold cross validation
    if args.mode == 0:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        len = np.zeros((label.shape[0],))
        acc_res= []
        F1_weighted =[]
        F1_macro = []

        best_acc_res = []
        best_F1_weighted = []
        best_F1_macro = []
        #record accuracy and f1 score

        # split train and test data
        for idx_train, idx_test in skf.split(len, len):
            # initialize a model
            GCN_model = GCN(nfea=features.shape[1] ,n_in=features.shape[2], n_lf=args.latentfactor, n_class=args.nclass, dropout=args.dropout)
            GCN_model.to(device)
            # define the optimizer
            optimizer = torch.optim.Adam(GCN_model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)
            best_acc = 0.0
            best_f1 = 0.0
            best_macro = 0.0

            idx_train, idx_test= torch.tensor(idx_train, dtype=torch.long, device=device), torch.tensor(idx_test, dtype=torch.long, device=device)
            for epoch in range(args.epochs):
                train(epoch, optimizer, features, fadj, labels, idx_train)
                acc, f_weighted, f_macro = test(epoch, features, fadj, labels, idx_test)
                if acc>best_acc:
                    best_acc = acc
                if f_weighted>best_f1:
                    best_f1 = f_weighted
                if f_macro>best_macro:
                    best_macro = f_macro
            # calculate the accuracy and f1 score
            acc1, f_weighted1, f_macro1 = test(epoch, features, fadj, labels, idx_test)

            acc_res.append(acc1)
            F1_weighted.append(f_weighted1)
            F1_macro.append(f_macro1)

            best_acc_res.append(best_acc)
            best_F1_weighted.append(best_f1)
            best_F1_macro.append(best_macro)
        print(best_acc_res)
        print(best_F1_weighted)
        print(best_F1_macro)

        print('5-fold  Acc(%.4f, %.4f)' % (np.mean(acc_res), np.std(acc_res)))
        print('5-fold  F1_weighted(%.4f, %.4f)' % (np.mean(F1_weighted), np.std(F1_weighted)))
        print('5-fold  F1_macro(%.4f, %.4f)' % (np.mean(F1_macro), np.std(F1_macro)))

        print('5-fold  best_Acc(%.4f, %.4f)' % (np.mean(best_acc_res), np.std(best_acc_res)))
        print('5-fold  best_F1_weighted(%.4f, %.4f)' % (np.mean(best_F1_weighted), np.std(best_F1_weighted)))
        print('5-fold  best_F1_macro(%.4f, %.4f)' % (np.mean(best_F1_macro), np.std(best_F1_macro)))


    elif args.mode == 1:
        indices = np.arange(label.shape[0])
        idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
        idx_train, idx_test = torch.tensor(idx_train, dtype=torch.long, device=device), torch.tensor(idx_test,dtype=torch.long,device=device)

        GCN_model = GCN(nfea=features.shape[1], n_in=features.shape[2], n_lf=args.latentfactor, n_class=args.nclass, dropout=args.dropout)
        GCN_model.to(device)
        # define the optimizer
        optimizer = torch.optim.Adam(GCN_model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)

        loss_values = []    #record the loss value of each epoch
        # record the times with no loss decrease, record the best epoch
        bad_counter, best_epoch = 0, 0
        best = 1000   #record the lowest loss value
        for epoch in range(args.epochs):
            loss_values.append(train(epoch, optimizer, features, fadj, labels, idx_train))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1     #In this epoch, the loss value didn't decrease
            if bad_counter == args.patience:
                break

            #save model of this epoch
            torch.save(GCN_model.state_dict(), 'model/BRCA/{}.pkl'.format(epoch))

            #reserve the best model, delete other models
            files = glob.glob('model/BRCA/*.pkl')
            for file in files:
                name = file.split('\\')[1]
                epoch_nb = int(name.split('.')[0])
                if epoch_nb != best_epoch:
                    os.remove(file)

        print('Training finished.')
        print('The best epoch model is ',best_epoch)
        GCN_model.load_state_dict(torch.load('model/BRCA/{}.pkl'.format(best_epoch)))
        test(epoch, features, fadj, labels, idx_test)
    print('Finished!')