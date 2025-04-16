import argparse
import warnings
from utils import read_dataset
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--omic_file', action='append',
                        help='REQUIRED: File path for omics files (should be matrix)', required=True)
    parser.add_argument('-l', '--label_file', help='REQUIRED: File path for label file', required=True)

    args = parser.parse_args(['-f', "original\BRCA\Met.csv", '-f', 'original\BRCA\SCNV.csv', '-f', 'original\BRCA\Seq_RNA.csv',
                              '-l', 'original\BRCA\label.csv'])
    return args

def find_common_feature(df_omics):
    patients = [df_tmp.columns.to_list() for df_tmp in df_omics]
    patients_shared = patients[0]
    for i in range(1, len(patients)):
        patients_shared = list(set(patients_shared).intersection(patients[i]))
    for i in range(len(df_omics)):
        df_omics[i] = df_omics[i].loc[:, patients_shared].sort_index()
    return df_omics



if __name__ == '__main__':
    args = get_args()
    df_omics, df_label, df_clin = read_dataset(args)
    df_label.to_csv('data/BRCA/label.csv')
    df_omics[0].to_csv('data/BRCA/Met.csv')
    df_omics[1].to_csv('data/BRCA/SCNV.csv')
    df_omics[2].to_csv('data/BRCA/Seq_RNA.csv')

