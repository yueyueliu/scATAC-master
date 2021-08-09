import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import os

from help.plot import plot_confusion_matrix, plot_embedding, plot_heatmap
from help.utils import read_labels, reassign_cluster_with_ref, pairwise_pearson
from help.specifity import *

dataname = "Leukemia"
indir = "E:/scATAC_impute/data_ATAC/{}/".format(dataname)
ref, classes, le = read_labels(indir+'labels.txt', return_enc=True)
y = le.inverse_transform(ref)

raw = pd.read_csv(indir+'data.txt', sep='\t', index_col=0)
out_dir = "E:/scATAC_impute/Result/{}/".format(dataname)
scimpute_imputed = pd.read_csv(out_dir + 'scimpute_imputed.csv', sep=',', index_col=0)
#scimpute_imputed = pd.read_csv(out_dir + 'scimpute_imputed.txt', sep=' ', index_col=0)
saver_imputed = pd.read_csv(out_dir+'SAVER_imputed.txt', sep='\t', index_col=0)
magic_imputed = pd.read_csv(out_dir+'Magic_imputed.txt', sep='\t', index_col=0)
deepimpute = pd.read_csv(out_dir+'deepimpute_imputed.txt',sep='\t', index_col=0)
deepimpute_imputed = pd.DataFrame(deepimpute.values.T,index=deepimpute.columns,columns=deepimpute.index)
prime_imputed = pd.read_csv(out_dir+'Prime_imputed.txt', sep='\t', index_col=0)
bayNorm_imputed = pd.read_csv(out_dir+'bayNorm_imputed.txt',sep='\t', index_col=0)
knnsmoothing_imputed = pd.read_csv(out_dir+'knnsmoothing_imputed.txt',sep='\t', index_col=0)

cell_corr = []
cell_method = []
cell_frac = []
methods = ['raw', 'magic', 'saver', 'scImpute', 'deepimpute', 'prime','bayNorm','knnsmoothing']

row_cluster = True
for i, data in enumerate([raw, magic_imputed, saver_imputed, scimpute_imputed,
                          deepimpute_imputed, prime_imputed, bayNorm_imputed, knnsmoothing_imputed]):
    for c in classes:
        cells = np.where(y==c)[0]
        aa = data.iloc[:, cells].T
        bb = raw.iloc[:, cells].sum(1)
        cell_corr.append(pairwise_pearson(aa, bb))
        cell_method.append([methods[i]]*len(cells))
        cell_frac.append([c]*len(cells))
cell_corr = pd.Series(np.concatenate(cell_corr, axis=0))
cell_method = pd.Series(np.concatenate(cell_method, axis=0))
cell_frac = pd.Series(np.concatenate(cell_frac, axis=0))

cell_concat = pd.concat([cell_corr, cell_method, cell_frac],  axis=1)
cell_concat.columns = ['correlation with metacell', 'method', 'cell type']
cell_concat.to_csv('metacell_{}.txt'.format(dataname), sep='\t')
print(1)