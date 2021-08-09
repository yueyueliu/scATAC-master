import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, scale
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Tkagg')
from matplotlib import pyplot as plt
import seaborn as sns
from help.utils import reassign_cluster_with_ref
from help.subroutines import *


def read_labels(ref, return_enc=False):
    ref = pd.read_csv(ref, sep='\t', index_col=0, header=None)
    encode = LabelEncoder()
    ref = encode.fit_transform(ref.values.squeeze())
    classes = encode.classes_
    if return_enc:
        return ref, classes, encode
    else:
        return ref, classes

def filtering(raw_matrix):
    matrix = numpy.array([x/x.sum()*1000000 for x in raw_matrix])
    gini = lambda x : 0.5 * numpy.abs(numpy.subtract.outer(x, x)).mean() / numpy.mean(x)
    if raw_matrix.shape[0]<=10000:
        coef_gini = numpy.array([gini(x) for x in matrix.T])
    else:
        coef_gini = numpy.array([len(numpy.where(x>0)[0]) for x in matrix.T])
    means = matrix.mean(axis=0)
    mean_cutoff = (means.mean() - numpy.std(means))*2
    kk, bb = numpy.polyfit(numpy.log10(means), coef_gini, 1)
    yy = coef_gini + numpy.log10(means) * (-kk)
    index = numpy.where((yy>bb)&(means>mean_cutoff))[0]
    matrix_filtered = matrix[:, index]
    return matrix_filtered
#
#
def weighted_tsne(matrix, clusters):
    cell_types = list(set(clusters['cluster'].values))
    adjusted = copy.deepcopy(matrix.values)
    for ctype in cell_types:
        cluster_cells = numpy.where(clusters==ctype)[0]
        weight = adjusted[cluster_cells, :].mean(axis=0) * float(0.2)
        adjusted[cluster_cells, :] = numpy.array([x+weight for x in adjusted[cluster_cells, :]])
    if matrix.shape[0]<=10000:
        tsne_result = TSNE(n_components=2, random_state=int(0)).fit_transform(adjusted)
    return tsne_result

def build_accesson(ngroup=600, ncell=301, npc=40,pathin="",pathout="",dataname=""):
    ngroups, ncell_cut = ngroup, ncell
    reads = pd.read_csv(os.path.join(pathin, 'data.txt'), sep='\t', index_col=0)#读GSE时候用index_col=0
    cells = pd.read_csv(os.path.join(pathin,'filtered_cells.txt'), sep='\t', index_col=0,
                            engine='c', na_filter=False, low_memory=False)
    cells = cells.index.values
    peaks = ['peak'+str(x) for x in range(0, reads.shape[0])]
    npc = min(int(npc), reads.shape[0], reads.shape[1])
    pca_result = PCA(n_components=npc, svd_solver='full').fit_transform(reads)
    connectivity = kneighbors_graph(pca_result, n_neighbors=10, include_self=False)
    connectivity = 0.5*(connectivity + connectivity.T)
    ward_linkage = cluster.AgglomerativeClustering(n_clusters=ngroups, linkage='ward', connectivity=connectivity)
    y_predict = ward_linkage.fit_predict(pca_result)
    peak_labels_df = pd.DataFrame(y_predict, index=peaks, columns=['group'])
    peak_labels_df.to_csv(os.path.join(pathout,'Accesson_peaks_{}.csv'.format(dataname)), sep='\t')
    groups = list(set(y_predict))
    reads = reads.values
    coAccess_matrix = np.array([reads[np.where(y_predict==x)[0],:].sum(axis=0) for x in groups])
    coAccess_matrix = coAccess_matrix.T
    coAccess_df = pd.DataFrame(coAccess_matrix, index=cells, columns=groups)
    coAccess_df.to_csv(os.path.join(pathout,'Accesson_reads_{}.csv'.format(dataname)),sep=',')
    return
#
def cluster_plot(norm='zscore',dataname="", clusternum=6,pathout="",classname=None):
    reads_df = pd.read_csv(os.path.join(pathout,'Accesson_reads_{}.csv'.format(dataname)), sep=',', index_col=0,
                   engine='c', na_filter=False, low_memory=False)
    normal = filtering(reads_df.values)
    if norm=='zscore':
        normal = scipy.stats.zscore(normal, axis=1)
    elif norm=='probability':
        normal = np.array([x/x.sum() for x in normal])
    else:
        print("Error: --norm should be zscore or probability")
    matrix = pd.DataFrame(normal, index=reads_df.index, columns=['c_'+str(x+1) for x in numpy.arange(0,len(normal.T))])
    connect = kneighbors_graph(matrix, n_neighbors=20, include_self=False)
    connectivity = 0.5*(connect + connect.T)
    n_clust = clusternum
    clusters = knn_cluster(matrix, n_clust, connectivity)
    tsne_result = weighted_tsne(matrix, clusters)
    plot_cluster(clusters, n_clust, tsne_result, pathout,dataname,classname)

    c = clusters.values.T
    c = c[:][0]
    return c

#
dataname = "Splenocyte"
ncell = 3166
clusternum = 12

pathin = "E:/scATAC_impute/data_ATAC/{}/".format(dataname)
pathout = "E:/scATAC_impute/cluster/raw/"
label, classname = read_labels(os.path.join(pathin, "labels.txt"))
raw = pd.read_csv(os.path.join(pathin,'data.txt'), sep='\t', index_col=0)
build_accesson(ncell=ncell, pathin=pathin,pathout=pathout,dataname=dataname,)
pred = cluster_plot(clusternum=clusternum,pathout=pathout,dataname=dataname,classname=classname)
pre = reassign_cluster_with_ref(pred, label)
pd.Series(pre, index=raw.columns).to_csv(os.path.join(pathout, 'label_{}.txt'.format(dataname)), sep='\t', header=False)

from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
a = adjusted_rand_score(label, pred)
print(a)
b = normalized_mutual_info_score(label, pred)
print(b)

