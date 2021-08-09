import numpy as np
import pandas as pd
import scipy
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, scale
from sklearn.metrics import classification_report, confusion_matrix, adjusted_rand_score

# ============== Data Processing ==============
# =============================================

def read_labels(ref, return_enc=False):
    ref = pd.read_csv(ref, sep='\t', index_col=0, header=None)

    encode = LabelEncoder()
    ref = encode.fit_transform(ref.values.squeeze())
    classes = encode.classes_
    if return_enc:
        return ref, classes, encode
    else:
        return ref, classes

def gene_filter_(data, X=6):
    total_cells = data.shape[1]
    count_1 = data[data > 1].count(axis=1)
    count_2 = data[data > 0].count(axis=1)

    genelist_1 = count_1[count_1 > 0.01*X * total_cells].index
    genelist_2 = count_2[count_2 < 0.01*(100-X) * total_cells].index
    genelist = set(genelist_1) & set(genelist_2)
    data = data.loc[genelist]
    return data

def sort_by_mad(data, axis=0):

    genes = data.mad(axis=axis).sort_values(ascending=False).index
    if axis==0:
        data = data.loc[:, genes]
    else:
        data = data.loc[genes]
    return data


# =========== scATAC Preprocessing =============
# ==============================================
def peak_filter(data, x=10, n_reads=2):
    count = data[data >= n_reads].count(1)
    index = count[count >= x].index
    data = data.loc[index]
    return data

def cell_filter(data):
    thresh = data.shape[0]/50
    # thresh = min(min_peaks, data_ATAC.shape[0]/50)
    data = data.loc[:, data.sum(0) > thresh]
    return data

def sample_filter(data, x=10, n_reads=2):
    data = peak_filter(data, x, n_reads)
    data = cell_filter(data)
    return data

# =================== Other ====================
# ==============================================

def estimate_k(data):
    p, n = data.shape
    if type(data) is not np.ndarray:
        data = data.toarray()
    x = scale(data)
    muTW = (np.sqrt(n-1) + np.sqrt(p)) ** 2
    sigmaTW = (np.sqrt(n-1) + np.sqrt(p)) * (1/np.sqrt(n-1) + 1/np.sqrt(p)) ** (1/3)
    sigmaHatNaive = x.T.dot(x)

    bd = np.sqrt(p) * sigmaTW + muTW
    evals = np.linalg.eigvalsh(sigmaHatNaive)

    k = 0
    for i in range(len(evals)):
        if evals[i] > bd:
            k += 1
    return k

def get_decoder_weight(model_file):
    state_dict = torch.load(model_file)
    weight = state_dict['decoder.reconstruction.weight'].data.cpu().numpy()
    return weight

def peak_selection(weight, weight_index, kind='both', cutoff=2.5):
    std = weight.std(0)
    mean = weight.mean(0)
    specific_peaks = []
    for i in range(10):
        w = weight[:,i]
        if kind == 'both':
            index = np.where(np.abs(w-mean[i]) > cutoff*std[i])[0]
        if kind == 'pos':
            index = np.where(w-mean[i] > cutoff*std[i])[0]
        if kind == 'neg':
            index = np.where(mean[i]-w > cutoff*std[i])[0]
        specific_peaks.append(weight_index[index])
    return specific_peaks
    

def pairwise_pearson(A, B):
    from scipy.stats import pearsonr
    corrs = []
    for i in range(A.shape[0]):
        if A.shape == B.shape:
            corr = pearsonr(A.iloc[i], B.iloc[i])[0]
        else:
            corr = pearsonr(A.iloc[i], B)[0]
        corrs.append(corr)
    return corrs

# ================= Metrics ===================
# =============================================

def reassign_cluster_with_ref(Y_pred, Y):
    def reassign_cluster(y_pred, index):#悦悦太聪明了
        y_ = np.zeros_like(y_pred)
        for i in range(y_pred.size):
            for j in range(index[1].size):
                if y_pred[i]==index[0][j]:
                    y_[i] = index[1][j]

        return y_

    from scipy.optimize import linear_sum_assignment as linear_assignment
    #from sklearn.utils.linear_assignment_ import linear_assignment
#     print(Y_pred.size, Y.size)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)

    return reassign_cluster(Y_pred, ind)

def cluster_report(ref, pred, classes):

    pred = reassign_cluster_with_ref(pred, ref)
    cm = confusion_matrix(ref, pred)
    print('\n## Confusion matrix ##\n')
    print(cm)
    print('\n## Cluster Report ##\n')
    print(classification_report(ref, pred, target_names=classes))
    ari_score = adjusted_rand_score(ref, pred)
    print("\nAdjusted Rand score : {:.4f}".format(ari_score))


# def binarization(imputed, peak_mean, cell_mean):
#     """
#     Transform imputed float values to binary
#         imputed values at (i, j) -> 1:
#             greater than average value of ith peak in raw data_ATAC
#             greater than average value of jth cell in raw data_ATAC
#         otherwise 0
#     """
# #     peak_mean = raw.mean(1)
# #     cell_mean = raw.mean(0)
#     v1 = imputed.gt(peak_mean, axis=0)
#     v2 = imputed.gt(cell_mean, axis=1)
#     binary = (v1 & v2).astype(int)
#     return binary
def binarization(imputed, raw):
    return scipy.sparse.csr_matrix((imputed.T > raw.mean(1).T).T & (imputed>raw.mean(0))).astype(np.int8)
