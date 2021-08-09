import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import os

from help.plot import plot_confusion_matrix, plot_embedding, plot_heatmap
from help.utils import read_labels, reassign_cluster_with_ref
from help.specifity import *

dataname = "Splenocyte"#Leukemia-GM12878HEK-GM12878HL-InSilico-Splenocyte-Forebrain
indir = "E:/scATAC_impute/data_ATAC/{}/".format(dataname)
_, classes, _ = read_labels(indir+'labels.txt', return_enc=True)

cell_concat = pd.read_csv('metacell_{}.txt'.format(dataname), sep='\t', index_col=0)
figsize = (min(len(classes)*2.5, 20), 4.5)
plt.figure(figsize=figsize)
plt.xlabel('round', fontsize=15)
plt.ylabel('round', fontsize=15)
plt.tick_params(labelsize=13)
g = sns.boxplot(x='cell type', y='correlation with metacell', hue='method', width=0.7, data=cell_concat)
#plt.ylim(0.4, 1)
plt.legend(fontsize=15, loc='right', bbox_to_anchor=(1.3, 0.2), frameon=False).set_visible(True)
#plt.legend().set_visible(False)
plt.title('{}'.format(dataname),fontsize=18)#标题的字体大小
#plt.gca().spines['right'].set_visible(False)
#plt.gca().spines['top'].set_visible(False)
plt.setp(g.get_xticklabels(), rotation=45)

plt.savefig('{}_correlation.tif'.format(dataname), format='tif', bbox_inches='tight', dpi=1200)
plt.show()