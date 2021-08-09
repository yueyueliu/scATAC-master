#!/usr/bin/python
# coding: utf-8
import warnings
warnings.filterwarnings("ignore")
#
import numpy,pandas,os,time
#import scanpy,anndata
from optparse import OptionParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn
from sklearn.decomposition import PCA
from help import subroutines
from rpy2.robjects.packages import importr
from rpy2 import robjects
#
#
def monocle_trajectory(options,method):
    reads_csv = options.s+'/matrix/{}_Accesson_reads.csv'.format(method)
    reads = pandas.read_csv(reads_csv, sep=',', index_col=0, engine='c', na_filter=False, low_memory=False)
    #accessons = pandas.read_csv(options.s+'/matrix/{}_Accesson_peaks.csv'.format(method), sep='\t', index_col=0)
    matrix = numpy.array(reads.values)
    #matrix[matrix == 0] = 1###有这些0会奥错
    normal = numpy.array([x/x.sum()*1000000 for x in matrix])
    reads = pandas.DataFrame(normal, index=reads.index.values, columns=reads.columns.values)
    npc = int(options.npc)
    pca_result = PCA(n_components=npc, svd_solver='full').fit_transform(reads)
    reads = pandas.DataFrame(pca_result[:, :npc], columns=['pc'+str(x) for x in numpy.arange(0,npc)], index=list(range(2658)))
    celltype_df = pandas.read_csv(options.cfile, sep='\t', index_col=0)
    if 'cluster' not in options.cfile:
        celltype_df['cluster'] = celltype_df['type']
    celltype_df = pandas.DataFrame(celltype_df.values,index=list(range(2658)),columns=celltype_df.columns)
    input_csv = options.s+'/matrix/{}_monocle_reads.csv'.format(method)
    cells_csv = options.s+'/matrix/{}_monocle_cells.tsv'.format(method)
    peaks_csv = options.s+'/matrix/{}_monocle_peaks.tsv'.format(method)
    trajectory_csv = options.s+'/result/{}_monocle_trajectory.csv'.format(method)
    reduced_csv = options.s+'/result/{}_monocle_reduced_dimension.csv'.format(method)
    peaks_df = pandas.DataFrame(reads.columns.values, index=reads.columns.values, columns=['gene_short_name'])
    peaks_df.to_csv(peaks_csv, sep='\t')
    celltype_df.to_csv(cells_csv, sep='\t')
    reads.T.to_csv(input_csv, sep=',')
#
    #os.system('Rscript run_monocle.R '+input_csv+' '+cells_csv+' '+peaks_csv+' '+reduced_csv+' '+trajectory_csv)
    importr("monocle")
    expr_matrix = robjects.r('read.csv')(input_csv, header=True, sep=',', **{'row.names':1, 'check.names':False})
    cells = robjects.r('read.delim')(cells_csv, **{'row.names':1})
    genes = robjects.r('read.delim')(peaks_csv, **{'row.names':1})
    pd = robjects.r('new')('AnnotatedDataFrame', data=cells)
    fd = robjects.r('new')('AnnotatedDataFrame', data=genes)
    matrix = robjects.r('as.matrix')(expr_matrix)
    negbinomial_size = robjects.r('negbinomial.size')()
    HSMM = robjects.r('newCellDataSet')(matrix, phenoData=pd, featureData=fd, expressionFamily=negbinomial_size)
    HSMM = robjects.r('estimateSizeFactors')(HSMM)
    HSMM = robjects.r('reduceDimension')(HSMM, max_components=3, norm_method='none', num_dim=20, reduction_method='DDRTree')
    HSMM = robjects.r('orderCells')(HSMM)
    robjects.r('write.csv')(HSMM.slots['reducedDimS'], reduced_csv)
    phenoData = HSMM.slots['phenoData']
    robjects.r('write.csv')(phenoData.slots['data'], trajectory_csv)
#
    return
#
#
def plot_traj(reduced_df, cells_df, out_fig, options):
    reduced_df = numpy.array(reduced_df)
    size = 20
    wordsize = 20
    colors = numpy.array(['skyblue','gold','#ff7f00','green','lightgreen', 'red', '#99CC00',
                       '#FF00FF', '#ff3399', '#a65628', '#984ea3', '#999999',
                      '#e41a1c', '#dede00', 'b', 'g', 'c', 'm', 'y', 'k',
                      '#ADFF2F', '#7CFC00', '#32CD32', '#90EE90', '#00FF7F', '#3CB371',
                      '#008000', '#006400', '#9ACD32', '#6B8E23', '#556B2F', '#66CDAA',
                      '#8FBC8F', '#008080', '#DEB887', '#BC8F8F', '#F4A460', '#B8860B',
                      '#CD853F', '#D2691E', '#8B4513', '#A52A2A', '#778899', '#2F4F4F',
                      '#FFA500', '#FF4500', '#DA70D6', '#FF00FF', '#BA55D3', '#9400D3',
                      '#8B008B', '#9370DB', '#663399', '#4B0082'])
    fig = plt.figure(1, figsize=(10,10))
    if int(options.dim)==3:
        ax = fig.add_subplot(111, projection='3d')
        cell_types = list(set(list(cells_df['cluster'].values)))
        cell_types.sort()
        for itype,ctype in enumerate(cell_types):
            cluster = cells_df.loc[cells_df['cluster']==ctype]
            cluster = cluster.index.values
            components = reduced_df[:,cluster]
            ax.scatter(components[0], components[1], components[2], c=colors[itype], edgecolors='none',
                       s=size, label=cell_types[itype])
        ax.set_zlabel('Component 3')
        beta1, beta2 = int(options.angle.split(',')[0]), int(options.angle.split(',')[1])
        ax.view_init(beta1, beta2)
        ax.set_zticklabels([])
        reduced_df = pandas.DataFrame(reduced_df)
        ax.set_zlim(reduced_df.iloc[2].min()-0.02, reduced_df.iloc[2].max()+0.02)
    elif int(options.dim)==2:
        ax = fig.add_subplot(111)
        cell_types = list(set(list(cells_df['cluster'].values)))
        cell_types.sort()
        for itype,ctype in enumerate(cell_types):
            cluster = cells_df.loc[cells_df['cluster']==ctype]
            cluster = cluster.index.values
            components = reduced_df[:,cluster]
            reduced_df = pandas.DataFrame(reduced_df)
            ax.scatter(components[0], components[1], c=colors[itype], edgecolors='none',
                       s=size, label=cell_types[itype])
    ax.legend(fontsize=wordsize, bbox_to_anchor=(1.3, 1.0))
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(reduced_df.iloc[0].min()-0.02, reduced_df.iloc[0].max()+0.02)
    ax.set_ylim(reduced_df.iloc[1].min()-0.02, reduced_df.iloc[1].max()+0.02)
    fig.savefig(out_fig, bbox_inches='tight')
    return
#
opts = OptionParser()
usage = "Build monocle trajectory\nusage: %prog -s project --npc 5 --cfile cluster.csv"
opts = OptionParser(usage=usage, version="%prog 1.0.6")
opts.add_option("-s", default="E:/scATAC_impute/pseudotime",help="The project folder.")
#opts.add_option("--method", default='monocle', help='Define trajectory constuction method, monocle or paga, default=monocle')
opts.add_option("--npc", default=5, help="Number of principle components used for pseudo-time trajectory, defaul=5")
opts.add_option("--cfile", default="E:/scATAC_impute/pseudotime/data/filtered_cells_saver.txt",help='Cell-types file, e.g. filtered_cells.csv or cluster.csv')
opts.add_option("--dim", default=3, help="Plot 2D or 3D trajectory, default=3")
opts.add_option("--angle", default='30,30', help='Angles to rotate the 3D trajectory, default=30,30')
options, arguments = opts.parse_args()

dataname = "GSE69769"
method = "knnsmoothing"

monocle_trajectory(options, method=method)
reduced_df = pandas.read_csv(options.s+'/result/{}_monocle_reduced_dimension.csv'.format(method), sep=',', index_col=0)
cells_df = pandas.read_csv(options.s+'/matrix/{}_monocle_cells.tsv'.format(method), sep='\t', index_col=0)
plot_traj(reduced_df, cells_df, options.s+'/figure/{}_pseudotime.tif'.format(method), options)
