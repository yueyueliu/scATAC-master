import sys,getopt,numpy,pandas,umap,scipy.stats
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#
#
def run_umap(indir, cellinfo, rand_stat=0, norm_method='zscore',dataname=""):
    mat_df = pandas.read_csv(indir+'Accesson_reads_{}.csv'.format(dataname), sep=',', index_col=0,
                             engine='c', na_filter=False, low_memory=False)
    if norm_method=='zscore':
        matrix = scipy.stats.zscore(mat_df.values, axis=1)
    elif norm_method=='probability':
        matrix = numpy.array([x*10000.0/x.sum() for x in mat_df.values])
    else:
        print('-n should be zscore or probability')
        sys.exit()
    umap_result = umap.UMAP(n_components=2, n_neighbors=15, random_state=rand_stat).fit_transform(matrix)
    cellinfo_df = pandas.read_csv(cellinfo, sep='\t', )
    umap_df = pandas.DataFrame(umap_result, index=cellinfo_df.index, columns=['UMAP1', 'UMAP2'])
    #umap_df.to_csv(outdir+'UMAP_{}.csv'.format(dataname), sep=',')
#
    if 'notes' in cellinfo_df.columns.values: cellinfo_df['type'] = cellinfo_df['notes']
    cTypes = list(set(cellinfo_df['type'].values))
    cTypes.sort()
    cTypeIndex = [numpy.where(cellinfo_df['type'].values==x) for x in cTypes]
    colors = numpy.array(['pink', 'red', '#377eb8', 'green', 'skyblue', 'lightgreen', 'gold',
                      '#ff7f00', '#000066', '#ff3399', '#a65628', '#984ea3', '#999999',
                      '#e41a1c', '#dede00', 'b', 'g', 'c', 'm', 'y', 'k',
                      '#ADFF2F', '#7CFC00', '#32CD32', '#90EE90', '#00FF7F', '#3CB371',
                      '#008000', '#006400', '#9ACD32', '#6B8E23', '#556B2F', '#66CDAA',
                      '#8FBC8F', '#008080', '#DEB887', '#BC8F8F', '#F4A460', '#B8860B',
                      '#CD853F', '#D2691E', '#8B4513', '#A52A2A', '#778899', '#2F4F4F',
                      '#FFA500', '#FF4500', '#DA70D6', '#FF00FF', '#BA55D3', '#9400D3',
                      '#8B008B', '#9370DB', '#663399', '#4B0082'])
    fig2, axes = plt.subplots(1, figsize=(5,5))

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(1);
    ax.spines['left'].set_linewidth(1);
    ax.spines['right'].set_linewidth(1);
    ax.spines['top'].set_linewidth(1);
    plt.tick_params(labelsize=8)

    for ict,ct in enumerate(cTypes):
        axes.scatter(umap_result[cTypeIndex[ict], 0], umap_result[cTypeIndex[ict], 1], c=colors[ict], label=ct, s=50)
    axes.legend(fontsize=10, bbox_to_anchor=(1.7, 0.2), loc='right', borderaxespad=0.)
    fig2.savefig(outdir+'UMAP_{}.tif'.format(dataname), bbox_inches='tight',dpi=600)
    return
#
#
dataname = "Splenocyte"#Leukemia-GM12878HEK-GM12878HL-InSilico-Splenocyte-Forebrain
methodname = "knnsmoothing"

indir = "E:/scATAC_impute/cluster/{}/".format(methodname)
cellinfo_dir = "E:/scATAC_impute/data_ATAC/{}/filtered_cells.txt".format(dataname)
outdir = "E:/scATAC_impute/UMAP/{}/".format(methodname)
norm_method = "zscore"
run_umap(indir=indir, cellinfo=cellinfo_dir, rand_stat=0, norm_method= "zscore",dataname=dataname)
#
#
#
