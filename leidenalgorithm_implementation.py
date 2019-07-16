#!/usr/bin/env python

import pandas as pd
import numpy as np
import loompy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import matplotlib as mpl
import loompy
import scipy.sparse as sparse
import leidenalg as la
import igraph as ig
import time
import json
import os
import errno
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
sys.path.append('/home/bojk/Data/minimeta_pyfiles/')

mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['figure.titlesize'] = 30
mpl.rcParams['legend.fontsize'] = 14

"""
Created 2019-06-19, Bojk Berghuis

preprocessing()     Go from counttable to distance matrix (cell distance). At this point pairwise Pearson correlation of 
                                logtransformed counts ENSG genes only, no selection. This might need further improvement or change.
run_leiden_all()     gets nearest neigbor pairs from precomputed distance matrix, forms clusters based on 
                                these pairs and a resolution parameter. 
                                Plots and saves figures to 'savedir'.
preprocessing_normCounttab()        similar to preprocessing, but with a normalized counttable as input.
"""

def run_leiden_all(distance_matrix,meta_tSNE,**kwargs):
    NNeighbors = kwargs['k']
    savedir = kwargs['save_dir']
    
    ind,vald,pairs,revd = nearestNeighbors(distance_matrix, NNeighbors)
    meta_tSNE,upairs = leiden_clustering(meta_tSNE, pairs,distance_matrix,**kwargs)
    plot_leiden_clusters(meta_tSNE,upairs,revd,**kwargs)
    meta_tSNE.to_csv(savedir+'/metadata_leiden.csv')
    return(meta_tSNE,pairs)
    

    
def preprocessing(countTab_path,meta_tSNE):
    counttable = pd.read_csv(countTab_path,index_col=0) #e.g. 'Datasets/TuPaMetaDataDivya/CombinedCountTable.csv'
    entire_start = time.time()
    print('----------------------------------------------------')
    print('Loaded counttable, started processing')
    hour = time.localtime()[3];minute = time.localtime()[4]
    print('Local time: '+str(hour)+':'+str(minute))
    print('----------------------------------------------------')
    norm_all = counttable.drop(counttable[counttable.T.sum()==0].index,axis=0).divide(meta_tSNE.nReads.divide(1e6))
    normlog2 = np.log2(norm_all+1)
    normlog2_nona = normlog2.dropna(how='all',axis=1)
    normENS = normlog2_nona[normlog2_nona.index.str.contains('ENSG')].copy()
    normENS.drop(normENS[normENS.sum(axis=1).sort_values()==0].index,inplace=True)
    distance_matrix = normENS.corr()
    
    entire_end = time.time()
    hour = time.localtime()[3]
    minute = time.localtime()[4]
    print('----------------------------------------------------')
    print('Processing counttable and computing distance matrix took {:.2f} seconds'.format((entire_end - entire_start)))
    print('Finished at local time: '+str(hour)+':'+str(minute))
    print('----------------------------------------------------')
    return distance_matrix

def preprocessing_normCounttab(countTab_norm):
    entire_start = time.time()
    print('----------------------------------------------------')
    print('Started processing counttable')
    hour = time.localtime()[3];minute = time.localtime()[4]
    print('Local time: '+str(hour)+':'+str(minute))
    print('----------------------------------------------------')
    norm_all = countTab_norm.drop(countTab_norm[countTab_norm.T.sum()==0].index,axis=0)
    normlog2 = np.log2(norm_all+1)
    normlog2_nona = normlog2.dropna(how='all',axis=1)
    normENS = normlog2_nona
    normENS.drop(normENS[normENS.sum(axis=1).sort_values()==0].index,inplace=True)
    distance_matrix = normENS.corr()
    
    entire_end = time.time()
    hour = time.localtime()[3]
    minute = time.localtime()[4]
    print('----------------------------------------------------')
    print('Processing counttable and computing distance matrix took {:.2f} seconds'.format((entire_end - entire_start)))
    print('Finished at local time: '+str(hour)+':'+str(minute))
    print('----------------------------------------------------')
    return distance_matrix

def nearestNeighbors(distance_matrix, NNeighbors):
    start = time.time()
    print('----------------------------------------------------')
    print('Calculating '+str(NNeighbors)+' nearest neigbors for each cell')
    hour = time.localtime()[3];minute = time.localtime()[4]
    print('Local time: '+str(hour)+':'+str(minute))
    print('----------------------------------------------------')
    newkeys = list(range(len(distance_matrix.index)))
    keys = distance_matrix.index
    celldict = dict(zip(keys,newkeys))
    revdict = dict(zip(newkeys,keys))
    distance_matrix_num = distance_matrix.rename(celldict).rename(celldict,axis=1).copy()
    indices = [];values = [];pairs=[]
    for cell in distance_matrix_num.index:
        valshort = list(distance_matrix_num.loc[cell].sort_values(ascending=False)[1:NNeighbors].values)
        valthresidx = [valshort.index(x) for x in valshort if x>.2]
        inxshort = list(distance_matrix_num.loc[cell].sort_values(ascending=False)[1:NNeighbors].index)
        idxthres = [inxshort[x] for x in valthresidx] # add only pairs with correlation >.2
        indices.append(idxthres)
        values.append(valshort)
        for i in range(len(idxthres)):
            pairs.append((cell,indices[cell][i]))
    #gra = ig.Graph(pairs)
    indict = dict(zip(newkeys,indices))
    valdict = dict(zip(newkeys,values))
    end = time.time();hour = time.localtime()[3];minute = time.localtime()[4]
    print('----------------------------------------------------')
    print('Collected list of pairs, this took {:.2f} seconds'.format((end - start)))
    print('Finished at local time: '+str(hour)+':'+str(minute))
    print('----------------------------------------------------')
    return indict,valdict,pairs,revdict

def leiden_clustering(meta_tSNE, pairs, distance_matrix,**kwargs):
    resolution_parameter = kwargs['resolution_parameter']
    annotation = kwargs['annot_col']
    start = time.time()
    print('----------------------------------------------------')
    print('Calculating new clusters from pairs with resolution parameter = '+str(resolution_parameter))
    hour = time.localtime()[3];minute = time.localtime()[4]
    print('Local time: '+str(hour)+':'+str(minute))
    print('----------------------------------------------------')
    unique_pairs = [tuple(x) for x in set([frozenset(x) for x in pairs])]
    gra = ig.Graph(unique_pairs)
    numID = list(range(len(meta_tSNE[annotation].unique())))
    lut = dict(zip(meta_tSNE[annotation].unique(),numID))
    meta_tSNE['cellID'] = meta_tSNE[annotation].map(lut)
    meta_tSNE.loc[distance_matrix.index,'betweenness'] = gra.betweenness()
    meta_tSNE.loc[distance_matrix.index,'degree'] = gra.degree()
    Orig_ID = list(meta_tSNE.loc[distance_matrix.index,'cellID'])
    #respar = 0.0001
    partition = la.CPMVertexPartition(gra, resolution_parameter=resolution_parameter, initial_membership=Orig_ID)
    la.Optimiser().optimise_partition(partition)
    clusters = partition.membership
    meta_tSNE.loc[distance_matrix.index,'new_clusters'] = clusters
    end = time.time();hour = time.localtime()[3];minute = time.localtime()[4]
    print('----------------------------------------------------')
    print('Assigned '+str(len(meta_tSNE.new_clusters.unique()))+' unique clusters in {:.2f} seconds'.format((end - start)))
    print('Finished at local time: '+str(hour)+':'+str(minute))
    print('----------------------------------------------------')
    return(meta_tSNE,unique_pairs)
    
    
def plot_leiden_clusters(meta_tSNE,unique_pairs,revdict,**kwargs):
    NN = kwargs['k']
    respar = kwargs['resolution_parameter']
    savedir = kwargs['save_dir']
    tsnex = kwargs['tsnex']
    tsney = kwargs['tsney']
    subset = kwargs['color_subset_only'] # True==color only the top 15 largest clusters, others gray
    num = kwargs['num_of_colors']
    print('----------------------------------------------------')
    print("Started plotting, saving to '"+savedir+"'")
    print('----------------------------------------------------')

    if subset==False:
        keys = list(np.sort(meta_tSNE['new_clusters'].unique())); values = sns.color_palette('cubehelix',len(keys))
        lut = dict(zip(keys,values))
        colors = meta_tSNE.new_clusters.map(lut)
    elif subset==True:
        keys = list(meta_tSNE['new_clusters'].value_counts()[:num].index); values = sns.color_palette('cubehelix',len(keys))
        meta_tSNE['subset_color'] = meta_tSNE['new_clusters']
        for idx in meta_tSNE.index:
            if meta_tSNE.loc[idx,'new_clusters'] not in keys:
                meta_tSNE.loc[idx,'subset_color'] = 9999
        keys.append(9999);values.append('gray')
        lut = dict(zip(keys,values))
        colors = meta_tSNE.subset_color.map(lut)
        tot = str(len(meta_tSNE.new_clusters.unique()))
    
    f = plt.figure()
    gs = gridspec.GridSpec(1,1)
    ax = f.add_subplot(gs[0,0])
    meta_tSNE.plot.scatter(tsnex,tsney,figsize=(10,10),ax=ax,color=colors,s=40,alpha=.7)#,label=colors.keys)s=meta_tSNE.nGene.divide(80)
    for x,y in lut.items():
        plt.bar(0,0,color=y,label=x,alpha=1)
        handles, labels = ax.get_legend_handles_labels()
        #plt.legend(handles[:],labels[:],bbox_to_anchor=(-0.0, 1.08, 1., .102), loc=2,
        plt.legend(handles[:],labels[:],bbox_to_anchor=(1, .9, .3, .102), loc=2,
                   ncol=1, mode="expand",fontsize=15)
    plt.yticks([]);plt.xticks([])
    if subset==False:
        plt.title('Leiden Clusters (res param = '+str(respar)+')')
    else:
        plt.title('Leiden Clusters (res param = '+str(respar)+'),\ntop '+str(num)+' out of '+tot+' clusters colored')
    plt.gcf().subplots_adjust(left=.05,right=0.78)
    plt.show()
    if subset==False:
        f.savefig(savedir+'/tSNE_newLeidenClusters_resPar='+str(respar)+'NN='+str(NN)+'.png')
    else:
        f.savefig(savedir+'/tSNE_newLeidenClusters_resPar='+str(respar)+'NN='+str(NN)+'_colorTop'+str(num)+'.png')
    

    plot_edges = kwargs['plotEdges']
    if plot_edges==True:

        f,ax = plt.subplots()
        meta_tSNE.plot.scatter(tsnex,tsney,figsize=(12,10),ax=ax,s=meta_tSNE['degree'].divide(.8)
                               ,alpha=.8,cmap='RdBu_r',c=np.log2(meta_tSNE['degree']))
        plt.yticks([]);plt.xticks([])
        plt.title('log2(number of edges)')
        plt.gcf().subplots_adjust(right=0.88)
        plt.show()
        f.savefig(savedir+'/tSNE_LeidenDegree_NN='+str(NN)+'.png')

        f,ax = plt.subplots()
        for pair in unique_pairs:
            xy1 = meta_tSNE.loc[revdict[pair[0]],[tsnex,tsney]]
            xy2 = meta_tSNE.loc[revdict[pair[1]],[tsnex,tsney]]
            ax.plot([xy1[0],xy2[0]],[xy1[1],xy2[1]],alpha=.05
                    ,c='gray')

        meta_tSNE.plot.scatter(tsnex,tsney,figsize=(12,10),ax=ax,s=meta_tSNE.degree.divide(.8)
                               ,alpha=1,cmap='RdBu_r',c=np.log2(meta_tSNE.degree))
        #plt.yticks([]);plt.xticks([])
        plt.title('log2(number of edges)')
        plt.gcf().subplots_adjust(right=0.88)
        plt.show()
        f.savefig(savedir+'/tSNE_LeidenDegree_allEdges_NN='+str(NN)+'.png')
    
    hour = time.localtime()[3];minute = time.localtime()[4]
    print('----------------------------------------------------')
    print('Finished at local time: '+str(hour)+':'+str(minute))
    print('----------------------------------------------------')
    
    
    
def plot_fano(normCount,savedir):
    # input: normalized count table 
    yax = normCount.T.var().divide(normCount.T.mean())
    xax = normCount.T.mean()
    
    # selection 
    selection = yax[yax>=500]
    exclude = ['ERCC','TRA','TRB','HLA','HLB','HLC','RPL','RPS','IGV','IGD','IGJ']

    exclist = []
    for partstr in exclude:
        if partstr in selection.index:
            exclist += list(selection[selection.index.str.contains(partstr)].index)

    selection = selection[~selection.index.isin(exclist)]
    idx = selection.index

    # start plotting 
    f = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(6,6)

    ax1 = f.add_subplot(gs[1:,:-1])
    ercc = xax[xax.index.str.contains("ERCC")].index
    ax1.axvspan(10,20,color='gray',alpha=.3,label=None)
    ax1.scatter(xax,yax,alpha=.2,label=None)
    ax1.scatter(xax.loc[selection.index],yax.loc[selection.index],alpha=.3,color='g',label='selected genes for dim reduction')
    ax1.scatter(xax.loc[ercc],yax.loc[ercc],alpha=.6,color='orange',label='ERCCs')
    ax1.set_xscale('log');ax1.set_yscale('log')
    ax1.set_xlim(1e-4,1e4)
    ax1.set_ylim(1,1e6)
    ax1.axhline(5e2,color='k',label=None)
    ax1.legend()

    ax2 = f.add_subplot(gs[0,:-1])
    bins = np.logspace(-4,4,40)
    counts, bin_edges = np.histogram(xax,bins=bins)
    exp_centers = []
    for i in range(len(counts)):
        exp_centers.append((np.log10(bin_edges)[i]+np.log10(bin_edges)[i+1])/2)
    bin_centers = [10**f for f in exp_centers]


    ax2.step(bin_edges[1:],counts,linewidth=3)
    ax2.set_xlim(1e-4,1e4)
    ax2.set_xscale('log')#,ax2.set_yscale('log')
    ax2.set_xticks([])
    ax2.axvspan(10,20,color='gray',alpha=.3,label=None)
    ax3 = f.add_subplot(gs[1:,-1])
    bins2 = np.logspace(0,6,30)
    counts, bin_edges = np.histogram(yax,bins=bins2)

    #counts = np.histogram(xax,bins=bins)
    ax3.hist(yax,bins=bins2,orientation='horizontal',histtype='step',linewidth=3)
    ax3.set_ylim(1,1e6)#,ax3.set_xlim(-2,10)
    ax3.set_yscale('log')#;ax3.set_xscale('log')
    ax3.set_yticks([])
    ax3.axhline(5e2,color='k')

    ax1.set_ylabel('Variance/mean')
    ax1.set_xlabel('mean')
    plt.show()
    f.savefig(savedir+'/fano.png')
    
    return idx


"""
SemiAnnotate scripting 2019-07-01
might turn into separate file - or not
"""

def normalize_UMGs(counttable,metadata):
    print('-------------------------------------------')
    print('genes to be removed from counttable:') # these counttables do not contain other than uniquely mapping genes (UMGs)
    l1 = list(counttable[counttable.index.str.contains('ERCC')].index)
    l2 = list(counttable[counttable.index.str.contains('_')].index)
    print(l1+l2)
    print('-------------------------------------------')
    exclude = list(set(l1+l2))
    genes_in = counttable[~counttable.index.isin(exclude)].index
    #print(genes_in)
    metadata['n_reads'] = counttable.loc[genes_in].sum()
    counttable_norm = counttable.divide(metadata['n_reads']).multiply(1e6).copy()
    return(counttable_norm,metadata)

def normmerge_twoCounttables(counttable_atlas,metadata_atlas,counttable_new,metadata_new,**kwargs):
    """
    IN: raw counttables
    """
    counts_atlas_norm,metadata_atlas = normalize_UMGs(counttable_atlas,metadata_atlas)
    counts_new_norm,metadata_new = normalize_UMGs(counttable_new,metadata_new)

    cellT = kwargs['cell type column']
    atlas_weights = kwargs['weights_atlas_cells']
    savedir = kwargs['savedir']
    date = kwargs['timestamp']

    print('-------------------------------------------')
    print('Cell types in atlas:')
    print(np.sort(metadata_atlas[cellT].unique()))
    print('-------------------------------------------')
    print('Cell types in new data:')
    print(np.sort(metadata_new[cellT].unique()))
    print('-------------------------------------------')

    # make average counttable
    atlas_cells = np.sort(metadata_atlas[cellT].unique())
    atlas_countmean = pd.DataFrame(index=counts_atlas_norm.index,columns=atlas_cells)
    #atlas_countmean = atlas_countmean.copy()
    for celltype in atlas_cells:
        cells = metadata_atlas[metadata_atlas[cellT]==celltype].index
        atlas_countmean[celltype] = counts_atlas_norm[cells].mean(axis=1)
        #BA_countfrac[celltype] = (BA_countNorm[cells] == 0).astype(int).sum(axis=1).divide(len(cells))

    matrix = atlas_countmean.join(counts_new_norm,how='inner')
    weights = np.array(list(metadata_atlas[cellT].value_counts().sort_index().values)+list(np.ones(len(metadata_new),int)))
    weights = np.array(list(np.ones(len(metadata_atlas[cellT].value_counts()))*atlas_weights)+list(np.ones(len(metadata_new),int)))
    # write params to json in new folder with date timestamp
    output_file = savedir+date+'/parameters_mergeCounttables_'+date+'.json'
    if not os.path.exists(os.path.dirname(output_file)):
        try:
            os.makedirs(os.path.dirname(output_file))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(output_file, 'w') as file:
        file.write(json.dumps(kwargs))
        file.close()

    return matrix, weights,metadata_atlas,metadata_new

def feature_select(matrix,metadata_atlas,**kwargs):
    #nf1 = 30 # number of features per cell type (atlas)
    #nf2 = 500 # number of overdispersed features
    cellT = kwargs['cell type column']
    nf1 = kwargs['number of features cell type']
    nf2 = kwargs['number of features new data']
    atlas_cells = np.sort(metadata_atlas[cellT].unique())
    n_fixed = len(atlas_cells)

    features = set()

    # Atlas markers
    if n_fixed > 1 and nf1>0:
        for icol in range(n_fixed):
            ge1 = matrix.iloc[:, icol]
            ge2 = (matrix.iloc[:, :n_fixed].sum(axis=1) - ge1).divide(n_fixed - 1)
            fold_change = np.log2(ge1 + 0.1) - np.log2(ge2 + 0.1)
            markers = np.argpartition(fold_change, -nf1)[-nf1:]
            features |= set(markers)
        features2 = features

        print('-------------------------------------------')
        print('Selected number of genes from atlas:')
        print(len(features))
    else:
        print('no atlas genes or classes included')
    # Unbiased on new data
    nd_mean = matrix.iloc[:, n_fixed:].mean(axis=1)
    nd_var = matrix.iloc[:, n_fixed:].var(axis=1)
    fano = (nd_var + 1e-10).divide(nd_mean + 1e-10)
    overdispersed = np.argpartition(fano, -nf2)[-nf2:]
    features |= set(overdispersed)
    print('-------------------------------------------')
    print('Selected number of genes from new cells:')
    print(len(set(overdispersed)))

    print('-------------------------------------------')
    print('Combined total of selected genes:')
    print(len(set(features)))
    print('-------------------------------------------')

    matrix_Feature_selected = matrix.iloc[list(features),:].copy()

    return matrix_Feature_selected

def weighted_PCA(feature_selected_matrix,weights,n_pcs,n_fixed):
    print('-------------------------------------------')
    print('perfoming weighted PCA')
    from scipy.sparse.linalg import eigsh
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform

    metric = 'correlation'
    matrix = feature_selected_matrix.values

    # 1. whiten
    weight = 1.0 * weights / weights.sum()
    mean_w = matrix @ weight
    var_w = (matrix**2) @ weight
    std_w = np.sqrt(var_w)
    matrix_w = ((matrix.T - mean_w) / std_w).T

    # take care of non-varying components
    matrix_w[np.isnan(matrix_w)] = 0
    # 2. weighted covariance
    # This matrix has size L x L. Typically L ~ 500 << N, so the covariance
    # L x L is much smaller than N x N, hence it's fine
    cov_w = matrix_w @ np.diag(weight) @ matrix_w.T
    # 3. PCA
    # lvects columns are the left singular vectors L x L (gene loadings)
    evals, lvects = eigsh(cov_w, k=n_pcs)
    # calculate the right singular vectors N x N given the left singular
    # vectors and the singular values svals = np.sqrt(evals)
    # NOTE: this is true even if we truncated the PCA via n_pcs << L
    # rvects columns are the right singular vectors
    svals = np.sqrt(evals)
    rvects = matrix_w.T @ lvects @ np.diag(1.0 / svals)
    princdf = pd.DataFrame(index=feature_selected_matrix.columns,data=rvects)

    #4. distance matrix
    #rvects_free = rvects[n_fixed:]
    distvector = pdist(rvects, metric=metric)
    distance_matrix = squareform(distvector)

    return princdf,distance_matrix

def unweighted_PCA(data,n_pcs,n_fixed):
    print('-------------------------------------------')
    print('perfoming UNweighted PCA')
    from scipy.sparse.linalg import eigsh
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform
    metric = 'correlation'
    matrix = data.values

    # 1. whiten
    weight = np.ones(len(data.T))
    mean_w = matrix @ weight
    var_w = (matrix**2) @ weight
    std_w = np.sqrt(var_w)
    matrix_w = ((matrix.T - mean_w) / std_w).T

    # take care of non-varying components
    matrix_w[np.isnan(matrix_w)] = 0
    # 2. weighted covariance
    # This matrix has size L x L. Typically L ~ 500 << N, so the covariance
    # L x L is much smaller than N x N, hence it's fine
    cov_w = matrix_w @ np.diag(weight) @ matrix_w.T
    # 3. PCA
    # lvects columns are the left singular vectors L x L (gene loadings)
    evals, lvects = eigsh(cov_w, k=n_pcs)
    # calculate the right singular vectors N x N given the left singular
    # vectors and the singular values svals = np.sqrt(evals)
    # NOTE: this is true even if we truncated the PCA via n_pcs << L
    # rvects columns are the right singular vectors
    svals = np.sqrt(evals)
    rvects = matrix_w.T @ lvects @ np.diag(1.0 / svals)
    princdf = pd.DataFrame(index=data.columns,data=rvects)

    #4. distance matrix
    #rvects_free = rvects[n_fixed:]
    distvector = pdist(rvects, metric=metric)
    distance_matrix = squareform(distvector)

    """x = StandardScaler().fit_transform(data)
    pca = PCA(n_components=n_pcs)
    principalComp = pca.fit_transform(x)
    princdf = pd.DataFrame(principalComp)
    princdf.index =data.index"""
    return princdf,distance_matrix

def perform_tSNE(pca_df, perplexity=None):
    print('-------------------------------------------')
    print('perfoming tSNE')
    if perplexity==None:
        perplexity = 20
        print('assigned default perplexity of 20')

    x_emb = TSNE(n_components=2,perplexity=perplexity,random_state=2).fit_transform(pca_df.values)
    tsnedf = pd.DataFrame(x_emb,index=pca_df.index)
    print('tSNE done.')
    print('-------------------------------------------')
    return tsnedf

def semiAnnotate_to_pca_to_tsnedf(feature_selected_matrix,weights,atlas_metadata,new_metadata,**kwargs):
    import semiannotate

    cellT = kwargs['cell type column']
    n_fixed = len(atlas_metadata[cellT].unique())
    thresn = kwargs['threshold_neigborhood']
    n_pcs= kwargs['n_pcs']
    respar = kwargs['resolution_parameter']
    selfEdge = kwargs['self_edging']

    #instantiate class
    sa = semiannotate.SemiAnnotate(
        feature_selected_matrix.values,
        sizes=weights,
        n_fixed=n_fixed,
        n_neighbors=5,
        n_pcs=n_pcs,
        distance_metric='correlation',
        threshold_neighborhood=thresn,
        resolution_parameter=respar
        )

    sa(select_features=False)
    if selfEdge==True:
        sa.compute_neighbors()
        #sa.compute_communities(_self_loops=True, _node_sizes=True)
        sa.compute_communities()
    new_metadata['new_class'] = pd.Series(index=feature_selected_matrix.columns[n_fixed:],data=sa.membership)

    # weighted PCA
    weight_PCA,wdistmat = weighted_PCA(feature_selected_matrix,weights,n_pcs,n_fixed)
    weighted_tSNE = perform_tSNE(weight_PCA)
    weighted_tSNE.rename(index=str,columns={0:'wDim1',1:'wDim2'},inplace=True)

    # unweighted PCA
    normal_PCA,udistmat = unweighted_PCA(feature_selected_matrix,n_pcs,n_fixed)

    Unweighted_tSNE = perform_tSNE(normal_PCA,20)
    Unweighted_tSNE.rename(index=str,columns={0:'uDim1',1:'uDim2'},inplace=True)

    weighted_tSNE['class'] = ''
    weighted_tSNE['original_membership'] = new_metadata[cellT]
    weighted_tSNE.iloc[n_fixed:,2]= sa.membership
    weighted_tSNE.iloc[:n_fixed,2] = range(n_fixed)

    a = np.array(list(set(sa.membership)))
    class_numbers = list(range(max(a+1)))
    new_classes = a[a>(n_fixed-1)]
    vals = list(weighted_tSNE.index[:n_fixed])
    for i in range(len(new_classes)):
        vals.append('newClass'+str(i+1))
    lutcl = dict(zip(class_numbers,vals))
    weighted_tSNE['new_membership'] = weighted_tSNE['class'].map(lutcl)
    tsne_df = weighted_tSNE.join(Unweighted_tSNE[['uDim1','uDim2']])

    return tsne_df,class_numbers,vals,wdistmat

def semiAnnotate_using_tsnedf(feature_selected_matrix,weights,atlas_metadata,new_metadata,tsnedf,**kwargs):
    import semiannotate

    cellT = kwargs['cell type column']
    n_fixed = len(atlas_metadata[cellT].unique())
    thresn = kwargs['threshold_neigborhood']
    n_pcs= kwargs['n_pcs']
    respar = kwargs['resolution_parameter']
    selfEdge = kwargs['self_edging']

    #instantiate class
    sa = semiannotate.SemiAnnotate(
        feature_selected_matrix.values,
        sizes=weights,
        n_fixed=n_fixed,
        n_neighbors=5,
        n_pcs=n_pcs,
        distance_metric='correlation',
        threshold_neighborhood=thresn,
        resolution_parameter=respar
        )

    sa(select_features=False)
    if selfEdge==True:
        sa.compute_neighbors()
        #sa.compute_communities(_self_loops=True, _node_sizes=True)
        sa.compute_communities()
    new_metadata['new_class'] = pd.Series(index=feature_selected_matrix.columns[n_fixed:],data=sa.membership)


    tsnedf['class'] = ''
    tsnedf.iloc[n_fixed:,2]= sa.membership
    tsnedf.iloc[:n_fixed,2] = range(n_fixed)

    a = np.array(list(set(sa.membership)))
    class_numbers = list(range(max(a+1)))
    new_classes = a[a>(n_fixed-1)]
    vals = list(tsnedf.index[:n_fixed])
    for i in range(len(new_classes)):
        vals.append('newClass'+str(i+1))
    lutcl = dict(zip(class_numbers,vals))
    tsnedf['new_membership'] = tsnedf['class'].map(lutcl)
    tsne_df = tsnedf.copy()#

    return tsne_df,class_numbers,vals


def make_pairs(distmat,threshold,max_neighbors):
    """
    returns the edges (based on distance matrix indices) with the closest distance
    that exist below a certain cutoff threshold, for a maximum of 'max_neighbors' edges
    """
    corr = str((1-threshold)*100)
    print('---------------------------------------')
    print('Making list of edges with '+corr+'% correlation and up')
    print('Max '+str(max_neighbors)+' edges per cell.')
    pairs = []
    for cell in distmat.index:
        temp = distmat.loc[cell].sort_values()[1:max_neighbors]
        neigh = temp[temp<threshold].index
        if len(neigh)>0:
            for pa in neigh:
                pairs.append((cell,pa))
    print('Found '+str(len(pairs))+' edges.')
    print('---------------------------------------')
    return pairs
