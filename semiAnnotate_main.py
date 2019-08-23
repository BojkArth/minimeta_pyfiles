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
import time
import json
import os
import errno
from sklearn.manifold import TSNE
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA

import semiannotate
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
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
Created 2019-07-01, Bojk Berghuis



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


def normmerge_twoCounttables_subsample(counttable_atlas,metadata_atlas,counttable_new,metadata_new,**kwargs):
    """
    IN: raw counttables
    """
    counts_atlas_norm,metadata_atlas = normalize_UMGs(counttable_atlas,metadata_atlas)
    counts_new_norm,metadata_new = normalize_UMGs(counttable_new,metadata_new)

    cellT = kwargs['cell type column']
    atlas_subsamples = kwargs['number_of_cells_per_type']
    savedir = kwargs['savedir']
    date = kwargs['timestamp']

    print('-------------------------------------------')
    print('Cell types in atlas:')
    print(np.sort(metadata_atlas[cellT].unique()))
    print('-------------------------------------------')
    print('Cell types in new data:')
    print(np.sort(metadata_new[cellT].unique()))
    print('-------------------------------------------')

    # make subsampled counttable
    atlas_cells = np.sort(metadata_atlas[cellT].unique())
    nums = list(range(len(atlas_cells)))
    celltype_dict = dict(zip(atlas_cells,nums))
    total_cells = len(atlas_cells)*atlas_subsamples
    atlas_counttable = None
    atlas_annotations = []
    for celltype in atlas_cells:
        cellsidx = metadata_atlas[metadata_atlas[cellT]==celltype].index
        randcells = np.random.randint(0,len(cellsidx),size=atlas_subsamples)
        cells = np.array(cellsidx)[randcells]
        atlas_annotations.append(list(np.ones(atlas_subsamples)*celltype_dict[celltype]))
        if atlas_counttable is None:
            atlas_counttable = counts_atlas_norm[cells].copy()
        else:
            atlas_counttable = atlas_counttable.join(counts_atlas_norm[cells])

    atlas_annotations = [int(i) for subl in atlas_annotations for i in subl]
    matrix = atlas_counttable.join(counts_new_norm,how='inner')

    # write params to json in new folder with date timestamp
    output_file = savedir+date+'/parameters_mergeCounttables_subsample_'+date+'.json'
    if not os.path.exists(os.path.dirname(output_file)):
        try:
            os.makedirs(os.path.dirname(output_file))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(output_file, 'w') as file:
        file.write(json.dumps(kwargs))
        file.close()

    return matrix, atlas_annotations, celltype_dict ,metadata_atlas,metadata_new

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
    savedir = kwargs['savedir']
    date = kwargs['timestamp']
    matrix_Feature_selected.to_csv(savedir+date+'/feature_selected_matrix+'+date+'.csv')
    return matrix_Feature_selected

def feature_select_sub(matrix,annotations,metadata_atlas,**kwargs):
    #nf1 = 30 # number of features per cell type (atlas)
    #nf2 = 500 # number of overdispersed features
    #cellT = kwargs['cell type column']
    nf1 = kwargs['number of features cell type']
    nf2 = kwargs['number of features new data']
    #atlas_cells = np.sort(metadata_atlas[cellT].unique())
    n_fixed = len(annotations)
    print(n_fixed)
    aa = annotations
    features = set()

    aau = np.unique(annotations)
    # Atlas markers
    if len(aau) > 1:
        for au in aau:
            icol = (aa == au).nonzero()[0]
            li = len(icol)
            ge1 = matrix.iloc[:, icol].mean(axis=1)
            ge2 = (matrix.iloc[:, :n_fixed].sum(axis=1) - ge1 * li).divide(n_fixed - li)
            fold_change = np.log2(ge1 + 0.1) - np.log2(ge2 + 0.1)
            markers = np.argpartition(fold_change, -nf1)[-nf1:]
            features |= set(markers)

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
    features = list(features)

    print('-------------------------------------------')
    print('Selected number of genes from new cells:')
    print(len(set(overdispersed)))

    print('-------------------------------------------')
    print('Combined total of selected genes:')
    print(len(set(features)))
    print('-------------------------------------------')

    matrix_Feature_selected = matrix.iloc[features,:].copy()
    savedir = kwargs['savedir']
    date = kwargs['timestamp']
    matrix_Feature_selected.to_csv(savedir+date+'/feature_selected_matrix_subsample+'+date+'.csv')
    return matrix_Feature_selected



def weighted_PCA(feature_selected_matrix,sizes,n_pcs,n_fixed):
    print('-------------------------------------------')
    print('performing weighted PCA')

    metric = 'correlation'
    matrix = feature_selected_matrix.values

        # 0. take log
    matrix = np.log10(matrix + 0.1)

    # 1. whiten
    weights = 1.0 * sizes / sizes.sum()
    mean_w = matrix @ weights
    var_w = ((matrix.T - mean_w)**2).T @ weights
    std_w = np.sqrt(var_w)
    Xnorm = ((matrix.T - mean_w) / std_w).T

    # take care of non-varying components
    Xnorm[np.isnan(Xnorm)] = 0

    # 2. weighted covariance
    # This matrix has size L x L. Typically L ~ 500 << N, so the covariance
    # L x L is much smaller than N x N, hence it's fine
    cov_w = np.cov(Xnorm, fweights=sizes)

    # 3. PCA
    # rvects columns are the right singular vectors
    evals, evects = np.linalg.eig(cov_w)
    # sort by decreasing eigenvalue (explained variance) and truncate
    ind = evals.argsort()[::-1][:n_pcs]
    # NOTE: we do not actually need the eigenvalues anymore
    lvects = evects.T[ind]

    # calculate right singular vectors given the left singular vectors
    # NOTE: this is true even if we truncated the PCA via n_pcs << L
    # rvects columns are the right singular vectors
    rvects = (lvects @ Xnorm).T

    # expand embedded vectors to account for sizes
    # NOTE: this could be done by carefully tracking multiplicities
    # in the neighborhood calculation, but it's not worth it: the
    # amount of overhead memory used here is small because only a few
    # principal components are used
    Ne = int(np.sum(sizes))
    rvectse = np.empty((Ne, n_pcs))
    i = 0
    for isi, size in enumerate(sizes):
        for j in range(int(size)):
            rvectse[i] = rvects[isi]
            i += 1

    princdf = pd.DataFrame(index=feature_selected_matrix.columns,data=rvectse)

    return princdf

def unweighted_PCA(data,n_pcs,n_fixed):
    print('-------------------------------------------')
    print('performing UNweighted PCA')

    metric = 'correlation'
    matrix = data.values

        # 0. take log
    matrix = np.log10(matrix + 0.1)

    # 1. whiten
    Xnorm = ((matrix.T - matrix.mean(axis=1)) / matrix.std(axis=1, ddof=0)).T

    # take care of non-varying components
    Xnorm[np.isnan(Xnorm)] = 0

    # 2. PCA
    pca = PCA(n_components=n_pcs,random_state=24870)
    # rvects columns are the right singular vectors
    rvects = pca.fit_transform(Xnorm.T)
    princdf = pd.DataFrame(index=data.columns,data=rvects)

    # 4. calculate distance matrix
    # rvects is N x n_pcs. Let us calculate the end that includes only the free
    # observations and then call cdist which kills the columns. The resulting
    # matrix has dimensions N1 x N
    distance_matrix = squareform(pdist(rvects, metric=metric))
    #dmax = dmat.max()

    """
    # 1. whiten
    weight = np.ones(len(data.T))
    mean_w = matrix @ weight
    var_w = (matrix**2-mean_w) @ weight
    std_w = np.sqrt(var_w)
    matrix_w = ((matrix.T - mean_w) / std_w).T

    # take care of non-varying components
    matrix_w[np.isnan(matrix_w)] = 0
    # 2. weighted covariance
    cov_w = matrix_w @ np.diag(weight) @ matrix_w.T
    # 3. PCA
    evals, lvects = eigsh(cov_w, k=n_pcs)
    # calculate the right singular vectors N x N given the left singular
    # vectors and the singular values svals = np.sqrt(evals)
    # rvects columns are the right singular vectors
    svals = np.sqrt(evals)
    rvects = matrix_w.T @ lvects @ np.diag(1.0 / svals)
    princdf = pd.DataFrame(index=data.columns,data=rvects)

    #4. distance matrix
    #rvects_free = rvects[n_fixed:]
    distvector = pdist(rvects, metric=metric)
    distance_matrix = squareform(distvector)
    """

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
    cellT = kwargs['cell type column']
    n_fixed = len(atlas_metadata[cellT].unique())
    thresn = kwargs['threshold_neigborhood']
    n_pcs= kwargs['n_pcs']
    respar = kwargs['resolution_parameter']
    selfEdge = kwargs['self_edging']
    savedir = kwargs['savedir']
    date = kwargs['timestamp']

    #instantiate class
    sa = semiannotate.SemiAnnotate(
        feature_selected_matrix.values,
        sizes=weights,
        n_fixed=n_fixed,
        n_neighbors=35,
        n_pcs=n_pcs,
        distance_metric='correlation',
        threshold_neighborhood=thresn,
        resolution_parameter=respar
        )

    sa(select_features=False)
    if selfEdge==True:
        print('self-edging ON')
        sa.compute_neighbors()
        #sa.compute_communities(_self_loops=True, _node_sizes=True)
        sa.compute_communities()
    new_metadata['new_class'] = pd.Series(index=feature_selected_matrix.columns[n_fixed:],data=sa.membership)

    # weighted PCA
    """
    weight_PCA,wdistmat = weighted_PCA(feature_selected_matrix,weights,n_pcs,n_fixed)
    weighted_tSNE = perform_tSNE(weight_PCA)
    weighted_tSNE.rename(index=str,columns={0:'wDim1',1:'wDim2'},inplace=True)
    """

    # unweighted PCA
    normal_PCA,udistmat = unweighted_PCA(feature_selected_matrix,n_pcs,n_fixed)

    Unweighted_tSNE = perform_tSNE(normal_PCA,20)
    Unweighted_tSNE.rename(index=str,columns={0:'uDim1',1:'uDim2'},inplace=True)

    Unweighted_tSNE['class'] = ''
    Unweighted_tSNE['original_membership'] = new_metadata[cellT]
    Unweighted_tSNE.iloc[n_fixed:,2]= sa.membership
    Unweighted_tSNE.iloc[:n_fixed,2] = range(n_fixed)

    a = np.array(list(set(sa.membership)))
    class_numbers = list(range(max(a+1)))
    new_classes = a[a>(n_fixed-1)]
    vals = list(Unweighted_tSNE.index[:n_fixed])
    for i in range(len(new_classes)):
        vals.append('newClass'+str("{0:0=2d}".format(i+1)))
    lutcl = dict(zip(class_numbers,vals))
    Unweighted_tSNE['new_membership'] = Unweighted_tSNE['class'].map(lutcl)
    tsne_df = Unweighted_tSNE.copy()#join(Unweighted_tSNE[['uDim1','uDim2']])

    # write params to json in new folder with date timestamp
    output_file = savedir+date+'/parameters_semiAnnotate_'+date+'.json'
    if not os.path.exists(os.path.dirname(output_file)):
        try:
            os.makedirs(os.path.dirname(output_file))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(output_file, 'w') as file:
        file.write(json.dumps(kwargs))
        file.close()

    return tsne_df,class_numbers,vals,udistmat

def semiAnnotate_using_tsnedf(feature_selected_matrix,weights,atlas_metadata,new_metadata,tsnedf,**kwargs):

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
        n_neighbors=25,
        n_pcs=n_pcs,
        distance_metric='correlation',
        threshold_neighborhood=thresn,
        resolution_parameter=respar
        )


    if selfEdge==True:
        sa.compute_neighbors()
        sa.compute_communities(_self_loops=True, _node_sizes=True)
    else:
        sa(select_features=False)

    # this is incorrect, fix it
    new_metadata['new_class'] = pd.Series(index=feature_selected_matrix.columns[n_fixed:],data=sa.membership)


    tsnedf['class'] = ''
    tsnedf.iloc[n_fixed:,2]= sa.membership
    tsnedf.iloc[:n_fixed,2] = range(n_fixed)

    a = np.array(list(set(sa.membership)))
    class_numbers = list(range(max(a+1)))
    new_classes = a[a>(n_fixed-1)]
    vals = list(tsnedf.index[:n_fixed])
    for i in range(len(new_classes)):
        vals.append('newClass'+str("{0:0=2d}".format(i+1)))
    lutcl = dict(zip(class_numbers,vals))
    tsnedf['new_membership'] = tsnedf['class'].map(lutcl)
    tsne_df = tsnedf.copy()#

    return tsne_df,class_numbers,vals


def semiAnnotate_subsample(feature_selected_matrix,annot,atlas_metadata,new_metadata,**kwargs):
    cellT = kwargs['cell type column']
    n_fixed = len(annot)
    atlas_classes = np.unique(annot)
    thresn = kwargs['threshold_neigborhood']
    n_pcs= kwargs['n_pcs']
    respar = kwargs['resolution_parameter']
    selfEdge = kwargs['self_edging']
    savedir = kwargs['savedir']
    date = kwargs['timestamp']


    #instantiate class
    sa = semiannotate.Subsample(
        feature_selected_matrix.values,
        atlas_annotations=annot,
        n_neighbors=10,
        n_pcs=n_pcs,
        distance_metric='correlation',
        threshold_neighborhood=thresn,
        clustering_metric='cpm',
        resolution_parameter=respar
        )

    sa(select_features=False)
    if selfEdge==True:
        print('self-edging ON')
        sa.compute_neighbors()
        #sa.compute_communities(_self_loops=True, _node_sizes=True)
        sa.compute_communities()
    new_metadata['new_class'] = pd.Series(index=feature_selected_matrix.columns[n_fixed:],data=sa.membership)

    # weighted PCA
    #weight_PCA,wdistmat = weighted_PCA(feature_selected_matrix,weights,n_pcs,n_fixed)
    #weighted_tSNE = perform_tSNE(weight_PCA)
    #weighted_tSNE.rename(index=str,columns={0:'wDim1',1:'wDim2'},inplace=True)


    # unweighted PCA
    normal_PCA,udistmat = unweighted_PCA(feature_selected_matrix,n_pcs,n_fixed)

    Unweighted_tSNE = perform_tSNE(normal_PCA,20)
    Unweighted_tSNE.rename(index=str,columns={0:'uDim1',1:'uDim2'},inplace=True)

    Unweighted_tSNE['class'] = ''
    Unweighted_tSNE['original_membership'] = np.nan

    atlas_cells = list(np.sort(atlas_metadata[cellT].unique()))

    # fill in "original_membership" originals
    Unweighted_tSNE.iloc[:n_fixed,3] = annot
    # fill in annotated "class"
    a = np.array(list(set(sa.membership)))
    added_class_numbers = np.sort([int(f) for f in a if f not in atlas_classes])
    if len(added_class_numbers)>0:
        added_strnum = [str(f) for f in added_class_numbers]
        membership_list = atlas_cells+added_strnum
        for i in range(len(added_class_numbers)):
            atlas_cells.append('newClass'+str("{0:0=2d}".format(i+1)))
        nums = list(range(max(added_class_numbers)))
    else:
        nums = list(range(len(atlas_cells)))
        membership_list = atlas_cells
    classdict = dict(zip(membership_list,nums))
    memdict = dict(zip(nums,atlas_cells))
    Unweighted_tSNE.iloc[:n_fixed,2] = list(map(classdict.get,annot))
    Unweighted_tSNE.iloc[n_fixed:,2]= list(map(classdict.get,sa.membership))

    # fill in new class strings
    Unweighted_tSNE['new_membership'] = Unweighted_tSNE['class'].map(memdict)
    tsne_df = Unweighted_tSNE.copy()

    # write params to json in new folder with date timestamp
    output_file = savedir+date+'/parameters_semiAnnotate_'+date+'.json'
    if not os.path.exists(os.path.dirname(output_file)):
        try:
            os.makedirs(os.path.dirname(output_file))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(output_file, 'w') as file:
        file.write(json.dumps(kwargs))
        file.close()

    return tsne_df,nums,atlas_cells

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

def make_pairdf(dist_matrix,NN,tsne_df,colname):
    """ returns a dataframe with pairs and some properties
    such as inter or intra class edge, eucledian distance between the two in the tsne plane"""
    xcol = colname+'1'
    ycol = colname+'2'
    pairs = make_pairs(dist_matrix,2,NN)
    pair_df = pd.DataFrame(pairs)
    for pair in pair_df.index:
        cell1 = pair_df.loc[pair,0]
        cell2 = pair_df.loc[pair,1]
        if type(dist_matrix.index[0])==int:
            class1 = tsne_df.iloc[cell1]['new_membership']
            class2 = tsne_df.iloc[cell2]['new_membership']
        else:
            class1 = tsne_df.loc[cell1,'new_membership']
            class2 = tsne_df.loc[cell2,'new_membership']
        if class1==class2:
            pair_df.loc[pair,'edge_type'] = 'inter_class'
        else:
            pair_df.loc[pair,'edge_type'] = 'intra_class'
        pair_df.loc[pair,'correlation'] = 1 - dist_matrix.loc[cell1,cell2]
        pair_df.loc[pair,'distance'] = dist_matrix.loc[cell1,cell2]
        if  type(dist_matrix.index[0])==int:
            xy1 = tsne_df.iloc[cell1][[xcol,ycol]]
            xy2 = tsne_df.iloc[cell2][[xcol,ycol]]
        else:
            xy1 = tsne_df.loc[cell1][[xcol,ycol]]
            xy2 = tsne_df.loc[cell2][[xcol,ycol]]
        pair_df.loc[pair,'edge_length'] =  np.sqrt((xy2[0]-xy1[0])**2+(xy2[1]-xy1[1])**2)
    return pair_df


def plot_tsnes(tsnedf,class_numbers,class_labels,weights,new_metadata,colordict=None,**kwargs):

    date = kwargs['timestamp']
    savedir = kwargs['savedir']
    thresn = kwargs['threshold_neigborhood']
    n_pcs= kwargs['n_pcs']
    respar = kwargs['resolution_parameter']
    if colordict is None:
        if len(class_labels)>8:
            if (len(class_labels)-8) % 2 == 0:
                added_colors = sns.color_palette('BrBG',len(class_labels)-8)
            else:
                all_colors = sns.color_palette('BrBG',len(class_labels)-7)
                mid = round((len(all_colors)-1)/2)
                added_colors = all_colors[:mid]+all_colors[mid+1:]
            values = sns.color_palette('Paired', 10)[:6]+sns.color_palette('Paired',12)[8:10]+added_colors
        else:
            values = sns.color_palette('husl',len(class_labels))
        #newlut = dict(zip(class_numbers,values))
        newlut = dict(zip(class_labels,values))
    else:
        newlut = colordict

    newcolor = tsnedf['new_membership'].map(newlut)
    f,ax = plt.subplots(figsize=(12,10))
    tsnedf.plot.scatter('uDim1','uDim2',s=weights*100
                        ,alpha=.5,color=newcolor,ax=ax)
    for x,y in newlut.items():
        if x in class_labels:
            plt.bar(0,0,color=y,label=x,alpha=1)
    handles, labels = ax.get_legend_handles_labels()
        #plt.legend(handles[:],labels[:],bbox_to_anchor=(-0.0, 1.08, 1., .102), loc=2,
    plt.legend(handles,class_labels,bbox_to_anchor=(1, .9, .43, .102), loc=2,
                   ncol=1, mode="expand",fontsize=15)
    plt.title('Atlas-based annotation')
    plt.xlim(-40,60)
    plt.ylim(-60,30)
    plt.gcf().subplots_adjust(left=.1,right=0.75)
    f.savefig(savedir+date+'/sA_tSNE_'+date+'_nPCs='+str(n_pcs)+'_thresNeigh='+str(thresn)+'_respar='+str(respar)+'.png')
    f.savefig(savedir+date+'/sA_tSNE_'+date+'_nPCs='+str(n_pcs)+'_thresNeigh='+str(thresn)+'_respar='+str(respar)+'.pdf')


    atlas_ct = [f for f in class_labels if 'newClass' not in f]
    num_AtlasCells = len(atlas_ct)
    keys = new_metadata['Tumor'].unique()
    newlut = dict(zip(keys,sns.color_palette('deep', len(keys)-2)+sns.color_palette('husl',2)))
    newcolor = new_metadata['Tumor'].map(newlut)
    f,ax = plt.subplots(figsize=(12,10))
    tsnedf[num_AtlasCells:].plot.scatter('uDim1','uDim2',s=60
                        ,alpha=.5,color=newcolor.loc[tsnedf[num_AtlasCells:].index],ax=ax)
    for x,y in newlut.items():
        plt.bar(0,0,color=y,label=x,alpha=1)
        handles, labels = ax.get_legend_handles_labels()
        #plt.legend(handles[:],labels[:],bbox_to_anchor=(-0.0, 1.08, 1., .102), loc=2,
    plt.legend(handles,labels,bbox_to_anchor=(1, .9, .43, .102), loc=2,
                   ncol=1, mode="expand",fontsize=15)
    plt.title('Cell origin')
    plt.gcf().subplots_adjust(left=.1,right=0.75)
    f.savefig(savedir+date+'/TumorPatients_newtSNE.pdf')
    f.savefig(savedir+date+'/TumorPatients_newtSNE.png')

def plot_tsnes_subsample(tsnedf,class_numbers,class_labels,weights,new_metadata,colordict=None,**kwargs):

    date = kwargs['timestamp']
    savedir = kwargs['savedir']
    thresn = kwargs['threshold_neigborhood']
    n_pcs= kwargs['n_pcs']
    respar = kwargs['resolution_parameter']
    if colordict is None:
        if len(class_labels)>8:
            if (len(class_labels)-8) % 2 == 0:
                added_colors = sns.color_palette('BrBG',len(class_labels)-8)
            else:
                all_colors = sns.color_palette('BrBG',len(class_labels)-7)
                mid = round((len(all_colors)-1)/2)
                added_colors = all_colors[:mid]+all_colors[mid+1:]
            values = sns.color_palette('Paired', 10)[:6]+sns.color_palette('Paired',12)[8:10]+added_colors
        else:
            values = sns.color_palette('deep',len(class_labels))
        newlut = dict(zip(class_numbers,values))
    else:
        newlut = colordict


    newcolor = tsnedf['new_membership'].map(newlut)
    "---------------------------------------------------------------------------------------------------------"
    """ SHOW SUBSAMPLED CELLS"""
    "---------------------------------------------------------------------------------------------------------"
    newcolor_atlas = tsnedf['original_membership'].map(newlut).fillna('gray')
    atlas_cells = tsnedf[~tsnedf.index.isin(new_metadata.index)].index

    f,ax = plt.subplots(figsize=(12,10))
    tsnedf.loc[new_metadata.index].plot.scatter('uDim1','uDim2',s=80
                        ,alpha=.5,color=newcolor_atlas.loc[new_metadata.index],ax=ax)
    tsnedf.loc[atlas_cells].plot.scatter('uDim1','uDim2',s=80
                        ,alpha=.5,color=newcolor_atlas.loc[atlas_cells],ax=ax,marker='+')
    for x,y in newlut.items():
        plt.bar(0,0,color=y,label=x,alpha=1)
        handles, labels = ax.get_legend_handles_labels()
        #plt.legend(handles[:],labels[:],bbox_to_anchor=(-0.0, 1.08, 1., .102), loc=2,
    plt.legend(handles,class_labels,bbox_to_anchor=(1, .9, .43, .102), loc=2,
                   ncol=1, mode="expand",fontsize=15)
    plt.title('Atlas-based annotation, subsampling')
    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.gcf().subplots_adjust(left=.1,right=0.75)
    f.savefig(savedir+date+'/sA_subsamp_tSNE_'+date+'_nPCs='+str(n_pcs)+'_thresNeigh='+str(thresn)+'_respar='+str(respar)+'.png')
    f.savefig(savedir+date+'/sA_subsamp_tSNE_'+date+'_nPCs='+str(n_pcs)+'_thresNeigh='+str(thresn)+'_respar='+str(respar)+'.pdf')


    "---------------------------------------------------------------------------------------------------------"
    """ SHOW NEW CLASSES"""
    "---------------------------------------------------------------------------------------------------------"

    f,ax = plt.subplots(figsize=(12,10))
    tsnedf.loc[new_metadata.index].plot.scatter('uDim1','uDim2',s=100
                        ,alpha=.7,color=newcolor.loc[new_metadata.index],ax=ax)
    tsnedf.loc[atlas_cells].plot.scatter('uDim1','uDim2',s=260
                        ,alpha=.7,color=newcolor.loc[atlas_cells],ax=ax,marker='*',linewidths=1)
    for x,y in newlut.items():
        if x in tsnedf['new_membership'].unique():
            plt.bar(0,0,color=y,label=x,alpha=1)
    handles, labels = ax.get_legend_handles_labels()
        #plt.legend(handles[:],labels[:],bbox_to_anchor=(-0.0, 1.08, 1., .102), loc=2,
    plt.legend(handles,class_labels,bbox_to_anchor=(1, .9, .43, .102), loc=2,
                   ncol=1, mode="expand",fontsize=15)
    plt.title('Atlas-based annotation')
    plt.xlim(-30,30)
    plt.ylim(-40,40)
    plt.gcf().subplots_adjust(left=.1,right=0.75)
    f.savefig(savedir+date+'/sA_subsamp2_tSNE_'+date+'_nPCs='+str(n_pcs)+'_thresNeigh='+str(thresn)+'_respar='+str(respar)+'.png')
    f.savefig(savedir+date+'/sA_subsamp2_tSNE_'+date+'_nPCs='+str(n_pcs)+'_thresNeigh='+str(thresn)+'_respar='+str(respar)+'.pdf')

    "---------------------------------------------------------------------------------------------------------"
    """ SHOW CELL ORIGIN (or other metadata)"""
    "---------------------------------------------------------------------------------------------------------"

    num_AtlasCells = len(tsnedf['original_membership'].dropna())
    keys = new_metadata['Tumor'].unique()
    newlut = dict(zip(keys,sns.color_palette('Dark2', len(keys))))
    newcolor = new_metadata['Tumor'].map(newlut)
    f,ax = plt.subplots(figsize=(12,10))
    tsnedf[num_AtlasCells:].plot.scatter('uDim1','uDim2',s=80
                        ,alpha=.5,color=newcolor.loc[tsnedf[num_AtlasCells:].index],ax=ax)
    for x,y in newlut.items():
        plt.bar(0,0,color=y,label=x,alpha=1)
        handles, labels = ax.get_legend_handles_labels()
        #plt.legend(handles[:],labels[:],bbox_to_anchor=(-0.0, 1.08, 1., .102), loc=2,
    plt.legend(handles,labels,bbox_to_anchor=(1, .9, .43, .102), loc=2,
                   ncol=1, mode="expand",fontsize=15)
    plt.title('Cell origin')
    plt.xlim(-40,40)
    plt.ylim(-40,40)
    plt.gcf().subplots_adjust(left=.1,right=0.75)
    f.savefig(savedir+date+'/TumorPatients_newtSNE_subsample.pdf')
    f.savefig(savedir+date+'/TumorPatients_newtSNE_subsample.png')
