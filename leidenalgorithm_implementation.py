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
