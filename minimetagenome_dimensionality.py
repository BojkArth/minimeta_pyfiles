#!/usr/bin/env python

import itertools
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import HTSeq
import os
import re
#import khmer
import imageio
import glob
import hdbscan
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

from sys import platform
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

"""
Created 2019-03-13	Bojk Berghuis

Functions:
"""
def perform_PCA(kmer_df):
    x = StandardScaler().fit_transform(kmer_df)
    pca = PCA(n_components=1000)
    principalComp = pca.fit_transform(x)
    princdf = pd.DataFrame(principalComp)
    princdf.index =kmer_df.index
    return(princdf)

def make_tsne_from_df(df,contig_df,maindir,savename):
    # compute tsne
    x = StandardScaler().fit_transform(df)
    x_emb = TSNE(n_components=2,perplexity=40,random_state=23944).fit_transform(x)
    tsnedf = pd.DataFrame(x_emb,index=df.index)
    tsnedf = tsnedf.join(contig_df['Sequence length'])
    tsnedf = tsnedf.join(contig_df['GC'])
    if firstround=='YES' and complete_analysis=='YES':
        #tsnedf['genome'] = [f.split('_')[0] for f in tsnedf.index]
        tsnedf.to_pickle(maindir+'tsnedf_'+savename+'.pickle')
    if complete_analysis!='YES':
        # calculate cluster quality
        #QCdf = make_QCdf_from_tsnedf(tsnedf)
        # plot some metrics
        plot_tsnedf_metrics(tsnedf,maindir,savename)
        return tsnedf
    else:
        return tsnedf

def plot_tsnedf_metrics(tsnedf,maindir,savename):
    f,ax = plt.subplots(1,1,figsize=(10,10))
    plt.scatter(tsnedf[0],tsnedf[1],s=tsnedf['Sequence length'].astype(float)/100,
                alpha=.05,c=tsnedf['GC'],cmap='RdBu_r')
    plt.title(savename+' colored by GC')
    f.savefig(maindir+'plots/tSNE_GC_'+savename+'.png')
    plt.close(f)




def perform_complete_analysis_Coverage(combined_df,kmerlength,contigdf,maindir,savename):
    global num_dims
    num_dims = len(combined_df.T)
    kmer_length = kmerlength#int(np.log(num_dims)/np.log(4))
    start_time = time.time()

    savename = savename+'_'+str(kmer_length)+'mers'

    maindir = maindir
    global complete_analysis
    complete_analysis = 'YES'
    global firstround
    firstround = 'YES'
    """
    if kmer_length==5:
        pcs_to_reduce = [int(round(f)) for f in np.logspace(0,3,20)[3:-1]]
    elif kmer_length>=6:
        pcs_to_reduce = [int(round(f)) for f in np.logspace(0,3,20)[8:]]
    else:
        print('define range of PCs first!')
        return 0
    """

    #optimaldf = pd.DataFrame(index=pcs_to_reduce+[num_dims],columns=['idxmax','max'])

    kmerdf_norm = combined_df.divide(contigdf['Sequence length'],axis=0) # normalize kmers only or kmer and coverage/bp-mapped???
    if os.path.isfile(maindir+'tsnedf_'+savename+'.pickle'):
        print('tSNE-df previously made, loading from pickle, full path = ')
        print(maindir+'tsnedf_'+savename+'.pickle')
        tsnedf_main = pd.read_pickle(maindir+'tsnedf_'+savename+'.pickle')
        plot_tsnedf_metrics(tsnedf_main,maindir,savename+'_'+str(num_dims)+'_dimensions')
    else:
        print('building tSNE of all '+str(num_dims)+' dimensions')
        tsnedf_main = make_tsne_from_df(kmerdf_norm,contigdf,maindir,savename) ##################################################
        end = time.time()
        print('finished building main tSNE, this took {:.2f} seconds'.format(end - start_time))
        print('performing cluster sweep of tSNE of all '+str(num_dims)+' dimensions')
    """
    start = time.time()
    hdbsweep = minCS_sweep(tsnedf_main,maindir,savename,savename) ##################################################
    end = time.time()
    print('finished cluster sweep of main tSNE, this took {:.2f} seconds'.format(end - start))
    print('Total elapsed time is {:.2f} minutes'.format((end-start_time)/60))
    """
    keys = [0,1];values = ['x_'+str(num_dims)+'mers','y_'+str(num_dims)+'mers']
    newnames = dict(zip(keys,values))
    tsnedf_main.rename(index=str,columns=newnames,inplace=True)

    """
    optimaldf.loc[num_dims,'max'] = hdbsweep['val_leaf'].max()
    optimaldf.loc[num_dims,'idxmax'] = hdbsweep['val_leaf'].idxmax()

    keys = hdbsweep.columns;values = [f+str(kmer_length)+'mers' for f in keys]
    lut = dict(zip(keys,values))
    hdbsweep.rename(index=int,columns=lut,inplace=True)
    """

    firstround=='NO'
    pcs_to_reduce = [int(round(f)) for f in np.logspace(0,3,20)[8:]]

    if os.path.isfile(maindir+'PCAdf_'+savename+'.pickle'):
        print('PCA performed earlier, loading file:\n'+maindir+'PCAdf_'+savename+'.pickle')
        pcdf = pd.read_pickle(maindir+'PCAdf_'+savename+'.pickle')
        if os.path.isfile(maindir+savename.split('_')[0]+str(kmer_length)+'mers_all_tSNEs_temp')
            tsnedf_main = pd.read_pickle(maindir+savename.split('_')[0]+str(kmer_length)+'mers_all_tSNEs_temp')
            cols = tsnedf_main.columns
            pcs_done = list(set([int(f[4:]) for f in cols[4:]]))
            [pcs_to_reduce.remove(f) for f in pcs_done]
            print('Some tSNEs already calculated, resuming at dim reduction of '+str(pcs_to_reduce[0])+' PCs')

    else:
        print('Performing PCA...')
        pcdf = perform_PCA(kmerdf_norm)
        pcdf.to_pickle(maindir+'PCAdf_'+savename+'.pickle')


    for numpcs in pcs_to_reduce:
        print('building tSNE of '+str(numpcs)+' PCs')
        start = time.time()
        tsnedf_temp = make_tsne_from_df(pcdf.iloc[:,:numpcs+1],contigdf,maindir,savename) ##################################################
        end = time.time()
        print('finished building tSNE of '+str(numpcs)+' PCs, this took {:.2f} seconds'.format(end - start))

        #savetemp = savename.split('_')[0]+'_'+str(numpcs)+'PCs'
        #print('performing cluster sweep of tSNE of '+str(numpcs)+' PCs')
        """
        start = time.time()
        hdb_temp = minCS_sweep(tsnedf_temp,maindir,savetemp,savetemp) ##################################################
        end = time.time()
        print('finished cluster sweep of tSNE with'+str(numpcs)+'PCs, this took {:.2f} seconds'.format(end - start))
        print('Total elapsed time is {:.2f} minutes'.format((end-start_time)/60))
        """
        keys = [0,1];values = ['x_PC'+str(numpcs),'y_PC'+str(numpcs)]
        newnames = dict(zip(keys,values))
        tsnedf_main = tsnedf_main.join(tsnedf_temp[[0,1]])
        tsnedf_main.rename(index=str,columns=newnames,inplace=True)
        tsnedf_main.to_pickle(maindir+savename.split('_')[0]+str(kmer_length)+'mers_all_tSNEs_temp')

        """
        optimaldf.loc[numpcs,'max'] = hdb_temp['val_leaf'].max()
        optimaldf.loc[numpcs,'idxmax'] = hdb_temp['val_leaf'].idxmax()

        keys = hdb_temp.columns;values = [f+'_PC'+str(numpcs) for f in keys]
        lut = dict(zip(keys,values))
        hdb_temp.rename(index=int,columns=lut,inplace=True)
        hdbsweep = hdbsweep.join(hdb_temp)
        """

    end = time.time()
    print('Finished everything.')
    print('Total elapsed time is {:.2f} minutes'.format((end-start_time)/60))
    tsnedf_main.to_pickle(maindir+savename.split('_')[0]+str(kmer_length)+'mers_all_tSNEs')
    #optimaldf.to_pickle(maindir+savename.split('_')[0]+'_optimalValues_perPC')
    #hdbsweep.to_pickle(maindir+savename.split('_')[0]+'_allPCs_clustersweep_Quality')
    return tsnedf_main#,optimaldf,hdbsweep

