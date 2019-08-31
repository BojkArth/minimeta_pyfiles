#!/usr/bin/env python

import pandas as pd
import numpy as np
import loompy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import matplotlib as mpl


"""
Plotting for Brain Atlas
"""
def plot_tsnes(tsnedf,class_numbers,class_labels,weights,new_metadata,colordict=None,**kwargs):

    date = kwargs['timestamp']
    savedir = kwargs['savedir']
    savename = kwargs['savename']

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
    tsnedf.plot.scatter('uDim1','uDim2',s=weights*50
                        ,alpha=.5,color=newcolor,ax=ax)
    for x,y in newlut.items():
        if x in class_labels:
            plt.bar(0,0,color=y,label=x,alpha=1)
    handles, labels = ax.get_legend_handles_labels()

    plt.legend(handles,class_labels,bbox_to_anchor=(1, .9, .43, .102), loc=2,
                   ncol=1, mode="expand",fontsize=15)
    plt.title('Atlas-based annotation')
    #plt.xlim(-40,60)
    #plt.ylim(-60,30)
    plt.gcf().subplots_adjust(left=.1,right=0.75)
    f.savefig(savedir+date+'/'+savename+'.png')
    f.savefig(savedir+date+'/'+savename+'.pdf')













"""
PLOTTING FOR PANCREAS DATA
"""

def plot_tsnes2(tsnedf,class_numbers,class_labels,weights,new_metadata,colordict=None,**kwargs):

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
    #plt.xlim(-40,60)
    #plt.ylim(-60,30)
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
