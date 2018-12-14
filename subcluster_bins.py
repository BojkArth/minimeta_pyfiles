#!/usr/bin/env python
# bin pruning: input: contigs in bin of interest; out: fasta containing pruned bin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import ast
import sys
import hdbscan
import HTSeq
import json
"""
Goal: perform unsupervised sub-clustering of a (re-assembled) genomic bin in GC-coverage space. Write the subcluster fastas if necessary.
Created: Bojk Berghuis, 17 September 2018

"""
def subcluster_bin(maindf,fastadir,**kwargs):
    # set variables
    df = pd.read_pickle(maindf)
    miniorbulk = maindf[-4:]

    """ **hdbkwds:'min_cluster_size','min_samples','cluster_selection_method':'leaf','alpha':1.0, allow_single_cluster=False (default)} # default cluster_selection_method = 'eom' (excess of mass) """
    minCS = kwargs['min_cluster_size']
    minS = kwargs['min_samples']
    CSM = kwargs['cluster_selection_method']
    ASC = kwargs['allow_single_cluster']
    binnum = kwargs['bin_number']
    subnum = kwargs['subbin_num']
    log = kwargs['log']
    expt_name = kwargs['expt_name']
    fastaname = expt_name+'_bin_'+binnum+'_'+miniorbulk+'.fasta'
    filename = expt_name+'_bin_'+binnum+'sub_bin_'+subnum+'_'+miniorbulk
    if miniorbulk=='mini':
        color = 'g'
    elif miniorbulk=='bulk':
        color = 'b'

    df_in = df[df['Bin']==binnum]
    df_out = df[df['Bin']!=binnum]
    xcol = 'GC Content'
    if log=='yes': ycol = 'FPK_log'
    else:ycol = 'FPK'

    temp = df_in[[xcol,ycol]]
    size = df_in['Sequence Length']

    labels = hdbscan.HDBSCAN(min_cluster_size=minCS,min_samples=minS,cluster_selection_method=CSM,allow_single_cluster=ASC).fit_predict(temp)

    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    df_in['subclusternum'] = labels[0:len(df)] #add cluster numbers to main df
    meanpos = df_in.groupby('subclusternum').mean()[[xcol,ycol]] #mean position in GC-coverage space
    #stats = df_in[df_in['subclusternum']!=-1].groupby('subclusternum').sum()[['Sequence Length','Read Depth']] # sequence length and # contigs for
    stats = df_in.groupby('subclusternum').sum()[['Sequence Length','Read Depth']] # sequence length and # contigs for
    stats.rename(index=str,columns={'Read Depth':'# contigs'},inplace=True)
    params = pd.DataFrame.from_dict(kwargs,orient='index')#
    params.rename(index=str,columns={0:'parameter values'},inplace=True)
    """
    df_in_nonsel = df_in[~df_in.index.isin(df_in_sel.index)]
    siz = int(df_in_sel['Sequence Length'].sum())
    contig = int(len(df_in_sel))"""

    # plot # (with selection constraints as text on fig)
    f = plt.figure()
    gs = gridspec.GridSpec(2,3)


    ax1 = f.add_subplot(gs[0,0]) #GC-coverage lin
    xcol = 'GC Content';
    df_in.plot.scatter(xcol,'FPK',s=size.divide(1e2),alpha=.3,ax=ax1,c=colors)
    plt.suptitle(expt_name+'_perp60_bin_'+binnum+'sub_bin_'+subnum+'_'+miniorbulk+'_HDBscan')
    if log!='yes':
        for txt in meanpos.index:
            ax1.annotate(str(txt), (meanpos.loc[txt,xcol],meanpos.loc[txt,ycol]))
    plt.xlim(.2,.8)
    plt.ylabel('Coverage (FPK)');plt.xlabel('GC content')


    ax2 = f.add_subplot(gs[0,1]) #full tSNE
    df_out.plot.scatter('x_60_a','y_60_a',alpha=.2,c=[.5,.5,.5],ax=ax2)
    df_in.plot.scatter('x_60_a','y_60_a',s=size.divide(3e2),alpha=.2,ax=ax2,c=colors)

    # print subcluster stats (#contigs, length)
    plt.text(50,-40,stats)
    plt.text(50,-100,params)
    """plt.text(50,0,'coverage mean: '+str(covmu)+'\nspread: '
                +str(covspread)+'\nlog: '+log+'\n'+str(contig)+' contigs\n'+str(siz)+' bp')
    if 'gc_mean' in kwargs:
        plt.text(50,-30,'GC mean: '+str(gcmean)+'\nspread: '+str(gcspr))
    if 'tsne' in kwargs:
        plt.text(50,-40,'tSNE_x: '+str(x_mean)+', spread: '+str(x_spread)
                +'\ntSNE_y: '+str(y_mean)+', spread: '+str(y_spread))"""


    ax3 = f.add_subplot(gs[1,0]) #GC-coverage log
    df_in.plot.scatter(xcol,'FPK_log',s=size.divide(1e2),alpha=.3,ax=ax3,c=colors)
    plt.ylabel('Coverage (log FPK)');plt.xlabel('GC content')
    if log=='yes':
        for txt in meanpos.index:
            print(txt)
            ax3.annotate(str(txt), (meanpos.loc[txt,xcol],meanpos.loc[txt,ycol]))

    ax4 = f.add_subplot(gs[1,1]) # tSNE, cluster region only
    df_in.plot.scatter('x_60_a','y_60_a',s=size.divide(1e2),alpha=.3,ax=ax4,c=colors)


    f.set_figheight(10)
    f.set_figwidth(15)
    f.savefig(fastadir+'subfasta/plots/'+filename+'_hdbclustering.png')
    plt.close(f)

    # write fasta
    input_file = fastadir+fastaname
    for subbins in df_in[df_in['subclusternum']!=-1]['subclusternum'].unique():
        temp = df_in[df_in['subclusternum']==subbins]
        output_file = fastadir+'subfasta/'+filename+'_subbin_'+str(subbins)+'.fasta'
        with open(output_file, 'w') as out_file:
            for s in HTSeq.FastaReader( input_file ):
                if temp.index.str.contains(s.name).any():
                    s.write_to_fasta_file(out_file)
            out_file.close()

    # write kwargs (for further reference or reproducibility)
    with open(fastadir+'subfasta/json/'+filename+'_hdbclustering.txt', 'w') as file:
        file.write(json.dumps(kwargs))
    #subbindf = append_subcluster(subbindf,coord,bins,name,log,tsnecoord,contig,siz,mini)


    #return() #subbindf

if __name__ == "__main__":
    df = pd.read_pickle(sys.argv[1])
    p = sys.argv[2]
    fasta_dir = sys.argv[3]
    params = ast.literal_eval(p)
    print(type(params))

    covmean = params[0]
    covspread = params[1]
    log = params[2]
    binnum = params[3]
    subnum = params[4]
    print(subnum)





