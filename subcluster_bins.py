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
    """if miniorbulk=='mini':
        color = 'g'
    elif miniorbulk=='bulk':
        color = 'b'"""

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



def subcluster_bin_post_reassembly(maindf,fastadir,**kwargs):
    """
    ***MAIN PRUNING FUNCTION***
    STRUCTURE WITH SUBFUNCTIONS/DEPENDENCIES:

    subcluster_bin_post_reassembly()
    	pruning_bins()
    	    subclustering()
    	    	hdbscan.HDBSCAN()
    	    plot_subclusters_twodepths()
    	    write_subfasta() [optional]

    subcluster for an entire set of reassembled bins OR only a selected bin
    one or two depths (WD vs FB & SM)
    plotting, writing fasta optional
    subclustering done on the mean coverage of every depth available (so datasets with 2 depths will produce redundant subclusters, I can later select which one
    or refine the process (e.g. by subclustering at the depth which has the highest median of the mean coverage (this measure to prevent outliers from determining the choice))
    """
    df = pd.read_pickle(maindf)
    df = df[df['length_linecount']>=5e3]
    #df = df[df['length_from_fasta']>=5e3]
    """ **hdbkwds:'min_cluster_size','min_samples','cluster_selection_method':'leaf','alpha':1.0, allow_single_cluster=False (default)} # default cluster_selection_method = 'eom' (excess of mass) """
    """minCS = kwargs['min_cluster_size']
    minS = kwargs['min_samples']
    CSM = kwargs['cluster_selection_method']
    ASC = kwargs['allow_single_cluster']"""
    expt_name = kwargs['expt_name']
    params = pd.DataFrame.from_dict(kwargs,orient='index')
    params.rename(index=str,columns={0:'parameter values'},inplace=True)

    if 'Bin' in kwargs: #apply pruning to a single bin (for bin-specific pruning)
        binnum = kwargs['Bin']
        if 'GCmin' in kwargs:
            df = df[(df.GC>=kwargs['GCmin']) & (df.GC<=kwargs['GCmax'])] # additional constraints if wanted
        pruning_bins(df, binnum,fastadir,**kwargs)
        # write kwargs (for further reference or reproducibility)
        with open(fastadir+'pruning/json/'+expt_name+'_bin_'+binnum+'_pruneparams_hdb.txt', 'w') as file:
            file.write(json.dumps(kwargs))
    else:
        for binnum in df['Bin'].unique():
            pruning_bins(df, binnum,fastadir,**kwargs)
        # write kwargs (for further reference or reproducibility)
        with open(fastadir+'pruning/json/'+expt_name+'parameters_hdbclustering.txt', 'w') as file:
            file.write(json.dumps(kwargs))

def pruning_bins(df,binnum,fastadir,**kwargs):
    fastaname = fastadir+'genome_contigs_withBulk.'+binnum+'.5000bp_filter.fasta'
    writefasta = kwargs['write_fasta']
    df_in = df[df['Bin']==binnum]
    if kwargs['GC_sensitive']=='YES': # added sensitivity to GC-space
        df_in['GC_sensitive'] = df_in['GC'].multiply(100)
        xcol = 'GC_sensitive'
    else:
        xcol = 'GC'
    ycollist = list(df.columns[df.columns.str.contains('cm_mean')])
    if len(ycollist)==2: # perform subclustering for both depths
        ycol1 = ycollist[0]
        ycol2 = ycollist[1]
        df_in,meanpos1,stats1,colors1 = subclustering(df_in,xcol,ycol1,**kwargs)
        df_in,meanpos2,stats2,colors2 = subclustering(df_in,xcol,ycol2,**kwargs)
        plot_subclusters_twodepths(df_in,ycol1,ycol2,colors1,colors2,meanpos1,meanpos2,fastadir+'pruning/')
        # write fasta, if requested
        if writefasta == 'YES':
            subclucol = meanpos1.index.name
            write_subfasta(fastaname,df_in,fastadir+'pruning/fastas/',subclucol)
            subclucol = meanpos2.index.name
            write_subfasta(fastaname,df_in,fastadir+'pruning/fastas/',subclucol)
        stats1.to_csv(fastadir+'pruning/json/stats_bin_'+binnum+'_'+ycol1+'.txt','\t')
        stats2.to_csv(fastadir+'pruning/json/stats_bin_'+binnum+'_'+ycol2+'.txt','\t')
    elif len(ycollist)==1:
        ycol = ycollist[0]
        df_in,meanpos,stats,colors = subclustering(df_in,xcol,ycol,**kwargs)
        if writefasta == 'YES':
            subclucol = meanpos.index.name
            write_subfasta(fastaname,df_in,fastadir+'pruning/fastas/',subclucol)
        stats.to_csv(fastadir+'pruning/json/stats_bin_'+binnum+'_'+ycol+'.txt','\t')

def write_subfasta(fastapath,df,savedir,subclucol):
    binnum = df.Bin.unique()[0]
    expt_name = df['expt_name'].unique()[0]
    for subbin in df[df[subclucol]!=-1][subclucol].unique():
        temp = df[df[subclucol]==subbin]
        output_file = savedir+expt_name+'_bin_'+binnum+'_'+subclucol[-4:]+'_sub_bin_'+subbin+'.fasta'
        with open(output_file, 'w') as out_file:
            for s in HTSeq.FastaReader( fastapath ):
                name = binnum+'_NODE_'+"{0:0=3d}".format(int(s.name.split('_')[1]))
                if temp.index.str.contains(name).any():
                    s.write_to_fasta_file(out_file)
            out_file.close()

def subclustering(df_in,xcol,ycol,**kwargs):
    minCS = kwargs['min_cluster_size']
    minS = kwargs['min_samples']
    CSM = kwargs['cluster_selection_method']
    ASC = kwargs['allow_single_cluster']

    temp = df_in[[xcol,ycol]]
    labels = hdbscan.HDBSCAN(min_cluster_size=minCS,min_samples=minS,cluster_selection_method=CSM,allow_single_cluster=ASC).fit_predict(temp)
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    xcol = 'GC' #for plotting in original GC space
    subclunum = 'subclusternum'+ycol[:-5]
    df_in[subclunum] = ["{0:0=3d}".format(f) if f>=0 else f for f in labels] #add cluster numbers to main df
    meanpos = df_in.groupby(subclunum).mean()[[xcol,ycol]] #mean position in GC-coverage space
    stats = df_in.groupby(subclunum).sum()['length_from_fasta'] # sequence length
    stats2 = df_in.groupby(subclunum).count()['length_from_fasta'] # contigs for
    stats2.rename('#contigs',inplace=True)
    statsfinal = pd.concat([stats,stats2],axis=1)
    return(df_in,meanpos,statsfinal,colors)

def plot_subclusters_twodepths(df,ycol1,ycol2,colors1,colors2,meanpos1,meanpos2,savedir):
    xcol = 'GC'
    size = df['length_from_fasta']
    expt_name = df['expt_name'].unique()[0]
    binnum = df['Bin'].unique()[0]

    f = plt.figure()
    gs = gridspec.GridSpec(4,2)
    ax1 = f.add_subplot(gs[0,0]) #GC-mean coverage lin depth 1
    df.plot.scatter(xcol,ycol1,s=size.divide(1e2),alpha=.3,ax=ax1,c=colors1)
    for txt in meanpos1.index:
        ax1.annotate(str(txt), (meanpos1.loc[txt,xcol],meanpos1.loc[txt,ycol1]))
    plt.xlim(.2,.8)
    plt.title(ycol1)
    plt.ylabel('Coverage')

    ax2 = f.add_subplot(gs[0,1]) #GC-mean coverage lin depth 2
    df.plot.scatter(xcol,ycol2,s=size.divide(1e2),alpha=.3,ax=ax2,c=colors2)
    for txt in meanpos2.index:
        ax2.annotate(str(txt), (meanpos2.loc[txt,xcol],meanpos2.loc[txt,ycol2]))
    plt.xlim(.2,.8)
    plt.title(ycol2)
    plt.ylabel('Coverage')


    ax3 = f.add_subplot(gs[1,0]) #GC-mean coverage log depth 1
    df.plot.scatter(xcol,ycol1,s=size.divide(1e2),alpha=.3,ax=ax3,c=colors1)
    plt.yscale('log')
    plt.xlim(.2,.8)
    plt.title(ycol1)
    plt.ylabel('Coverage')

    ax4 = f.add_subplot(gs[1,1]) #GC-mean coverage log depth 2
    df.plot.scatter(xcol,ycol2,s=size.divide(1e2),alpha=.3,ax=ax4,c=colors2)
    plt.yscale('log')
    plt.xlim(.2,.8)
    plt.title(ycol2)
    plt.ylabel('Coverage')

    ycollist = list(df.columns[df.columns.str.contains('absCov_')])
    absd1 = ycollist[0];absd2 = ycollist[1]
    ax5 = f.add_subplot(gs[2,0]) #GC-abs coverage  depth 1
    df.plot.scatter(xcol,absd1,s=size.divide(1e2),alpha=.3,ax=ax5,c=colors1)
    #plt.yscale('log')
    plt.xlim(.2,.8)
    plt.title(absd1)
    plt.ylabel('Coverage')

    ax6 = f.add_subplot(gs[2,1]) #GC-abs coverage depth 2
    df.plot.scatter(xcol,absd2,s=size.divide(1e2),alpha=.3,ax=ax6,c=colors2)
    #plt.yscale('log')
    plt.xlim(.2,.8)
    plt.title(absd2)
    plt.ylabel('Coverage')

    ycollist = list(df.columns[df.columns.str.contains('_median')])
    absd1 = ycollist[0];absd2 = ycollist[1]
    ax7 = f.add_subplot(gs[3,0]) #GC-abs coverage  depth 1
    df.plot.scatter(xcol,absd1,s=size.divide(1e2),alpha=.3,ax=ax7,c=colors1)
    #plt.yscale('log')
    plt.xlim(.2,.8)
    plt.title(absd1)
    plt.ylabel('Coverage');plt.xlabel('GC content')

    ax8 = f.add_subplot(gs[3,1]) #GC-abs coverage depth 2
    df.plot.scatter(xcol,absd2,s=size.divide(1e2),alpha=.3,ax=ax8,c=colors2)
    #plt.yscale('log')
    plt.xlim(.2,.8)
    plt.title(absd2)
    plt.ylabel('Coverage');plt.xlabel('GC content')

    plt.suptitle(expt_name+'_subclustering_bin_'+binnum+'_HDBscan')
    f.set_figheight(22)
    f.set_figwidth(15)
    f.savefig(savedir+expt_name+'_subclustering_bin_'+binnum+'.png')
    plt.close(f)



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





