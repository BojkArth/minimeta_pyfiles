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

def prune_bin(maindf,fastadir,**kwargs):
    # set variables
    df = pd.read_pickle(maindf)
    miniorbulk = maindf[-4:]
    """ **kwargs: coverage_mean, coverage_spread, log, bin_number, subbin_num"""
    covmu = kwargs['coverage_mean']
    covspread = kwargs['coverage_spread']
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

    if log=='yes':
        df_in_sel = df_in[(df_in['FPK_log']<=covmu+covspread/2) & (df_in['FPK_log']>=covmu-covspread/2)]
    else:
        df_in_sel = df_in[(df_in['FPK']<=covmu+covspread/2) & (df_in['FPK']>=covmu-covspread/2)]

    # further selection on GC content if applicable
    if 'gc_mean' in kwargs:
        gcmean = kwargs['gc_mean']
        gcspr = kwargs['gc_spread']
        df_in_sel = df_in_sel[(df_in_sel['GC Content']>=gcmean-gcspr/2) & (df_in_sel['GC Content']<=gcmean+gcspr/2)]
    # further selection in tSNE space if applicable
    if 'tsne' in kwargs:
        tsne = kwargs['tsne']
        x_mean = tsne[0];x_spread = tsne[1]
        y_mean = tsne[2];y_spread = tsne[3]
        df_in_sel = df_in_sel[(df_in_sel['x_60_a']>=x_mean-x_spread/2) & (df_in_sel['x_60_a']<=x_mean+x_spread/2) & (df_in_sel['y_60_a']>=y_mean-y_spread/2) & (df_in_sel['y_60_a']<=y_mean+y_spread/2)]


    df_in_nonsel = df_in[~df_in.index.isin(df_in_sel.index)]
    siz = int(df_in_sel['Sequence Length'].sum())
    contig = int(len(df_in_sel))

    # plot # (with selection constraints as text on fig)
    f = plt.figure()
    gs = gridspec.GridSpec(2,3)
    ax1 = f.add_subplot(gs[0,0])
    xcol = 'GC Content';ycol = 'FPK'
    if len(df_in_nonsel)>0:df_in_nonsel.plot.scatter(xcol,ycol,s=df_in_nonsel['Sequence Length'].divide(1e2),alpha=.3,ax=ax1,c=[.5,.5,.5])
    if len(df_in_sel)>0:df_in_sel.plot.scatter(xcol,ycol,s=df_in_sel['Sequence Length'].divide(1e2),alpha=.3,ax=ax1,c=color)
    plt.suptitle(expt_name+'_perp60_bin_'+binnum+'sub_bin_'+subnum+'_'+miniorbulk)
    plt.xlim(.2,.8)
    plt.ylabel('Coverage (FPK)');plt.xlabel('GC content')

    ax2 = f.add_subplot(gs[0,1])
    df_out.plot.scatter('x_60_a','y_60_a',alpha=.2,c=[.5,.5,.5],ax=ax2)
    if len(df_in_nonsel)>0:df_in_nonsel.plot.scatter('x_60_a','y_60_a',s=df_in_nonsel['Sequence Length'].divide(3e2),alpha=.2,ax=ax2,c='r')
    if len(df_in_sel)>0:df_in_sel.plot.scatter('x_60_a','y_60_a',s=df_in_sel['Sequence Length'].divide(3e2),alpha=.2,ax=ax2,c=color)
    plt.text(50,0,'coverage mean: '+str(covmu)+'\nspread: '
                +str(covspread)+'\nlog: '+log+'\n'+str(contig)+' contigs\n'+str(siz)+' bp')
    if 'gc_mean' in kwargs:
        plt.text(50,-30,'GC mean: '+str(gcmean)+'\nspread: '+str(gcspr))
    if 'tsne' in kwargs:
        plt.text(50,-40,'tSNE_x: '+str(x_mean)+', spread: '+str(x_spread)
                +'\ntSNE_y: '+str(y_mean)+', spread: '+str(y_spread))
    ax3 = f.add_subplot(gs[1,0])
    if len(df_in_nonsel)>0:df_in_nonsel.plot.scatter(xcol,'FPK_log',s=df_in_nonsel['Sequence Length'].divide(1e2),alpha=.3,ax=ax3,c=[.5,.5,.5])
    if len(df_in_sel)>0:df_in_sel.plot.scatter(xcol,'FPK_log',s=df_in_sel['Sequence Length'].divide(1e2),alpha=.3,ax=ax3,c=color)

    if len(df_in_sel)==0:
        plt.legend(['out'])
    elif len(df_in_nonsel)==0:
        plt.legend([miniorbulk])
    else: plt.legend(['out',miniorbulk])

    #plt.xlim(.2,.8)
    plt.ylabel('Coverage (log FPK)');plt.xlabel('GC content')

    ax4 = f.add_subplot(gs[1,1])
    if len(df_in_nonsel)>0:df_in_nonsel.plot.scatter('x_60_a','y_60_a',s=df_in_nonsel['Sequence Length'].divide(1e2),alpha=.3,ax=ax4,c=[.5,.5,.5])
    if len(df_in_sel)>0:df_in_sel.plot.scatter('x_60_a','y_60_a',s=df_in_sel['Sequence Length'].divide(1e2),alpha=.3,ax=ax4,c=color)

    f.set_figheight(10)
    f.set_figwidth(15)
    f.savefig(fastadir+'subfasta/plots/'+filename+'.png')
    plt.close(f)
    # write fasta
    input_file = fastadir+fastaname
    output_file = fastadir+'subfasta/'+filename+'.fasta'
    with open(output_file, 'w') as out_file:
        for s in HTSeq.FastaReader( input_file ):
            if df_in_sel.index.str.contains(s.name).any():
                s.write_to_fasta_file(out_file)
        out_file.close()

    # write kwargs (for further reference or reproducibility)
    with open(fastadir+'subfasta/json/'+filename+'.txt', 'w') as file:
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





