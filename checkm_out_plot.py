#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import ast
import sys
import checkm_out as co

def plot_checkm_clusters(maindf,checkmdir,checkm_picklename):
    checkm_output = co.make_checkm_output_df(checkmdir,checkm_picklename)
    for bins in checkm_output.index:
        contamination = str(np.round(checkm_output.loc[bins,'Contamination'],2))
        completeness = str(np.round(checkm_output.loc[bins,'Completeness'],2))
        heterogeneity = str(np.round(checkm_output.loc[bins,'Strain heterogeneity'],2))
        length = str(checkm_output.loc[bins,'Genome size'])
        contigs = str(checkm_output.loc[bins,'# contigs'])
        N50 = str(checkm_output.loc[bins,'N50 (contigs)'])
        expt_name = bins[0:-8]
        num = bins[-3:] #this depends on the fasta file names that go into checkm

        tempdf = maindf[maindf['Bin']==num]
        tempdf_out = maindf[maindf['Bin']!=num]
        tempdf_sel_b = tempdf[tempdf.index.str.contains('Bulk')]
        tempdf_sel_m = tempdf[~tempdf.index.str.contains('Bulk')]

        xcol = 'GC Content';ycol = 'FPK'

        f = plt.figure()
        gs = gridspec.GridSpec(2,3)
        ax1 = f.add_subplot(gs[0,0])
        if len(tempdf_sel_b)>0:tempdf_sel_b.plot.scatter(xcol,ycol,s=tempdf_sel_b['Sequence Length'].divide(1e2),alpha=.3,ax=ax1,c='b')
        if len(tempdf_sel_m)>0:tempdf_sel_m.plot.scatter(xcol,ycol,s=tempdf_sel_m['Sequence Length'].divide(1e2),alpha=.3,ax=ax1,c='g')
        plt.suptitle(expt_name+' perpl.60 bin '+num)
        plt.xlim(.2,.8)
        plt.ylabel('Coverage (FPK)');plt.xlabel('GC content')

        ax2 = f.add_subplot(gs[0,1])
        if len(tempdf_out)>0:tempdf_out.plot.scatter('x_60_a','y_60_a',s=tempdf_out['Sequence Length'].divide(3e2),alpha=.2,ax=ax2,c=[.5,.5,.5])
        if len(tempdf)>0:tempdf.plot.scatter('x_60_a','y_60_a',s=tempdf['Sequence Length'].divide(3e2),alpha=.2,ax=ax2,c='g')
        xpos = 60;ypos = 0
        plt.text(xpos,ypos+20,'Completeness: '+completeness)
        plt.text(xpos,ypos+10,'Contamination: '+contamination)
        plt.text(xpos,ypos+0,'Heterogeneity: '+heterogeneity)
        plt.text(xpos,ypos-10,'Length: '+length)
        plt.text(xpos,ypos-20,'#contigs: '+contigs)
        plt.text(xpos,ypos-30,'N50: '+N50)


        ax3 = f.add_subplot(gs[1,0])
        if len(tempdf_sel_b)>0:tempdf_sel_b.plot.scatter(xcol,'FPK_log',s=tempdf_sel_b['Sequence Length'].divide(1e2),alpha=.3,ax=ax3,c='b')
        if len(tempdf_sel_m)>0:tempdf_sel_m.plot.scatter(xcol,'FPK_log',s=tempdf_sel_m['Sequence Length'].divide(1e2),alpha=.3,ax=ax3,c='g')
        if len(tempdf_sel_m)==0:
            plt.legend(['bulk'])
        elif len(tempdf_sel_b)==0:
            plt.legend(['mini'])
        else: plt.legend(['bulk','mini'])
        plt.xlim(.2,.8)
        plt.ylabel('Coverage (log FPK)');plt.xlabel('GC content')

        ax4 = f.add_subplot(gs[1,1])
        if len(tempdf_sel_b)>0:tempdf_sel_b.plot.scatter('x_60_a','y_60_a',s=tempdf_sel_b['Sequence Length'].divide(1e2),alpha=.3,ax=ax4,c='b')
        if len(tempdf_sel_m)>0:tempdf_sel_m.plot.scatter('x_60_a','y_60_a',s=tempdf_sel_m['Sequence Length'].divide(1e2),alpha=.3,ax=ax4,c='g')

        f.set_figheight(20)
        f.set_figwidth(30)
        f.savefig(filedir+bins+'.png')

def compare_two_checkmdfs(df1,df2,hue_name,savedir,savename):
    #takes two checkm dfs and compares the 3 quality scores
    #outputs a violin
    dfcompare = df1.append(df2)
    dfviolin = pd.melt(dfcompare,value_vars=['Completeness','Contamination','Strain heterogeneity']
    			,id_vars=hue_name)
    dfviolin['value_int'] = dfviolin['value'].astype(int)

    f = plt.figure()
    sns.violinplot(x='variable',y='value_int', hue=hue_name, data=dfviolin,split=True,inner='quartile')#,bw=.15)
    plt.xlabel('');plt.ylabel('%')
    plt.ylim(top=120,bottom=-20)
    plt.legend(loc='upper left')
    f.set_figheight(7)
    f.set_figwidth(7)
    f.savefig(savedir+savename+'.png')
    f.savefig(savedir+savename+'.pdf')

def compare_two_checkmdfs_sina(df1,df2,hue_name,savedir,savename):
    import sinaplot as sin
    #takes two checkm dfs and compares the 3 quality scores
    #outputs a violin
    dfcompare = df1.append(df2)
    dfviolin = pd.melt(dfcompare,value_vars=['Completeness','Contamination','Strain heterogeneity']
    			,id_vars=hue_name)
    dfviolin['value_int'] = dfviolin['value'].astype(int)
    cat1,cat2 = dfviolin[hue_name].unique()[0],dfviolin[hue_name].unique()[1]
    f = plt.figure()
    sin.sinaplot(x='variable',y='value_int', hue=hue_name, data=dfviolin,split=True,inner='quartile',violin=True,violin_facealpha=0.250)#,bw=.15)
    plt.xlabel('');plt.ylabel('%')
    plt.xticks(fontsize=13)
    plt.ylim(top=120,bottom=-20)
    #plt.legend(loc='lower right')
    plt.legend(loc='lower center', ncol=2)
    plt.title('N '+cat1+'= '+str(len(df1))+', N '+cat2+'= '+str(len(df2)))
    f.set_figheight(7)
    f.set_figwidth(7)
    f.savefig(savedir+savename+'.png')
    f.savefig(savedir+savename+'.pdf')

if __name__=='__main__':
    checkmdir=sys.argv[1] #dir of checkm data
    checkm_picklename = sys.argv[2] # name to save pickle to
    maindf = pd.read_pickle(sys.argv[3]) # corresponding df with bin names
    plot_checkm_clusters(maindf,checkmdir,checkm_picklename);

