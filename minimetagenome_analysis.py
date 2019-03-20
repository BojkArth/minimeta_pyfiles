#!/user/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import ast
import sys
import hdbscan
import time
import scipy.io as sio
from os import listdir
from os.path import isfile, join
import os
from random import shuffle
import json
import HTSeq
from collections import Counter
from sys import platform

#my own modules
import fromasciitodf as fa

# this module should contain all the functions to make a main dataframe from an IMG scaffold cart,
# tSNE coordinates, minimeta well and bulk coverage (assembly stats)
# plot various overview tSNE features such as coloring contigs in tSNE space by phylogeny, gc-content, coverage, mini/bulk origin, sampling depth, etc
# in addition, this should be able to do hdbscan-assisted clustering of the tSNE, storing that in a dataframe and making the fastas

"""
----- INPUT -----
shorthand, scaffold cart, matfile (contig names), tSNE ascii files


Workflows:
1) just make dataframe: 			make_maindf
2) make dataframe and plot tsnes:	 	make_maindf+plot_main_features
3) make chip count df:				make_alignment_report_df
4) take dataframe and subcluster using HDBSCAN:	cluster_main_tsne

mainpath = '$HOME/Data/AssemblyData/'+projectDirectory  # this is the location where scaffold carts and tSNE ascii files are located
projectDirectory = e.g. 'Permafrost'

annotation data is stored on 'datastorage' disk (and hardcoded below)
"""
def make_maindf(shorthand,mainpath):
    if shorthand == "WD": numstr = '178665';longhand = 'WestDock_IMG_3300022818';expt_name = 'WestDock';
    elif shorthand == "SM": numstr = '178664';longhand = 'SagMAT_IMG_3300022816';expt_name = 'SagMAT';
    elif shorthand == "FB": numstr = '141108';longhand = 'FranklinBluffs_IMG_3300019790';expt_name = 'FranklinBluffs';

    #create main dataframe (from IMG scaffold cart)
    data = pd.read_csv('ScaffoldCarts/'+expt_name+'/scaffold_cart_'+expt_name+'.csv','\t')
    contigor = pd.read_csv('/home/datastorage/IMG_ANNOTATION_DATA/Permafrost_annotationData_5kb/'+longhand+'/'+numstr+'.assembled.names_map','\t',header=None)
    data['Scaffold Name'] = [s[-6:] for s in data['Scaffold ID']]
    contigor['Scaffold Name'] = [s[-6:] for s in contigor[1]]
    data['contig_name'] = ''
    data = data.set_index('Scaffold Name',drop=False).set_value(contigor.set_index('Scaffold Name').index,'contig_name',contigor.set_index('Scaffold Name')[0])

    # create tSNE dataframe (load contig name from assembly .mat files for correct x,y assignment)
    ar = sio.loadmat(mainpath+'super_contig_coverage_kmer_data.'+expt_name+'.mat')['new_header']
    ardf = pd.DataFrame.from_records(ar)
    ardf2 = ardf.copy()
    ardf['nameforreal'] = [b[0] for b in ardf[0]];ardf = ardf.drop(0,axis=1)
    cols = ['x_60_a','y_60_a','x_60_n','y_60_n','x_80_a','y_80_a','x_80_n','y_80_n',
       'x_50_a','y_50_a','x_50_n','y_50_n','x_30_a','y_30_a','x_30_n','y_30_n',
       'x_40_a','y_40_a','x_40_n','y_40_n','x_70_a','y_70_a','x_70_n','y_70_n']
    tSNE_df = pd.concat([ardf,pd.DataFrame(columns=cols)])

    # load, append & split ascii files here
    path = mainpath+'tSNE_xy/' #/home/bojk/Data/AssemblyData/Permafrost/
    fil = 'kmer5Occurrence_bhtsne_locations_'
    num2 = list(range(30,90,10)) # perplexities
    k=0
    for mat in num2:
        perplexity = str(mat)
        a = list(open(path+fil+perplexity+'.'+expt_name+'.txt'))
        df = fa.fromasciitodf(a,perplexity)

        tSNE_df.loc[:,'x_'+perplexity+'_a'] = df['x_'+perplexity+'a']
        tSNE_df.loc[:,'y_'+perplexity+'_a'] = df['y_'+perplexity+'a']
        tSNE_df.loc[:,'x_'+perplexity+'_n'] = df['x_'+perplexity+'n']
        tSNE_df.loc[:,'y_'+perplexity+'_n'] = df['y_'+perplexity+'n']

    # make combined df
    data_tSNE = pd.DataFrame(index=data.set_index('contig_name',drop=False).index.values,columns=list(data.columns)+list(tSNE_df.columns))
    data_tSNE.loc[:,list(data.columns)] = data.set_index('contig_name',drop=False)
    data_tSNE.loc[:,list(tSNE_df.columns)] = tSNE_df.set_index('nameforreal',drop=False)
    data_tSNE[list(data_tSNE.columns[8:15])] = data_tSNE[list(data_tSNE.columns[8:15])].fillna('Unassigned')
    data_tSNE[cols] = data_tSNE[cols].astype(float)
    data_tSNE.to_pickle(mainpath+expt_name+'_maindf')
    return(data_tSNE)

def cluster_main_tsne(maindf,**kwargs):
    # set variables
    if len(maindf)<1000: #this is a hack, as in usually you provide a df that goes in, but now it can also be a path to pickle
        df = pd.read_pickle(maindf)
    """ hdb kwargs:'min_cluster_size','min_samples','cluster_selection_method':'leaf','alpha':1.0, allow_single_cluster=False (default)} # default cluster_selection_method = 'eom' (excess of mass)
        other kwargs: 'perplexity_na','expt_name','write_fasta','write_df','is_this_a_series'
    """
    minCS = kwargs['min_cluster_size']
    minS = kwargs['min_samples']
    CSM = kwargs['cluster_selection_method']
    ASC = kwargs['allow_single_cluster']
    perp = kwargs['perplexity_na'] # this is a string of the form '40_a' or '60_n' (so perplexity and absolute or normalized tsne)
    expt_name = kwargs['expt_name']
    write_fasta = kwargs['write_fasta'] # save the fasta Y/n (as in: do dry run clustering first, as fasta writing takes time)
    writedf = kwargs['write_df']
    series = kwargs['is_this_a_series'] # is this a single run or a minCS sweep ('yes'/'no')
    outdir = kwargs['outdir']
    
    if platform == "linux" or platform == "linux2":
        fastadir = '/home/datastorage/ASSEMBLY_DATA/Permafrost'+expt_name.split('_')[0]+'/Combined_Analysis/'
        fastaname = 'super_contigs.Permafrost'+expt_name.split('_')[0]+'.fasta'
    elif platform == "darwin":
        fastadir = '/Users/bojk/Google Drive/QuakeLab/Data/Permafrost/'+expt_name.split('_')[0]+'/'
        fastaname = 'super_contigs.Permafrost'+expt_name.split('_')[0]+'.fasta'
            
    
    filename = expt_name+'_main_tSNE_perp_'+perp+'_HDBscan'
    
    if len(str(minCS))==1:
        filen_ext = '_minCS_00'+str(minCS)+'_minS_'+str(minS)+'_CSM_'+CSM
    elif len(str(minCS))==2:
        filen_ext = '_minCS_0'+str(minCS)+'_minS_'+str(minS)+'_CSM_'+CSM
    elif len(str(minCS))==3:
        filen_ext = '_minCS_'+str(minCS)+'_minS_'+str(minS)+'_CSM_'+CSM
    # make outdir if it does not exist
    #outdir = 'Permafrost/'+expt_name+'/bins/fasta/originals/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    xcol = 'x_'+perp; ycol = 'y_'+perp;
    temp = maindf[[xcol,ycol]]
    size = maindf['Sequence Length']

    labels = hdbscan.HDBSCAN(min_cluster_size=minCS,min_samples=minS,cluster_selection_method=CSM,allow_single_cluster=ASC).fit_predict(temp)

    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    maindf.loc[maindf.iloc[0:len(maindf)].index,'DBclusternum'] = labels[0:len(maindf)] #add cluster numbers to main df
    maindf['Bin'] = ["{0:0=3d}".format(f) for f in maindf['DBclusternum']] # convert into 3-digit string
    maindf[[xcol,ycol]] = maindf[[xcol,ycol]].astype(float)
    meanpos = maindf.groupby('DBclusternum').mean()[[xcol,ycol]] #mean position in tSNE space
    #stats = df_in[df_in['subclusternum']!=-1].groupby('subclusternum').sum()[['Sequence Length','Read Depth']] # sequence length and # 	contigs for
    #maindf[['Sequence Length','Read Depth']] = maindf[['Sequence Length','Read Depth']].astype(int)
    stats = maindf.groupby('DBclusternum').sum()[['Sequence Length','Read Depth']] # sequence length and # contigs for
    stats.rename(index=str,columns={'Read Depth':'# contigs'},inplace=True)
    params = pd.DataFrame.from_dict(kwargs,orient='index')#
    params.rename(index=str,columns={0:'parameter values'},inplace=True)

    # plot # (with selection constraints as text on fig)
    f = plt.figure()
    gs = gridspec.GridSpec(2,3)

    ax1 = f.add_subplot(gs[:,0:2]) #main tSNE
    maindf.plot.scatter(xcol,ycol,s=size.divide(3e2).astype(float),alpha=.05,ax=ax1,c=colors)
    plt.xlabel('tSNE 1');plt.ylabel('tSNE 2')
    #plt.text(80,20,params)
    #for txt in meanpos.index:
    #    ax1.annotate(str(txt), (meanpos.loc[txt,xcol],meanpos.loc[txt,ycol]))
    plt.suptitle(filename+filen_ext)

    ## print subcluster stats (#contigs, length)
    ##plt.text(50,0,stats)
    #print(params)


    ax2 = f.add_subplot(gs[0,2])
    maindf.groupby('DBclusternum').sum()['Sequence Length'].plot.hist(bins=np.logspace(4,9,50),ax=ax2,alpha=.9)
    plt.axvline(1e6,color='k')
    plt.xscale('log')
    plt.xlabel('Bin size (Bp)')
    plt.title('Min cluster size = '+str(minCS),fontsize=30)

    # plot the progress of the number of clusters, given parameters.
    # this requires somehow retrieving the data of the previous run(s)
    pickle_name = expt_name+'numcludf_minS_'+str(minS)+'.pickle'
    if series=='yes':
        if (minCS==10):
            # make and save new df
            numcludf = pd.DataFrame(index=[minCS], columns=['minCS','Num_of_clusters'])
            numcludf.iloc[0,0] = minCS
            numcludf.iloc[0,1] = maindf['DBclusternum'].max() + 1
            numcludf.to_pickle(pickle_name)
        else:
            # load saved df
            numcludf = pd.read_pickle(pickle_name)
            temp = pd.DataFrame(index=[minCS], columns=['minCS','Num_of_clusters'])
            temp.iloc[0,0] = minCS
            temp.iloc[0,1] = maindf['DBclusternum'].max() + 1
            numcludf = numcludf.append(temp,ignore_index=False)
            numcludf.to_pickle(pickle_name)
    if series=='yes':
        ax3 = f.add_subplot(gs[1,2])
        numcludf.plot(x='minCS',y='Num_of_clusters',ax=ax3,marker='o',legend=False) #kind='line'
        plt.ylabel('Number of clusters')
        plt.xlabel('Minimum cluster size')

        ###### logspace settings:
        plt.xlim(8,600)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(bottom=10,top=5e3)
        plt.text(40,0.8*numcludf.iloc[0,1],numcludf.loc[minCS,'Num_of_clusters'].astype(str)+' clusters',fontsize=20) #log"""
        ############################

        ###### linspace settings:
        """plt.xlim(0,500)
        plt.ylim(bottom=0)
        plt.text(200,0.8*numcludf.iloc[0,1],numcludf.loc[minCS,'Num_of_clusters'].astype(str)+' clusters',fontsize=20) #lin"""
        #############################################

    #plt.gcf().subplots_adjust(left=0.1,right=0.7)#,bottom=0.43,top=0.99)
    f.set_figheight(10)
    f.set_figwidth(15)
    f.savefig(outdir+'plots/'+filename+filen_ext+'.png')
    plt.close(f)

    # write fasta
    if write_fasta=='YES':
        input_file = fastadir+fastaname
        for bins in np.sort(maindf[maindf['DBclusternum']!=-1]['DBclusternum'].unique())[::-1]:
            temp = maindf[maindf['DBclusternum']==bins]
            bins = str(bins)
            if len(bins)==1:
                output_file = outdir+'/'+expt_name+'_bin_00'+bins+'.fasta'
            elif len(bins)==2:
                output_file = outdir+'/'+expt_name+'_bin_0'+bins+'.fasta'
            elif len(bins)==3:
                output_file = outdir+'/'+expt_name+'_bin_'+bins+'.fasta'

            with open(output_file, 'w') as out_file:
                for s in HTSeq.FastaReader( input_file ):
                    if temp.index.str.contains(s.name).any():
                        s.write_to_fasta_file(out_file)
                out_file.close()

    # write kwargs (for further reference or reproducibility)
    with open(outdir+'json/'+filename+'.txt', 'w') as file:
        file.write(json.dumps(kwargs))
    if writedf=='YES':
        maindf.to_pickle(outdir+filename+filen_ext)
    return(maindf) #subbindf



def test_hdbscan_param_space(df,**kwargs):
    minCS = list(range(10,94,4))+list(range(100,520,20)) # start with small step sizes, then expand (this should produce 40) # logspace?
    minCS = [int(i) for i in list(np.round(np.logspace(1,np.log10(500),30)))]
    for clusize in minCS:
        kwargs['min_cluster_size'] = clusize
        kwargs['write_df'] = 'nope'
        kwargs['write_fasta'] = 'nope'
        cluster_main_tsne(df,**kwargs) # vary cluster size here, pass other HDB arguments through kwargs



def make_alignment_report_df(maindir,shorthand):
    if shorthand=='FB':
        expt_name='FranklinBluffs'
    elif shorthand=='SM':
        expt_name='SagMAT'
    elif shorthand=='WD':
        expt_name='WestDock'

    # needed: demux.csv,alignment report
    ### load alignment report files and determine contig-chip origin ###
    alignment_report = pd.read_csv('/home/datastorage/ASSEMBLY_DATA/Permafrost'+expt_name+'/Combined_Analysis/super_contigs.Permafrost'+expt_name+'.alignment_report.txt','\t')
    alignment_report_nor = pd.read_csv('/home/datastorage/ASSEMBLY_DATA/Permafrost'+expt_name+'/Combined_Analysis/super_contigs.Permafrost'+expt_name+'.normalized_alignment_report.txt','\t')

    # replace column names by human interpretable names (i.e. well&chip# + sampling depth)
    demux = pd.read_csv('AssemblyData/Permafrost/'+expt_name+'_demux.csv','\t')
    chip = list(demux['Folder Name'].str[0:6].unique())
    chipname  = ['chip'+str(i+1)+'_' for i in range(0,len(chip))]
    demux['chip'] = [chipname[chip.index(s)] for s in demux['Folder Name'].str[0:6]]
    demux['well'] = [s[-3:] if (len(s)>20) else 'NC_'+s[-3:] for s in demux['Sample Name']]

    if shorthand=='WD': # WestDock only has 1 sampling depth
        demux['label'] = demux['chip']+demux['well']
    elif shorthand=='SM': # the others have 2 depths, 2 chips per sample
        demux['depth'] = [s[17:19] if (len(s)>25) else 'nc' for s in demux['Sample Name']]
        demux['label'] = demux['chip']+demux['well']+'_'+demux['depth']
        coldict = {'SegMat_25cm_Bulk':'SagMAT_25cm_Bulk','SegMat_15cm_Bulk':'SagMAT_15cm_Bulk'}
        alignment_report = alignment_report.rename(columns=coldict)
        alignment_report_nor = alignment_report_nor.rename(columns=coldict)
    elif shorthand=='FB': # the others have 2 depths, 2 chips per sample
        demux['depth'] = [s[19:21] if (len(s)>25) else 'nc' for s in demux['Sample Name']]
        demux['label'] = demux['chip']+demux['well']+'_'+demux['depth']

    coldict = dict(zip(list(demux['Folder Name']),list(demux['label'])))
    alignment_report = alignment_report.rename(columns=coldict)
    alignment_report_nor = alignment_report_nor.rename(columns=coldict)
    alignment_report.set_index('Unnamed: 0',inplace=True)
    alignment_report_nor.set_index('Unnamed: 0',inplace=True)

    alignment_report.to_pickle(maindir+expt_name+'_chipdf_bpcounts')
    alignment_report_nor.to_pickle(maindir+expt_name+'_chipdf_normalized')
    demux.to_pickle(maindir+expt_name+'_demux')
    return(alignment_report,alignment_report_nor)


def plot_main_features(df,expt_name,savepath):
    ################################
    ### start plotting the tSNEs ###
    ################################
    # color by:  phylum, GC, domain, mini/meta, sampling depth
    # do for all perplexity values
    perp = [30,40,50,60,70,80]
    normalized = ['a','n']
    #### GC ####
    for num in perp:
        for norm in normalized:
            xname = 'x_'+str(num)+'_'+norm
            yname = 'y_'+str(num)+'_'+norm
            f = plt.figure()
            f.set_figheight(15)
            f.set_figwidth(18)
            gs = gridspec.GridSpec(1,1)
            ax = f.add_subplot(gs[0,0])
            cm = plt.cm.get_cmap('RdBu_r')

            cax = plt.scatter(df[xname],df[yname],s=df['Sequence Length'].astype(int)/4e2,
                    c=df['GC Content'].astype(float),cmap=cm, alpha=.2)
            cbar = f.colorbar(cax)
            #cbar.ax.set_yticklabels(['< -1', '0', '> 1'])
            if norm == 'a':
                nor = 'absolute'
            else:
                nor = 'normalized'
            title = expt_name+'_'+str(num)+'_'+nor+'_GC'
            plt.title(title)
            f.savefig(savepath+title+'.png')
            plt.close(f)


    #### Domain ####
    vals = df['Lineage Domain'].unique();keys = sns.color_palette('Set2',n_colors=len(vals))
    lut = dict(zip(vals,keys))
    color = df['Lineage Domain'].map(lut)
    for num in perp:
        for norm in normalized:
            xname = 'x_'+str(num)+'_'+norm
            yname = 'y_'+str(num)+'_'+norm
            f = plt.figure()
            f.set_figheight(15)
            f.set_figwidth(15)
            gs = gridspec.GridSpec(1,1)
            ax = f.add_subplot(gs[0,0])
            plt.scatter(df[xname],df[yname],s=df['Sequence Length'].astype(int)/4e2,
                        c=color, alpha=.2)
            for x,y in lut.items():
                plt.bar(0,0,color=y,label=x,alpha=1)
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles[1:],labels[1:],bbox_to_anchor=(-0.02, 1.02, 1., .102), loc=2,
                   ncol=6, mode="expand",fontsize=10)
            if norm == 'a':
                nor = 'absolute'
            else:
                nor = 'normalized'
            title = expt_name+'_'+str(num)+'_'+nor+'_domain'
            plt.title(title)
            f.savefig(savepath+title+'.png')
            plt.close(f)

    #### Phylum ####
    vals = list(df['Lineage Phylum'].unique());shuffle(vals);keys = sns.color_palette('cubehelix',n_colors=len(vals))
    lut = dict(zip(vals,keys))
    color = df['Lineage Phylum'].map(lut)
    for num in perp:
        for norm in normalized:
            xname = 'x_'+str(num)+'_'+norm
            yname = 'y_'+str(num)+'_'+norm
            f = plt.figure()
            f.set_figheight(15)
            f.set_figwidth(15)
            gs = gridspec.GridSpec(1,1)
            ax = f.add_subplot(gs[0,0])
            plt.scatter(df[xname],df[yname],s=df['Sequence Length'].astype(int)/4e2,
                        c=color, alpha=.2)
            for x,y in lut.items():
                plt.bar(0,0,color=y,label=x,alpha=1)
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles[1:],labels[1:],bbox_to_anchor=(-0.02, 1.02, 1., .102), loc=2,
                   ncol=6, mode="expand",fontsize=10)
            if norm == 'a':
                nor = 'absolute'
            else:
                nor = 'normalized'
            title = expt_name+'_'+str(num)+'_'+nor+'_phylum'
            plt.title(title)
            f.savefig(savepath+title+'.png')
            plt.close(f)

    #### Phylum, only those with >200 contigs ####
    a = df['Lineage Phylum'].value_counts()
    a[a>200];incrowd = list(a[a>200].index)
    vals = incrowd;keys = sns.color_palette('cubehelix',n_colors=len(vals))
    lut = dict(zip(vals,keys))

    incrowd_df = df[df['Lineage Phylum'].isin(incrowd)]
    outcrowd_df = df[~df['Lineage Phylum'].isin(incrowd)]
    color = incrowd_df['Lineage Phylum'].map(lut)
    for num in perp:
        for norm in normalized:
            xname = 'x_'+str(num)+'_'+norm
            yname = 'y_'+str(num)+'_'+norm
            f = plt.figure()
            f.set_figheight(15)
            f.set_figwidth(15)
            gs = gridspec.GridSpec(1,1)
            ax = f.add_subplot(gs[0,0])
            plt.scatter(outcrowd_df[xname],outcrowd_df[yname],s=outcrowd_df['Sequence Length'].astype(int)/4e2,
                        c=[0.5,0.5,0.5], alpha=.5)
            plt.scatter(incrowd_df[xname],incrowd_df[yname],s=incrowd_df['Sequence Length'].astype(int)/4e2,
                        c=color, alpha=.5)

            for x,y in lut.items():
                plt.bar(0,0,color=y,label=x,alpha=1)
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles[2:],labels[2:],bbox_to_anchor=(-0.02, 1.02, 1., .102), loc=2,
                   ncol=6, mode="expand",fontsize=10)
            if norm == 'a':
                nor = 'absolute'
            else:
                nor = 'normalized'
            title = expt_name+'_'+str(num)+'_'+nor+'_phylum_top'
            plt.title(title)
            f.savefig(savepath+title+'.png')
            plt.close(f)

    #### Bulk vs. Minimeta ####

    bulk = df[df.index.str.contains('Bulk')]
    mini = df[~df.index.str.contains('Bulk')]

    for num in perp:
        for norm in normalized:
            xname = 'x_'+str(num)+'_'+norm
            yname = 'y_'+str(num)+'_'+norm
            f = plt.figure()
            f.set_figheight(15)
            f.set_figwidth(15)
            gs = gridspec.GridSpec(1,1)
            ax = f.add_subplot(gs[0,0])
            plt.scatter(bulk[xname],bulk[yname],s=bulk['Sequence Length'].astype(int)/4e2,
                        c='b', alpha=.2)
            plt.scatter(mini[xname],mini[yname],s=mini['Sequence Length'].astype(int)/4e2,
                        c='g', alpha=.2)
            plt.legend(['bulk','mini'])
            if norm == 'a':
                nor = 'absolute'
            else:
                nor = 'normalized'
            title = expt_name+'_'+str(num)+'_'+nor+'_seqType'
            plt.title(title)
            f.savefig(savepath+title+'.png')
            plt.close(f)

    #### Bulk depth ####

    depths = list(bulk.index.str[-9:].unique())
    if len(depths)>1:
        dep1 = bulk[bulk.index.str.contains(depths[0])]
        dep2 = bulk[bulk.index.str.contains(depths[1])]

        for num in perp:
            for norm in normalized:
                xname = 'x_'+str(num)+'_'+norm
                yname = 'y_'+str(num)+'_'+norm
                f = plt.figure()
                f.set_figheight(15)
                f.set_figwidth(15)
                gs = gridspec.GridSpec(1,1)
                ax = f.add_subplot(gs[0,0])
                plt.scatter(dep1[xname],dep1[yname],s=dep1['Sequence Length'].astype(int)/4e2,
                            c='b', alpha=.2)
                plt.scatter(dep2[xname],dep2[yname],s=dep2['Sequence Length'].astype(int)/4e2,
                            c='g', alpha=.2)
                plt.legend([depths[0],depths[1]])
                if norm == 'a':
                    nor = 'absolute'
                else:
                    nor = 'normalized'
                title = expt_name+'_'+str(num)+'_'+nor+'_bulkDepth'
                plt.title(title)
                f.savefig(savepath+title+'.png')
                plt.close(f)

        for num in perp:
            for norm in normalized:
                xname = 'x_'+str(num)+'_'+norm
                yname = 'y_'+str(num)+'_'+norm
                f = plt.figure()
                f.set_figheight(15)
                f.set_figwidth(15)
                gs = gridspec.GridSpec(1,1)
                ax = f.add_subplot(gs[0,0])
                plt.scatter(dep2[xname],dep2[yname],s=dep2['Sequence Length'].astype(int)/4e2,
                            c='g', alpha=.2)
                plt.scatter(dep1[xname],dep1[yname],s=dep1['Sequence Length'].astype(int)/4e2,
                            c='b', alpha=.2)
                plt.legend([depths[0],depths[1]])
                if norm == 'a':
                    nor = 'absolute'
                else:
                    nor = 'normalized'
                title = expt_name+'_'+str(num)+'_'+nor+'_bulkDepth_rev'
                plt.title(title)
                f.savefig(savepath+title+'.png')
                plt.close(f)

def plot_umapmain_features(df,expt_name,savepath):
    ################################
    ### start plotting the tSNEs ###
    ################################
    # color by:  phylum, GC, domain, mini/meta, sampling depth
    # do for all perplexity values

    #### GC ####
    xname = 'umap_x'
    yname = 'umap_y'
    f = plt.figure()
    f.set_figheight(15)
    f.set_figwidth(18)
    gs = gridspec.GridSpec(1,1)
    ax = f.add_subplot(gs[0,0])
    cm = plt.cm.get_cmap('RdBu_r')
    cax = plt.scatter(df[xname],df[yname],s=df['Sequence Length'].astype(int)/4e2,
              c=df['GC Content'].astype(float),cmap=cm, alpha=.2)
    cbar = f.colorbar(cax)
    title = expt_name+'_umap_GC'
    plt.title(title)
    f.savefig(savepath+title+'.png')
    plt.close(f)


    #### Domain ####
    vals = df['Lineage Domain'].unique();keys = sns.color_palette('Set2',n_colors=len(vals))
    lut = dict(zip(vals,keys))
    color = df['Lineage Domain'].map(lut)
    f = plt.figure()
    f.set_figheight(15)
    f.set_figwidth(15)
    gs = gridspec.GridSpec(1,1)
    ax = f.add_subplot(gs[0,0])
    plt.scatter(df[xname],df[yname],s=df['Sequence Length'].astype(int)/4e2,
                        c=color, alpha=.2)
    for x,y in lut.items():
        plt.bar(0,0,color=y,label=x,alpha=1)
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[1:],labels[1:],bbox_to_anchor=(-0.02, 1.02, 1., .102), loc=2,
                   ncol=6, mode="expand",fontsize=10)
    title = expt_name+'_umap_domain'
    plt.title(title)
    f.savefig(savepath+title+'.png')
    plt.close(f)

    #### Phylum ####
    vals = list(df['Lineage Phylum'].unique());shuffle(vals);keys = sns.color_palette('cubehelix',n_colors=len(vals))
    lut = dict(zip(vals,keys))
    color = df['Lineage Phylum'].map(lut)
    f = plt.figure()
    f.set_figheight(15)
    f.set_figwidth(15)
    gs = gridspec.GridSpec(1,1)
    ax = f.add_subplot(gs[0,0])
    plt.scatter(df[xname],df[yname],s=df['Sequence Length'].astype(int)/4e2,
                        c=color, alpha=.2)
    for x,y in lut.items():
        plt.bar(0,0,color=y,label=x,alpha=1)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[1:],labels[1:],bbox_to_anchor=(-0.02, 1.02, 1., .102), loc=2,
                   ncol=6, mode="expand",fontsize=10)
    title = expt_name+'_umap_phylum'
    plt.title(title)
    f.savefig(savepath+title+'.png')
    plt.close(f)

    #### Phylum, only those with >200 contigs ####
    a = df['Lineage Phylum'].value_counts()
    a[a>200];incrowd = list(a[a>200].index)
    vals = incrowd;keys = sns.color_palette('cubehelix',n_colors=len(vals))
    lut = dict(zip(vals,keys))

    incrowd_df = df[df['Lineage Phylum'].isin(incrowd)]
    outcrowd_df = df[~df['Lineage Phylum'].isin(incrowd)]
    color = incrowd_df['Lineage Phylum'].map(lut)

    f = plt.figure()
    f.set_figheight(15)
    f.set_figwidth(15)
    gs = gridspec.GridSpec(1,1)
    ax = f.add_subplot(gs[0,0])
    plt.scatter(outcrowd_df[xname],outcrowd_df[yname],s=outcrowd_df['Sequence Length'].astype(int)/4e2,
                        c=[0.5,0.5,0.5], alpha=.5)
    plt.scatter(incrowd_df[xname],incrowd_df[yname],s=incrowd_df['Sequence Length'].astype(int)/4e2,
                        c=color, alpha=.5)

    for x,y in lut.items():
        plt.bar(0,0,color=y,label=x,alpha=1)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[2:],labels[2:],bbox_to_anchor=(-0.02, 1.02, 1., .102), loc=2,
                   ncol=6, mode="expand",fontsize=10)
    title = expt_name+'_umap_phylum_top'
    plt.title(title)
    f.savefig(savepath+title+'.png')
    plt.close(f)

    #### Bulk vs. Minimeta ####

    bulk = df[df.index.str.contains('Bulk')]
    mini = df[~df.index.str.contains('Bulk')]

    f = plt.figure()
    f.set_figheight(15)
    f.set_figwidth(15)
    gs = gridspec.GridSpec(1,1)
    ax = f.add_subplot(gs[0,0])
    plt.scatter(bulk[xname],bulk[yname],s=bulk['Sequence Length'].astype(int)/4e2,
                        c='b', alpha=.2)
    plt.scatter(mini[xname],mini[yname],s=mini['Sequence Length'].astype(int)/4e2,
                        c='g', alpha=.2)
    plt.legend(['bulk','mini'])
    title = expt_name+'_umap_seqType'
    plt.title(title)
    f.savefig(savepath+title+'.png')
    plt.close(f)

    #### Bulk depth ####

    depths = list(bulk.index.str[-9:].unique())
    if len(depths)>1:
        dep1 = bulk[bulk.index.str.contains(depths[0])]
        dep2 = bulk[bulk.index.str.contains(depths[1])]

        f = plt.figure()
        f.set_figheight(15)
        f.set_figwidth(15)
        gs = gridspec.GridSpec(1,1)
        ax = f.add_subplot(gs[0,0])
        plt.scatter(dep1[xname],dep1[yname],s=dep1['Sequence Length'].astype(int)/4e2,
                            c='b', alpha=.2)
        plt.scatter(dep2[xname],dep2[yname],s=dep2['Sequence Length'].astype(int)/4e2,
                            c='g', alpha=.2)
        plt.legend([depths[0],depths[1]])
        title = expt_name+'_umap_bulkDepth'
        plt.title(title)
        f.savefig(savepath+title+'.png')
        plt.close(f)

        f = plt.figure()
        f.set_figheight(15)
        f.set_figwidth(15)
        gs = gridspec.GridSpec(1,1)
        ax = f.add_subplot(gs[0,0])
        plt.scatter(dep2[xname],dep2[yname],s=dep2['Sequence Length'].astype(int)/4e2,
                         c='g', alpha=.2)
        plt.scatter(dep1[xname],dep1[yname],s=dep1['Sequence Length'].astype(int)/4e2,
                            c='b', alpha=.2)
        plt.legend([depths[0],depths[1]])
        title = expt_name+'_umap_bulkDepth_rev'
        plt.title(title)
        f.savefig(savepath+title+'.png')
        plt.close(f)

    keys = df['Bin'].unique()
    palette = sns.color_palette('deep', len(df['Bin'].unique()))
    lut = dict(zip(keys,palette))
    colors = df['Bin'].map(lut)
    #### tSNE clusters ####
    xname = 'umap_x'
    yname = 'umap_y'
    f = plt.figure()
    f.set_figheight(15)
    f.set_figwidth(18)
    gs = gridspec.GridSpec(1,1)
    ax = f.add_subplot(gs[0,0])

    plt.scatter(df[xname],df[yname],s=df['Sequence Length'].astype(int)/4e2,
              c=colors, alpha=.2)

    title = expt_name+'_umap_tSNE'
    plt.title(title)
    f.savefig(savepath+title+'.png')
    plt.close(f)


def make_kmer_df(dataset,expt_name):

    if dataset=='Permafrost':
        kmerdir = 'Permafrost/'
        matfiledir = 'AssemblyData/Permafrost/'

    #get index
    temp = sio.loadmat(matfiledir+'super_contig_coverage_kmer_data.'+expt_name+'.mat')['new_header']
    temp2 = [f[0] for f in temp]
    df_index = [f[0] for f in temp2]

    #get kmertable
    kmertable = pd.read_table(kmerdir+expt_name+'_5mers.txt',header=None)
    kmertable = kmertable.fillna(0).T
    kmertable['index'] = df_index
    kmertable = kmertable.set_index('index')
    kmertable.to_pickle(matfiledir+'kmerdf.pickle')


    #get coverage
    acoverage = sio.loadmat(matfiledir+'super_contig_coverage_kmer_data.'+expt_name+'.mat')['absolute_coverage']
    ncoverage = sio.loadmat(matfiledir+'super_contig_coverage_kmer_data.'+expt_name+'.mat')['normalized_coverage']
    acovdf = pd.DataFrame(acoverage,index=df_index)
    ncovdf = pd.DataFrame(ncoverage,index=df_index)

    if expt_name=='WestDock':
        acovdf['bulk'] = acovdf.iloc[:,-1]
        ncovdf['bulk'] = ncovdf.iloc[:,-1]

        kmertable_acov = kmertable.join(acovdf['bulk'])
        kmertable_ncov = kmertable.join(ncovdf['bulk'])
        kmertable_acov.to_pickle(matfiledir+'kmerdf_acov.pickle')
        kmertable_ncov.to_pickle(matfiledir+'kmerdf_ncov.pickle')
    #add more stuff for the other permafrost datasets here.

    return(kmertable,kmertable_acov,kmertable_ncov)
