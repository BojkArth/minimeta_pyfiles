#!/usr/bin/env python

#import matplotlib.pyplot as plt
#import seaborn as sns
#from matplotlib import gridspec
#import ast
#import sys
#import hdbscan
#import time
#import scipy.io as sio
#from os import listdir
#from os.path import isfile, join
#from random import shuffle
#import json
#from collections import Counter
import pandas as pd
import numpy as np
import os
import HTSeq
from datetime import datetime

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_contig_GC(dirname,fasta,temp_df):
    # this is a function called in 'make_contig_stats_df'
    # does not work if called separately
    #bin_num = fasta.split('.')[1]
    temp_df['GC'] = '';temp_df['length_from_fasta']=''
    for seq in HTSeq.FastaReader(dirname+fasta):
        seqstring = seq.seq.decode('utf-8')
        #a = Counter(seqstring)
        #GC = (a['G']+a['C'])/len(seqstring) # this turned out to be slower than this:
        GC = (seqstring.count('G')+seqstring.count('C'))/len(seqstring)
        temp_df.loc[seq.name,'GC'] = GC
        temp_df.loc[seq.name,'length_from_fasta'] = len(seqstring)
        number = is_number(GC)
        if number==True:
            print('GC of contig '+seq.name+'is '+str(GC)+', while length='+str(len(seqstring)))
    return(temp_df)


def make_contig_stats_df(path):
    # collects everything except GC-content for each reassembled contig
    #path = 'Permafrost/FranklinBluffs/bins/fasta/reassembly/'
    #path contains all reassembly files (.txt +fasta)

    start_time = datetime.now()
    files = [f for f in os.listdir(path) if 'shotgunReads_realignmentDepth' in f]
    filesca = [f for f in os.listdir(path) if 'shotgunReads_absolute' in f]
    filescn = [f for f in os.listdir(path) if 'shotgunReads_normal' in f]
    fasta_files = [f for f in os.listdir(path) if 'fasta' in f] # reads the fasta file (make sure there is only one type of fasta file in this directory (contigs or scaffolds)
    bins = [f.split('.')[1] for f in np.sort(files)]
    totdf = None
    for bin_num in bins[2:4]:
        names = [f for f in files+filesca+filescn+fasta_files if '.'+bin_num+'.' in f]
        bin_df = pd.read_table(path+names[0])
        cols = bin_df.columns
        d1 = cols[-2].split('_')[1]
        d2 = cols[-1].split('_')[1]
        columns = ['Bin','length',d1+'_mean',d1+'_std',d2+'_mean',d2+'_std',d1+'_median',d2+'_median','absCov_'+d1,'absCov_'+d2,'norCov_'+d1,'norCov_'+d2]
        statsdf = pd.DataFrame(index=bin_df.ContigName.unique(),columns=columns)
        #print(bin_num)
        statsdf['Bin'] = bin_num
        statsdf['length'] = bin_df.groupby('ContigName').max()['Position']
        statsdf['length_linecount'] = bin_df.groupby('ContigName').count()['Position']
        statsdf[d1+'_mean'] = bin_df.groupby('ContigName').mean()[cols[-2]]
        statsdf[d2+'_mean'] = bin_df.groupby('ContigName').mean()[cols[-1]]
        statsdf[d1+'_std'] = bin_df.groupby('ContigName').std()[cols[-2]]
        statsdf[d2+'_std'] = bin_df.groupby('ContigName').std()[cols[-1]]
        statsdf[d1+'_median'] = bin_df.groupby('ContigName').median()[cols[-2]]
        statsdf[d2+'_median'] = bin_df.groupby('ContigName').median()[cols[-1]]
        # incorporate additional info here (absolute/normalized alignment file)
        statsdf['absCov_'+d1] = pd.read_table(path+names[1]).set_index('Unnamed: 0')[cols[-2]]
        statsdf['absCov_'+d2] = pd.read_table(path+names[1]).set_index('Unnamed: 0')[cols[-1]]
        statsdf['norCov_'+d1] = pd.read_table(path+names[2]).set_index('Unnamed: 0')[cols[-2]]
        statsdf['norCov_'+d2] = pd.read_table(path+names[2]).set_index('Unnamed: 0')[cols[-1]]
        #classify based on highest mean, median, or readcount (I do this here instead of out of the loop, since there is the chance of redundancy in the index for totdf)
        statsdf['class_mean'] = [d1 if statsdf.loc[f,d1+'_mean']>statsdf.loc[f,d2+'_mean'] else d2 if statsdf.loc[f,d1+'_mean']<statsdf.loc[f,d2+'_mean'] else 'equal' for f in statsdf.index]
        statsdf['class_median'] = [d1 if statsdf.loc[f,d1+'_median']>statsdf.loc[f,d2+'_median'] else d2 if statsdf.loc[f,d1+'_median']<statsdf.loc[f,d2+'_median'] else 'equal' for f in statsdf.index]
        statsdf['class_count'] = [d1 if statsdf.loc[f,'absCov_'+d1]>statsdf.loc[f,'absCov_'+d2] else d2 if statsdf.loc[f,'absCov_'+d1]<statsdf.loc[f,'absCov_'+d2] else 'equal' for f in statsdf.index]
        # get GC-content from fasta
        #fasta_time = datetime.now()
        #print('Started accessing fasta of bin '+bin_num+' for GC-content')
        statsdf = get_contig_GC(path,names[3],statsdf)
        #elapsed_fasta_time = datetime.now() - fasta_time
        #print('Added GC-content from fasta, time elapsed (hh:mm:ss.ms) {}'.format(elapsed_fasta_time))
        # make final dataframe in first loop
        if totdf is None:
            totdf = statsdf.copy()
        else:
            totdf = totdf.append(statsdf)

    expt_name = path.split('/')[1] #this works for Permafrost data, see top.
    totdf['expt_name'] = expt_name
    totdf['new_index'] = totdf['expt_name']+'_bin_'+totdf['Bin']+'_'+totdf.index
    totdf['depthfrac'+d1] = totdf['absCov_'+d1].divide(totdf['absCov_'+d1]+totdf['absCov_'+d2])
    # this will allow me to compare bin-associated depth fractions before and after reassembly
    # use fractions obtained here with checkm info on N50 and length
    totdf.to_pickle(path+expt_name+'_reassembly_contig_stats.pickle')
    time_elapsed = datetime.now() - start_time
    print('Added read count and saved, time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    print(str(len(totdf[totdf['length']>=5e3]))+' contigs above 5kb for reassembled '+expt_name)
    return(totdf)
    
def get_contig_GC2(dirname,fasta,temp_df):
    # this is a function called in 'make_contig_stats_df'
    # does not work if called separately
    bin_num = fasta.split('.')[1]
    temp_df['GC'] = -1;temp_df['length_from_fasta']=-1
    for seq in HTSeq.FastaReader(dirname+fasta):
        seqstring = seq.seq.decode('utf-8')
        #a = Counter(seqstring)
        #GC = (a['G']+a['C'])/len(seqstring) # this turned out to be slower than this:
        GC = (seqstring.count('G')+seqstring.count('C'))/len(seqstring)
        indx = bin_num+'_NODE_'+"{0:0=3d}".format(int(seq.name.split('_')[1]))
        temp_df.loc[indx,'GC'] = GC
        temp_df.loc[indx,'length_from_fasta'] = len(seqstring)
        #number = is_number(GC)
        #if number==True:
        #    print('GC of contig '+seq.name+'is '+str(GC)+', while length='+str(len(seqstring)))
    return(temp_df)
    
def make_CS_df_new_index(path):
    """
    This function changes the index to not be the contig name (as above in make_contig_stats_df), but only takes the first part
    the node number. This because for some reason the node name from the fasta does not correspond
    to the node names from the corresponding readDepth and readCount files. Sometimes the sequence length
    differs by quite a bit, and I currently do not know where this comes from. To be continued, but for 
    I just assume the coverage stats to correctly correspond to the fasta sequence stats.
    """
    # collects everything except GC-content for each reassembled contig
    #path = 'Permafrost/FranklinBluffs/bins/fasta/reassembly/'
    #path contains all reassembly files (.txt +fasta)

    start_time = datetime.now()
    files = [f for f in os.listdir(path) if 'shotgunReads_realignmentDepth' in f]
    filesca = [f for f in os.listdir(path) if 'shotgunReads_absolute' in f]
    filescn = [f for f in os.listdir(path) if 'shotgunReads_normal' in f]
    fasta_files = [f for f in os.listdir(path) if 'fasta' in f] # reads the fasta file (make sure there is only one type of fasta file in this directory (contigs or scaffolds)
    bins = [f.split('.')[1] for f in np.sort(files)]
    totdf = None
    for bin_num in bins:
        names = [f for f in files+filesca+filescn+fasta_files if '.'+bin_num+'.' in f]
        bin_df = pd.read_table(path+names[0])
        cols = bin_df.columns
        
        bin_df['idxName'] = [bin_num+'_NODE_'+"{0:0=3d}".format(int(f.split('_')[1])) for f in bin_df.ContigName]
        node_num = ["{0:0=3d}".format(int(f.split('_')[1])) for f in bin_df.ContigName.unique()] # extract num and make 3-digit string
        idx = [bin_num+'_NODE_'+f for f in node_num]
        d1 = cols[-2].split('_')[1]
        d2 = cols[-1].split('_')[1]
        columns = ['Bin','length',d1+'_mean',d1+'_std',d2+'_mean',d2+'_std',d1+'_median',d2+'_median','absCov_'+d1,'absCov_'+d2,'norCov_'+d1,'norCov_'+d2]
        statsdf = pd.DataFrame(index=idx,columns=columns)
        #print(bin_num)
        
        statsdf['Bin'] = bin_num
        statsdf['length'] = bin_df.groupby('idxName').max()['Position']
        statsdf['length_linecount'] = bin_df.groupby('idxName').count()['Position']
        statsdf[d1+'_mean'] = bin_df.groupby('idxName').mean()[cols[-2]]
        statsdf[d2+'_mean'] = bin_df.groupby('idxName').mean()[cols[-1]]
        statsdf[d1+'_std'] = bin_df.groupby('idxName').std()[cols[-2]]
        statsdf[d2+'_std'] = bin_df.groupby('idxName').std()[cols[-1]]
        statsdf[d1+'_median'] = bin_df.groupby('idxName').median()[cols[-2]]
        statsdf[d2+'_median'] = bin_df.groupby('idxName').median()[cols[-1]]
        # incorporate additional info here (absolute/normalized alignment file)
        dfa = pd.read_table(path+names[1])
        dfn = pd.read_table(path+names[2])
        dfa['idx'] = [bin_num+'_NODE_'+"{0:0=3d}".format(int(f.split('_')[1])) for f in dfa['Unnamed: 0']]
        dfa.set_index('idx',inplace=True)
        dfn.index = dfa.index
        statsdf['absCov_'+d1] = dfa[cols[-2]]
        statsdf['absCov_'+d2] = dfa[cols[-1]]
        statsdf['norCov_'+d1] = dfn[cols[-2]]
        statsdf['norCov_'+d2] = dfn[cols[-1]]
        #classify based on highest mean, median, or readcount (I do this here instead of out of the loop, since there is the chance of redundancy in the index for totdf)
        statsdf['class_mean'] = [d1 if statsdf.loc[f,d1+'_mean']>statsdf.loc[f,d2+'_mean'] else d2 if statsdf.loc[f,d1+'_mean']<statsdf.loc[f,d2+'_mean'] else 'equal' for f in statsdf.index]
        statsdf['class_median'] = [d1 if statsdf.loc[f,d1+'_median']>statsdf.loc[f,d2+'_median'] else d2 if statsdf.loc[f,d1+'_median']<statsdf.loc[f,d2+'_median'] else 'equal' for f in statsdf.index]
        statsdf['class_count'] = [d1 if statsdf.loc[f,'absCov_'+d1]>statsdf.loc[f,'absCov_'+d2] else d2 if statsdf.loc[f,'absCov_'+d1]<statsdf.loc[f,'absCov_'+d2] else 'equal' for f in statsdf.index]
        # get GC-content from fasta
        #fasta_time = datetime.now()
        #print('Started accessing fasta of bin '+bin_num+' for GC-content')
        statsdf = get_contig_GC2(path,names[3],statsdf)
        statsdf['length_diff(abs)'] = statsdf['length_from_fasta']-statsdf['length_linecount']
        statsdf['length_diff(%)'] = statsdf['length_diff(abs)']/statsdf['length_from_fasta']*100 #% that 'length_from_fasta' is longer or shorter 
        statsdf['length_diff(%)'] = [statsdf.loc[f,'length_diff(%)'] if statsdf.loc[f,'length_from_fasta']!=-1 else 'no fasta' for f in statsdf.index]

        #elapsed_fasta_time = datetime.now() - fasta_time
        #print('Added GC-content from fasta, time elapsed (hh:mm:ss.ms) {}'.format(elapsed_fasta_time))
        # make final dataframe in first loop
        if totdf is None:
            totdf = statsdf.copy()
        else:
            totdf = totdf.append(statsdf)

    expt_name = path.split('/')[1] #this works for Permafrost data, see top.
    totdf['expt_name'] = expt_name
    totdf['new_index'] = totdf['expt_name']+'_bin_'+totdf['Bin']+'_'+totdf.index
    totdf['depthfrac'+d1] = totdf['absCov_'+d1].divide(totdf['absCov_'+d1]+totdf['absCov_'+d2])
    # this will allow me to compare bin-associated depth fractions before and after reassembly
    # use fractions obtained here with checkm info on N50 and length
    totdf.to_pickle(path+expt_name+'_reassembly_contig_stats.pickle')
    time_elapsed = datetime.now() - start_time
    print('Added read count and saved, time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    print(str(len(totdf[totdf['length']>=5e3]))+' contigs above 5kb for reassembled '+expt_name)
    return(totdf)

