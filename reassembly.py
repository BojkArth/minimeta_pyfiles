#!/usr/bin/env python

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
from datetime import datetime

def prune_bin_from_fasta(input_dir):
    # this is post-reassembly stuff
    fasta_files = [f for f in os.listdir(input_dir) if 'fasta' in f]

    for fasta in fasta_files:
        bin_num = fasta.split('.')[1]

    a = Counter(string)
    (a['G']+a['C'])/len(string)
    #do something
    return(output)


def make_contig_stats_df(path):
    # collects everything except GC-content for each reassembled contig
    #path = 'Permafrost/FranklinBluffs/bins/fasta/reassembly/'
    #path contains all reassembly files (.txt +fasta)

    start_time = datetime.now()
    files = [f for f in os.listdir(path) if 'shotgunReads_realignmentDepth' in f]
    filesca = [f for f in os.listdir(path) if 'shotgunReads_absolute' in f]
    filescn = [f for f in os.listdir(path) if 'shotgunReads_normal' in f]
    bins = [f.split('.')[1] for f in np.sort(files)]
    totdf = None
    for bin_num in bins[0:10]:
        names = [f for f in files+filesca+filescn if bin_num in f]
        bin_df = pd.read_table(path+names[0])
        cols = bin_df.columns
        d1 = cols[-2].split('_')[1]
        d2 = cols[-1].split('_')[1]
        columns = ['Bin','length',d1+'_mean',d1+'_std',d2+'_mean',d2+'_std',d1+'_median',d2+'_median','absCov_'+d1,'absCov_'+d2,'norCov_'+d1,'norCov_'+d2]
        statsdf = pd.DataFrame(index=bin_df.ContigName.unique(),columns=columns)
        print(bin_num)
        statsdf['Bin'] = bin_num
        statsdf['length'] = bin_df.groupby('ContigName').max()['Position']
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
