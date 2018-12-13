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
    totdf = None
    for file in files:
        bin_df = pd.read_table(path+file)
        bin_num = file.split('.')[1]
        cols = bin_df.columns
        d1 = cols[-2].split('_')[1]
        d2 = cols[-1].split('_')[1]
        columns = ['Bin','length',d1+'_mean',d1+'_std',d2+'_mean',d2+'_std',d1+'_median',d2+'_median']
        statsdf = pd.DataFrame(index=bin_df.ContigName.unique(),columns=columns)

        statsdf['Bin'] = bin_num
        statsdf['length'] = bin_df.groupby('ContigName').max()['Position']
        statsdf[d1+'_mean'] = bin_df.groupby('ContigName').mean()[cols[-2]]
        statsdf[d2+'_mean'] = bin_df.groupby('ContigName').mean()[cols[-1]]
        statsdf[d1+'_std'] = bin_df.groupby('ContigName').std()[cols[-2]]
        statsdf[d2+'_std'] = bin_df.groupby('ContigName').std()[cols[-1]]
        statsdf[d1+'_median'] = bin_df.groupby('ContigName').median()[cols[-2]]
        statsdf[d2+'_median'] = bin_df.groupby('ContigName').median()[cols[-1]]

        if totdf is None:
            totdf = statsdf.copy()
        else:
            totdf = totdf.append(statsdf)

    expt_name = path.split('/')[1] #this works for Permafrost data, see top.
    totdf['expt_name'] = expt_name
    totdf['new_index'] = totdf['expt_name']+'_bin_'+totdf['Bin']+'_'+totdf.index
    totdf.to_pickle(path+expt_name+'_reassembly_contig_stats.pickle')

    time_elapsed = datetime.now() - start_time
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    print(str(len(totdf[totdf['length']>=5e3]))+' contigs above 5kb for reassembled '+expt_name)

    #need to add frac d1 frac d2 based on median and mean
    # this will allow me to compare bin-associated depth fractions before and after reassembly
    # use fractions obtained here with checkm info on N50 and length

    return(totdf)
