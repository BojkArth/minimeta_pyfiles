#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import ast
import sys

# makes a dataframe from the checkm output and saves as pickle (no plotting)
# input1  = folder containing the checkm output
# input2 = name of file (how you want it saved)

def read_checkm_tsv(inputdir):
    # read the checkm bin_stats_ext.tsv file (this has the other metadata like GC etc)
    metadata = list(open(inputdir+'storage/bin_stats_ext.tsv'))
    metadata = [m.split('\n') for m in metadata]
    metadf = pd.DataFrame.from_records(metadata)
    metadf = metadf.drop(1,1)
    metadf.columns = ['bla']
    metadf = metadf['bla'].str.split('\t',1,expand=True)
    metadf = metadf.set_index(0)
    metadf.columns = ['metadata']

    columns = ['marker lineage','Completeness','Contamination','# contigs','Genome size','# genomes','# marker sets','# markers','# predicted genes','Coding density','GC','GC std','Longest contig','Mean contig length','N50 (contigs)']
    metadata_obs = pd.DataFrame(index=metadf.index,columns=columns)

    for d in metadf.index:
        dictionary = ast.literal_eval(metadf.loc[d,'metadata'])
        for j in metadata_obs.columns:
            metadata_obs.loc[d,j] = dictionary[j]
    checkm_metadata = metadata_obs.copy()
    #checkm_metadata.to_pickle(filedir+'metadata_pickle')
    return(checkm_metadata)

def make_checkm_output_df(indir,filen):
    # run the read_checkm_tsv and add the outputfile.txt
    meta = read_checkm_tsv(indir)
    #print(meta)
    strhet = pd.read_table(indir+'outputfile.txt')
    strhet = strhet.set_index('Bin Id')

    meta = meta.join(strhet['Strain heterogeneity'])
    meta = meta.sort_values('Completeness',ascending=False)
    meta.to_csv(indir+filen+'.txt','\t')
    meta.to_pickle(indir+filen)
    return(meta)

if __name__ == '__main__':
    filedir=sys.argv[1]
    filename = sys.argv[2]
    checkm_output = make_checkm_output_df(filedir,filename);
