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
import HTSeq

def make_cluster_fasta(pathtofasta,fastaname,pathtodf,dfname):
    input_file = pathtofasta+fastaname
    df = pd.read_pickle(pathtodf+dfname)

    for binn in df['DBclusternum'].unique():
        if binn!=-1:
            temp = df[df['DBclusternum']==binn]
            output_file = pathtodf+fastaname[0:-6]+'_subbin_'+str(binn)+'.fasta' #store in same folder as df
            with open(output_file, 'w') as out_file:
                for s in HTSeq.FastaReader( input_file ):
            	    if temp.index.str.contains(s.name).any():
                        s.write_to_fasta_file(out_file)
                out_file.close()
    return()


if __name__ == "__main__":
    # input args
    path2fasta = sys.argv[1]
    fasta_name = sys.argv[2]
    path2df = sys.argv[3]
    df_name = sys.argv[4]
    # make fastas based on hdbscan clusters (df should have contig names as index, cluster numbers will be in 'DBclusternum' column)
    make_cluster_fasta(path2fasta,fasta_name,path2df,df_name)


