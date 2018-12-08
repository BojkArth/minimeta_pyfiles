#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys
import HTSeq
import os
from os import listdir
from os.path import isfile, join

# filters a fasta file in folder based on sequence length


def filter_fasta_length(path,inputfile,threshold):

    input_file = path+inputfile+'.fasta'

    #check if subdir exists:
    directory = path+'filtered'+str(threshold)+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    output_file = directory+inputfile+'.filt'+str(threshold)+'.fasta' #store in same folder as df

    with open(output_file, 'w') as out_file:
        for s in HTSeq.FastaReader( input_file ):
            if len(s.seq)>=threshold:
                s.write_to_fasta_file(out_file)
        out_file.close()
    return()

def filter_all_fasta(path,threshold):
    filelist = [f for f in listdir(path) if isfile(join(path, f)) and '.fasta' in f ]
    for files in filelist:
        filter_fasta_length(path,files[0:-6],threshold)

# command line use
if __name__ == "__main__":
    # input args
    if len(sys.argv)==4: # then you have a single file as input
        path2fasta = sys.argv[1]
        fasta_name = sys.argv[2]
        threshold = sys.argv[3]
        filter_fasta_length(path2fasta,fasta_name,threshold)
    elif len(sys.argv==3): # then you want an entire directory with fastas to be filtered
        path2fasta = sys.argv[1]
        threshold = sys.argv[2]
        filter_all_fasta(path2fasta,threshold)

