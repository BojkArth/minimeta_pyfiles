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

import pymer

def count_5mers(inputfile):
    #ind = pd.read_pickle('5mers').index
    kmertable = pd.DataFrame(index=['AAAAA'],columns=[]) #initialize
    for s in HTSeq.FastaReader(inputfile):
        kc = pymer.ExactKmerCounter(5)
        kc.consume(s.seq.decode('utf-8'))#read.sequence)
        a = kc.to_dict()
        b = pd.DataFrame.from_dict(a,orient='index')
        b.rename(index=str,columns={0:s.name},inplace=True)
        kmertable = kmertable.join(b,how='outer')
        kmertable.fillna(0,inplace=True)
    kmertable.to_pickle(inputfile[0:-6]+'_5mertable')
    return(kmertable)

if __name__ == '__main__':
    """
    command line
    USAGE = python make_5mers.py $inputfile
    """
    inputfile = sys.argv[1]
    count_5mers(inputfile)


