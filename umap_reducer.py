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

import umap

def umap_reducer(inputdir,inputfile,savename):
    #ind = pd.read_pickle('5mers').index
    kmertable = pd.read_pickle(inputdir+inputfile)
    embedding  = umap.UMAP().fit_transform(kmertable)
    print(embedding)

    f = plt.figure()
    gs = gridspec.GridSpec(1,1)
    ax1=f.add_subplot(gs[0,0])
    plt.scatter(embedding[:,0],embedding[:,1],alpha=.3,ax=ax1)
    f.set_figheight(10)
    f.set_figwidth(10)
    plt.show()
    f.savefig(inputdir+savename)
    plt.close(f)

if __name__ == '__main__':
    """
    command line
    USAGE = python make_umap.py $inputdir $inputfile $savename
    """
    inputdir = sys.argv[1]
    inputfile = sys.argv[2]
    savename = sys.argv[3]
    umap_reducer(inputdir,inputfile,savename)


