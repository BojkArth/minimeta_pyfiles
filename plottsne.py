#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import ast
import sys
import fromasciitodf as fa

from os import listdir
from os.path import isfile, join

#plots fasttsne output from ascii files and saves the data as pickles
#two dirs should exist in path: 'dfs' and 'figs'
#naming depends on what the matlab scripts on the singlecell cluster puts out

path = sys.argv[1] #folder containing ascii files that are the output of the fast-tSNE (barnes-hutt) on the singlecell cluster
files = [f for f in listdir(path) if isfile(join(path, f))]

#bins = [f[-7:-4] for f in files];bindf = pd.DataFrame(bins)
#binun = bindf[0].unique()

for fil in files:
    namstr = fil[0:-4]
    binna = namstr[36:]
    per = namstr[0:35]
    name = binna+'_'+per

    a = list(open(path+fil))
    perplexity = fil[33:35]
    print(perplexity+'_'+namstr[-3:])
    df = fa.fromasciitodf(a,perplexity)
    df.to_pickle(path+'dfs/'+name)

    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(7)
    gs = gridspec.GridSpec(1,2)
    ax1 = f.add_subplot(gs[0,0])
    xcol = df.columns[0];ycol = df.columns[1]
    df.plot.scatter(xcol,ycol,alpha=0.3,ax=ax1)
    plt.title('absolute')

    ax2 = f.add_subplot(gs[0,1])
    xcol = df.columns[3];ycol = df.columns[4]
    df.plot.scatter(xcol,ycol,alpha=0.3,ax=ax2)
    plt.title('normalized')


    plt.suptitle(name)
    #plt.show()


    f.savefig(path+'figs/'+name+'.png')




