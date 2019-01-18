#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import ast
import sys


def fromasciitodf(inputfile,perplexity):
    perplexity = str(perplexity)
    dat = []
    data = list(pd.read_table(inputfile,header=None)[0])
    for i in data:
        dt = i.split()
        daaat = [float(dt[0]),float(dt[1])]
        dat.append(daaat)
    tsne = pd.DataFrame.from_records(dat,columns=['x','y'])
    tSNE_a = tsne.iloc[0:int(len(tsne)/2),:].rename(index=int,columns={'x':'x_'+perplexity+'a','y':'y_'+perplexity+'a'})
    tSNE_n = tsne.iloc[int(len(tsne)/2):,:].rename(index=int,columns={'x':'x_'+perplexity+'n','y':'y_'+perplexity+'n'})
    tSNE_n = tSNE_n.reset_index(drop=True)
    outdf = pd.concat([tSNE_a,tSNE_n],axis=1)
    #outdf.to_pickle()
    return(outdf)



