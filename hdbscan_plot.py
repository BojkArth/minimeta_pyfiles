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


def HDBscan_plot(df,perplexity,norm,args,kwds,plot_kwds):
    start_time = time.time()
    #plot_kwds = {'alpha' : 0.25,'s' : 30, 'linewidths':0}
    if norm == "yes":
        nor = 'n'
        no = 'normalized_cov'
    else:
        nor = 'a'
        no = 'absolute_cov'
    perpl = '_'+str(perplexity)
    xcol = 'x'+perpl+nor
    ycol = 'y'+perpl+nor
    minsize = kwds['min_cluster_size']
    if xcol in df.columns:
        temp = df.loc[:,[xcol,ycol]]
        size = df.loc[:,'Sequence Length']
        labels = hdbscan.HDBSCAN(*args, **kwds).fit_predict(temp)
        end_time = time.time()
        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        df['DBclusternum'] = labels[0:len(df)] #add cluster numbers to main df
        meanpos = df.groupby('DBclusternum').mean()[[xcol,ycol]]
	# plotting
        f = plt.figure()
        gs = gridspec.GridSpec(1,1)
        ax1 = f.add_subplot(gs[0,0])
        plt.scatter(temp[xcol], temp[ycol], s = size.astype(int)/1e2, c=colors, **plot_kwds)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        clustnum = len(np.unique(labels[labels>-1]))
        plt.suptitle(str(clustnum)+' clusters found by HDBscan'
			 +' for\n'+name+', '+no+', perplx = '+perpl[1:]+', min clust size = '
			 +str(minsize)+'\nmin samples = '+str(hdbkwds['min_samples'])+', cluster selection method = '+str(hdbkwds['cluster_selection_method']), fontsize=18)
        for txt in meanpos.index:
            ax1.annotate(str(txt), (meanpos.loc[txt,xcol],meanpos.loc[txt,ycol]))
        f.set_figwidth(10)
        f.set_figheight(10)
        f.savefig(path+name+'_HDBSCAN_clustering'+perpl+nor+'MinCluSi_'+str(minsize)+'.png')
        plt.close()
        df.to_pickle(path+name+'_hdb_perp_'+perpl+'MinCluSi_'+str(minsize))
        return(df)

path = sys.argv[1]
name = sys.argv[2]
data = pd.read_pickle(path+name)
print(sys.argv[4])
perps = [30,40,50,60,70,80]
hdbkwds = {'min_cluster_size':int(sys.argv[3]),'min_samples':int(sys.argv[4]),'cluster_selection_method':'leaf','alpha':1.0} # default cluster_selection_method = 'eom' (excess of mass)
hdbargs = ()
pltkwds = {'alpha' : 0.3, 'linewidths':0}

for perp in perps:
    HDBscan_plot(data,perp,'no',hdbargs,hdbkwds,pltkwds)

 
