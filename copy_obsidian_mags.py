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
from shutil import copyfile



#plots fasttsne output from ascii files and saves the data as pickles
#two dirs should exist in path: 'dfs' and 'figs'
#naming depends on what the matlab scripts on the singlecell cluster puts out

#path = sys.argv[1] #folder containing files that need to be numbered
#num = sys.argv[2] #starting number in 3-digit string format

path = '/home/datastorage/IMG_ANNOTATION_DATA/Obsidian_MAGs/'
obsidian_dirs = [f for f in listdir(path) if 'Knumbers' not in f and 'NN_MAGs' not in f]
newpath = '/home/datastorage/IMG_ANNOTATION_DATA/Obsidian_proteins/'

for mag  in obsidian_dirs:
    filepath = path+mag+'/IMG_Data/'+mag[-10:]+'/'+mag[-10:]
    gff = filepath+'.gff'
    faa = filepath+'.genes.faa'

    newgff = newpath+mag[0:-15]+'.gff'
    newfaa = newpath+mag[0:-15]+'.genes.faa'

    copyfile(gff,newgff)
    copyfile(faa,newfaa)



