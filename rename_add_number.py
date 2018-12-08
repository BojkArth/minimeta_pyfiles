#!/usr/bin/env python

import numpy as np
import ast
import sys
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import json

"""
renames (to format used for reassembly) and adds 3-digit number to the file
USAGE: python rename_add_number path_to_files starting_number
"""

path = sys.argv[1] #folder containing files that need to be numbered
num = int(sys.argv[2]) #starting number in int format
startnum = num
files = [f for f in listdir(path) if isfile(join(path, f)) and '.fasta' in f]
newfiles = []
for f  in files:
    filepath = path+f
    newname = 'genome_cluster.'+"{0:0=3d}".format(num)+'.fasta'
    newpath = path+'newfiles/'+newname
    copyfile(filepath,newpath)
    num+=1
    newfiles = newfiles+[newname]

filedict = dict(zip(files,newfiles))
# write dict for further reference
with open(path+'newfiles/renamed_fastas_'+"{0:0=3d}".format(startnum)+'-'+"{0:0=3d}".format(num-1)+'.txt', 'w') as file:
    file.write(json.dumps(filedict))



