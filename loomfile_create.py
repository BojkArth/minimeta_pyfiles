#!/usr/bin/env python

import pandas as pd
import numpy as np
import loompy


import sys
sys.path.append('/home/bojk/Data/minimeta_pyfiles/')

"""
12-06-19 Bojk Berghuis
makes loomfile from counttable csv file (rows=genes,columns=cells)
"""

def make_loomfile(path,filename):
    countTable = pd.read_csv('Datasets/TuPaMetaDataDivya/CombinedCountTable.csv',index_col=0)
    filename = "Datasets/TuPaMetaDataDivya/CombinedCountTable2.loom"
    matrix = countTable.values
    row_attrs = { "Genes": np.array(countTable.index) }
    col_attrs = { "Cells": np.array(countTable.columns) }
    loompy.create(filename, matrix, row_attrs, col_attrs)



if __name__ == '__main__':

    path = sys.argv[1]
    filename = sys.argv[2]
    make_loomfile(path,filename)
    
