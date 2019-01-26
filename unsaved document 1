#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import ast
import sys

from urllib.request import Request, urlopen, URLError
from urllib.parse import urlencode
from lxml import etree
import lxml
import html2text as h2t
from html.parser import HTMLParser

class myhtmlparser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()

        self.NEWTAGS = []
        self.NEWATTRS = []
        self.HTMLDATA = []
    def handle_starttag(self, tag, attrs):
        self.NEWTAGS.append(tag)
        self.NEWATTRS.append(attrs)
    def handle_data(self, data):
        self.HTMLDATA.append(data)
    def clean(self):
        self.NEWTAGS = []
        self.NEWATTRS = []
        self.HTMLDATA = []


def retrieveKEGGmodules(df,typedata,main_dir):
    #typedata can be: 'all','complete', or 'conservative'
    if typedata == 'conservative':
        mode = 'complete+ng1'
    else:
        mode = typedata

    # make dataframe
    Mlist = pd.DataFrame(columns=df.columns)
    # make a datafame containing the KEGG entries for known hits. This is incomplete!
    # as I have hits not showing up in the known genomes.
    # edit Oct 2017: just make a range of len(max(Modules in KEGG database))
    dta = list(range(0,845))
    dta2 = ['M0000'+str(dta[s]) for s in range(0,
                        10)]+['M000'+str(dta[s]) for s in range(10,
                            100)]+['M00'+str(dta[s]) for s in range(100,len(dta))]
    #Mmap = pd.DataFrame(index=known_genomes.index,columns=Klist.columns)
    Mmap = pd.DataFrame(index=dta2,columns=df.columns)


    a=0
    start_time = time.time()
    for s in df.columns:

        #get list of KEGG genes for a given cluster (tab-delimited text)
        with open(main_dir+'Knumbers/'+s+'.txt', 'rt') as f:
            dat = f.read()

        # retrieve data from KEGG database
        url = 'http://www.genome.jp/kegg-bin/find_module_object'
        params = urlencode({'unclassified': dat, 'mode': mode}).encode()
        html = urlopen(url, params).read()


        # save as html file for reference purposes (and look at detailed output)
        with open(main_dir+'Knumbers/html/'+s+'_'+typedata+'.html', 'wb') as f:
            f.write(html)

        # convert data to list
        pstring = html.decode('utf8')
        parser = myhtmlparser()
        parser.feed(pstring)

        # Extract data from parser
        tags  = parser.NEWTAGS
        attrs = parser.NEWATTRS
        data  = parser.HTMLDATA

        # Clean the parser
        parser.clean()

        # extract Module numbers and feed to dataframe
        indices = [i for i, s in enumerate(data) if 'M00' in s]
        mdata = [data[s] for s in indices]
        mdatadf = pd.DataFrame(mdata,columns=[s])

        if a==0:
            Mlist = mdatadf.copy()
            a=1
        else:
            Mlist = Mlist.join(mdatadf,how='outer')
        Mmap.loc[mdatadf.set_index(s).index,s] = 1

    end_time = time.time()
    print('Retrieving modules from KEGG database took {:.2f} s'.format(end_time - start_time))
    Mlist.to_pickle(main_dir+'Knumbers/module_list_'+typedata)
    Mmap.to_pickle(main_dir+'Knumbers/module_map_'+typedata)

    return (Mlist,Mmap)
