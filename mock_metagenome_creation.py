#!/usr/bin/env python

import itertools
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import HTSeq
import os
import re
#import khmer
import imageio
import glob
import hdbscan
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

from sys import platform
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
Created 2019-02-14	Bojk Berghuis

Functions:
fragment_genomes_make_kmertable		the function that does everything: make mock metagenome from a collection of reference genomes (fna files)
					INPUT:	directory containing genome folders, list of lengths, kmer length (int), specific name to add to saved files
					OUTPUT:	genome fastas (with N50 similar to the list of lengths provided)
						metagenome fasta (idem)
						statsdf: pickle/df containing the genome stats (total length, #contigs, N50, GC) before and after fragmentation
						contigdf: pickle/df containing the new contigs for all genomes (length, GC)
						kmerdf: pickle/df of shape (#contigs, kmer dimensionality) with the kmer counts for each contig

make_kmertable_from_fasta_contigs	if you only want to produce the kmer frequency spectrum, and not change the metagenome by reconstructing a (slightly) different one,
					use this.
					INPUT:	metagenome fasta
						kmer length (int)
					OUTPUT: contigdf: pickle/df containing the new contigs for all genomes (length, GC)
						kmerdf: pickle/df of shape (#contigs, kmer dimensionality) with the kmer counts for each contig

make_tsne_from_df			perform dimensional reduction on given kmer dataframe. Dataframe will be scaled to mean=0, variance=1
					INPUT:	df: the dataframe containing the multidimensional data (can be kmers or PCAs of kmers, for instance)
						contigdf: output df from 'fragment_genomes_make_kmertable' function
						statsdf: output df from 'fragment_genomes_make_kmertable' function
						maindir: same directory as main function
						savename: specific string to be added to output files and figsize
					OUTPUT:	tsnedf: dataframe containing the 2D coordinates of the reduced data (using TSNE algorithm)
						QCdf: dataframe containing the quality of clustering (based on the ground truth of the contigs that is known)
"""
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def fragment_genomes_make_kmertable(maindir,contiglengths,kmernum,savename):
    """
    This function creates a mock metagenome from a collection of genomes downloaded from IMG
    The genome sequences should be in .fna format in a folder with the genome name: maindir/[genome_name]/[genome_name].fna
    Other functions in this module can extract kmertables from existing fastas, but this function does everything.

    "maindir"		the directory that contains the folder of genomes "maindir/[genome_name]/[genome_name].fna"
    "contiglengths"	a list of contiglengths provided to pull new sequence lengths at random from (I use my PermafrostFranklinBluffs
    			metagenome here first, since this is a large (60k) collection of contigs with a length distribution
    			that I want to emulate.).
    "kmernum"		length k (in number of bases) of the fragment that will be used to calculate the base-frequency spectrum for
    "savename"		the specific name you want to add to the output (metagenome fasta, statsdf, contigdf, kmerdf)

    Note that since were pulling sequence lengths at random from a given collection of sequence lengths from one of my own metagenomes,
    each time you run this you will end up with a slightly different metagenome (the overall picture should be roughly similar of course)
    This means that the kmertable, contigdf, and statsdf belong to a specific metagenome fasta created at the same time.
    To recreate kmertable, contigdf and statsdf from an existing metagenome fasta created like this, run other functions in this module.

    """
    folders = [f for f in os.listdir(maindir) if 'fasta' not in f and 'plots' not in f and 'stats' not in f]
    print('List of genomes:')

    # make metagenome from genomes in folder (with N50 representing my data)
    statsdfcols = ['orig_totlen','orig_contigs','orig_N50','orig_maxlen','orig_GC','orig_GCstd',
                  'frag_totlen','frag_contigs','frag_N50','frag_maxlen','frag_GC','frag_GCstd']
    statsdf = pd.DataFrame(index=folders,columns=statsdfcols)
    kmers = [''.join(i) for i in itertools.product('ACTG',repeat=kmernum)] #dim = 1024 5mers, 256 4mers, 4096 6mers
    kmerdf = pd.DataFrame(columns=kmers)
    contigdf = pd.DataFrame(columns=['Sequence length', 'GC','description'])

    entire_start = time.time()
    metagenome_file = maindir+'metagenome_from_allgenomes_'+savename+'.fasta'
    with open(metagenome_file,'w') as meta_out:
        for file in folders:
            fasta = maindir+file+'/'+file+'.fna'
            output_file = maindir+file+'_fragmented_'+savename+'.fasta'
            contig_count = 0
            frag_seqLen=[];orig_seqLen=[];orig_GC=[];frag_GC=[]
            start = time.time()
            with open(output_file,'w') as out_file:
                for s in HTSeq.FastaReader(fasta):
                    seqstring = s.seq.decode('utf-8')
                    orig_seqLen.append(len(seqstring))
                    GC = (seqstring.count('G')+seqstring.count('C'))/len(seqstring)
                    orig_GC.append(GC)
                    while True:
                        newlength = contiglengths[np.random.randint(len(contiglengths))] # pull random length from distribution of sequence lengths
                        frag_seqLen.append(newlength)
                        GC = (seqstring[:newlength].count('G')+seqstring[:newlength].count('C'))/len(seqstring[:newlength])
                        frag_GC.append(GC)
                        newcontig = seqstring[:newlength]
                        seq = HTSeq.Sequence(newcontig.encode('utf-8'),name=file+'_contig_'+"{0:0=3d}".format(contig_count))
                        seq.descr=s.name.replace(' ','_')+'_'+s.descr.replace(' ','_')
                        # add contig info to contigdf
                        contigdf.loc[seq.name] = {'Sequence length':newlength,'GC':GC,'description':seq.descr}
                        # calculate kmerfrequency
                        #counts = khmer.Counttable(kmernum,1e7,kmernum+10)
                        #counts.consume(newcontig)
                        for kmer in kmers:
                            #kmerdf.loc[seq.name,kmer] = sum(newcontig[i:].startswith(kmer) for i in range(len(newcontig)))
                            kmerdf.loc[seq.name,kmer] = len(re.findall('(?='+kmer+')',newcontig))
                            #kmerdf.loc[seq.name,kmer] = counts.get(kmer)
                        seq.write_to_fasta_file(out_file)
                        seq.write_to_fasta_file(meta_out)
                        seqstring = seqstring[newlength:]
                        contig_count+=1
                        if newlength>len(seqstring):
                            if len(seqstring)>=5e3: #only write remaining to fasta if >5kb
                                frag_seqLen.append(len(seqstring))
                                GC = (seqstring.count('G')+seqstring.count('C'))/len(seqstring)
                                frag_GC.append(GC)
                                seq = HTSeq.Sequence(seqstring.encode('utf-8'),name=file+'_contig_'+"{0:0=3d}".format(contig_count))
                                seq.descr=s.name.replace(' ','_')+'_'+s.descr.replace(' ','_')
                                contigdf.loc[seq.name] = {'Sequence length':len(seqstring),'GC':GC,'description':seq.descr}
                                contig_count+=1
                                for kmer in kmers:
                                    kmerdf.loc[seq.name,kmer] = sum(seqstring[i:].startswith(kmer) for i in range(len(seqstring)))
                                seq.write_to_fasta_file(out_file)
                                seq.write_to_fasta_file(meta_out)
                            break
                out_file.close()
            statsdf.loc[file,'orig_totlen'] = sum(orig_seqLen)
            statsdf.loc[file,'frag_totlen'] = sum(frag_seqLen)
            statsdf.loc[file,'orig_contigs'] = len(orig_seqLen)
            statsdf.loc[file,'frag_contigs'] = len(frag_seqLen)
            statsdf.loc[file,'orig_N50'] = np.median(orig_seqLen)
            statsdf.loc[file,'frag_N50'] = np.median(frag_seqLen)
            statsdf.loc[file,'orig_maxlen'] = max(orig_seqLen)
            statsdf.loc[file,'frag_maxlen'] = max(frag_seqLen)
            statsdf.loc[file,'orig_minlen'] = min(orig_seqLen)
            statsdf.loc[file,'frag_minlen'] = min(frag_seqLen)
            statsdf.loc[file,'orig_GC'] = np.mean(orig_GC)
            statsdf.loc[file,'frag_GC'] = np.mean(frag_GC)
            statsdf.loc[file,'orig_GCstd'] = np.std(orig_GC)
            statsdf.loc[file,'frag_GCstd'] = np.std(frag_GC)
            end = time.time()
            print('Collecting kmers for genome '+file+' took {:.2f} s'.format(end - start))
        statsdf.to_pickle(maindir+'stats/statsdf_'+savename+'.pickle')
        contigdf.to_pickle(maindir+'stats/contigdf_'+savename+'.pickle')
        kmerdf.to_pickle(maindir+'stats/kmerdf_'+savename+'.pickle')
        entire_end = time.time()
        print('Constructing mock metagenome from '+maindir+'\n with kmer = '+str(kmernum)+' took {:.2f} s'.format(entire_end - entire_start))
        meta_out.close()
    return(statsdf,contigdf,kmerdf)



def make_kmertable_from_fasta_contigs(fastapath,kmernum,savedir):
    entire_start = time.time()
    # count total number of sequences in fasta:
    if platform == "linux" or platform == "linux2":
        # linux
        numseq = 16985 # do this here for now
    elif platform == "darwin":
        # OS X
        bashCommand = "grep '>' "+fastapath+" | wc -l"
        command = os.popen(bashCommand)
        out = command.read()
        numseq = out.strip()
    elif platform == "win32":
        numseq='we dont use windows'
    print('Total number of fasta sequences:'+str(numseq))
    
    if float(numseq)<10:
        print('aborting, something went wrong in counting fasta lines')
        return 0

    kmers = [''.join(i) for i in itertools.product('ACTG',repeat=kmernum)] #dim = 1024 5mers, 256 4mers, 4096 6mers

    # check if previous file already exists
    if os.path.isfile(savedir+'kmerdf_from_metagenome_fasta_'+str(kmernum)+'mer_incomplete.pickle'):
        kmerdf = pd.read_pickle(savedir+'kmerdf_from_metagenome_fasta_'+str(kmernum)+'mer_incomplete.pickle')
        contigdf = pd.read_pickle(savedir+'contigdf_from_metagenome_fasta_'+str(kmernum)+'mer_incomplete.pickle')
        batchcount = len(kmerdf)
        fastacount = batchcount
        print('The first '+str(fastacount)+' kmers have already been collected for this fasta, loaded from local path:')
        print(savedir+'kmerdf_from_metagenome_fasta_'+str(kmernum)+'mer_incomplete.pickle')
    else:
        kmerdf = pd.DataFrame(columns=kmers)
        contigdf = pd.DataFrame(columns=['Sequence length', 'GC','description'])
        batchcount = 0
        fastacount = 0
    #counts = khmer.Counttable(kmernum,1e7,kmernum+10)


    start = time.time()
    print('----------------------------------------------------')
    print('Started collecting kmers, dimensionality = '+str(len(kmers)))
    hour = time.localtime()[3];minute = time.localtime()[4]
    print('Local time: '+str(hour)+':'+str(minute))
    print('----------------------------------------------------')
    for s in HTSeq.FastaReader(fastapath):
        if s.name not in kmerdf.index:
            seqstring = s.seq.decode('utf-8')
            length = len(seqstring)
            GC = (seqstring.count('G')+seqstring.count('C'))/len(seqstring)
            contigdf.loc[s.name] = {'Sequence length':length,'GC':GC,'description':s.descr}
            #counts.consume(seqstring)
            for kmer in kmers:
                #kmerdf.loc[seq.name,kmer] = sum(newcontig[i:].startswith(kmer) for i in range(len(newcontig)))
                kmerdf.loc[s.name,kmer] = len(re.findall('(?='+kmer+')',seqstring))
                #kmerdf.loc[s.name,kmer] = counts.get(kmer)
            fastacount+=1
        end = time.time()
        if fastacount==500 or fastacount-batchcount==500:
            batchcount = batchcount+500
            hour = time.localtime()[3]
            minute = time.localtime()[4]
            print('----------------------------------------------------')
            print('Collecting kmers for '+str(fastacount)+' contigs out of '+str(numseq)+' done, took {:.2f} s'.format(end-start))
            print('Progress: '+str(np.round(fastacount/int(numseq)*1e2,2))+' %, local time: '+str(hour)+':'+str(minute))
            print('Total elapsed time is {:.2f} minutes'.format((end-entire_start)/60))
            print('Saving current contig kmers to pickle...')
            kmerdf.to_pickle(savedir+'kmerdf_from_metagenome_fasta_'+str(kmernum)+'mer_incomplete.pickle')
            contigdf.to_pickle(savedir+'contigdf_from_metagenome_fasta_'+str(kmernum)+'mer_incomplete.pickle')
            print('relative path = '+savedir+'kmerdf_from_metagenome_fasta_'+str(kmernum)+'mer_incomplete.pickle')
            print('----------------------------------------------------')
            start = time.time()

    entire_end = time.time()
    hour = time.localtime()[3]
    minute = time.localtime()[4]
    print('Constructing '+str(kmernum)+'-mer table from '+fastapath.split('/')[-1]+' took {:.2f} hours'.format((entire_end - entire_start)/3600))
    print('Finished at local time: '+str(hour)+':'+str(minute))
    contigdf.to_pickle(savedir+'contigdf_from_metagenome_fasta_'+str(kmernum)+'mer.pickle')
    kmerdf.to_pickle(savedir+'kmerdf_from_metagenome_fasta_'+str(kmernum)+'mer.pickle')
    return(contigdf,kmerdf)

def make_coverage(contigdf,statsdf,maindir,savename):
    # compute coverage by getting mean genome coverage from lognormal distribution
    # assign exact coverage to each contig by pulling from normal distribution centered around mean genome cov
    num_genomes = len(statsdf)
    mu, sigma = 5., 1. # mean and standard deviation
    coverage = np.random.lognormal(mu, sigma, num_genomes)
    coverage = pd.DataFrame(coverage)
    coverage.rename(index=str,columns={0:'mean_coverage'},inplace=True)
    coverage.set_index(statsdf.index,inplace=True)
    a=[f for f in time.localtime()]
    b = [str("{0:0=2d}".format(f)) for f in a[1:]]
    timestamp = [str(a[0])+b[0]+b[1]+'_'+b[2]+':'+b[3]+':'+b[4]]
    timestamp = timestamp[0]
    coverage.to_pickle(maindir+'stats/'+'coverage'+savename+'_'+timestamp)
    statsdf['mean_coverage'] = coverage['mean_coverage']
    for genome in contigdf.genome:
        idx = contigdf[contigdf.genome==genome].index
        mu = statsdf.loc[genome,'mean_coverage'] # mean of normal distribution
        contigdf.loc[idx,'coverage_frac'] = np.random.normal(mu,.1*mu,len(idx)) # normal 1
        contigdf.loc[idx,'coverage_log'] = np.random.normal(mu,np.log(mu),len(idx)) # normal 2
    contigdf.to_pickle(maindir+'stats/contigdf_'+savename+'_'+timestamp) # for reproducibility
    statsdf.to_pickle(maindir+'stats/statsdf_'+savename+'_'+timestamp) # for reproducibility

    keys = statsdf.index; values = sns.color_palette('deep',len(keys))
    lut = dict(zip(keys,values))
    colors = contigdf.genome.map(lut)
    f,ax = plt.subplots(figsize=(8,8))
    contigdf.plot.scatter('GC','coverage_frac',s=contigdf['Sequence length'].astype(float).divide(1e2),color=colors,alpha=.2,ax=ax)
    plt.yscale('log')
    plt.ylabel('Coverage')

    f.savefig(maindir+'plots/GCvsCoverage_'+savename+'_'+timestamp+'.png')
    
    return contigdf,statsdf
 
"""
The functions above create the metagenome samples and kmer frequency tables.
Below is everything else. What you need is kmer, contig and genome stats dataframe

"""

"""def distance_calc(p1,p2):
    x1 = p1[0];y1 = p1[1]
    x2 = p2[0];y2 = p2[1]
    dist = np.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

def distance_matrix_from_tsne(tSNE_df):
    dist_matrix = pd.DataFrame(index=tSNE_df.index,columns=tSNE_df.index)
    for contig in dist_matrix.columns:
        dist_matrix[contig] = distance_calc(tSNE_df.loc[contig],tSNE_df)
    return dist_matrix"""
    
def perform_complete_analysis(kmerdf,contigdf,statsdf,**kwargs):
    """
    Given a set of kmers, this function performs a full parameter sweep of
    	number of PCs
    	minimum cluster size (HDBscan)
    kwargs must contain:
    	maindir
    	savename
    	coverage_included
    """
    maindir,savename,coverage,scaling = kwargs['maindir'],kwargs['savename'],kwargs['coverage_included'],kwargs['scaling']
    global num_kmers
    num_kmers = len(kmerdf.T)
    kmer_length = int(np.log(num_kmers)/np.log(4))
    start_time = time.time()

    savename = savename+'_'+str(kmer_length)+'mers'

    maindir = maindir
    global complete_analysis
    complete_analysis = 'YES'
    global firstround
    firstround = 'YES'
    if kmer_length==5:
        pcs_to_reduce = [int(round(f)) for f in np.logspace(0,3,20)[8:-4]]
    elif kmer_length>=6:
        pcs_to_reduce = [int(round(f)) for f in np.logspace(0,3,20)[8:-4]]
    else:
        print('define range of PCs first!')
        return 0

    optimaldf = pd.DataFrame(index=pcs_to_reduce+[num_kmers],columns=['idxmax','max'])
    
    kmerdf_norm = kmerdf.divide(contigdf['Sequence length'],axis=0)
    if coverage=='YES':
        kmerdf_norm = kmerdf_norm.join(contigdf['coverage_frac'])
        print('Contig coverage added as dimension')

    ##################################################
    ##################################################
    if os.path.isfile(maindir+'tsnedf_'+savename+'.pickle'):
        print('tSNE-df previously made, loading from pickle, full path = ')
        print(maindir+'tsnedf_'+savename+'.pickle')
        tsnedf_main = pd.read_pickle(maindir+'tsnedf_'+savename+'.pickle')
        plot_tsnedf_metrics(tsnedf_main,maindir,savename+'_'+str(num_kmers)+'_dimensions')
    else:
        if scaling==True:
            print('building tSNE of all '+str(num_kmers)+' dimensions')
            tsnedf_main = make_tsne_from_df(kmerdf_norm,contigdf,statsdf,maindir,savename)
        elif scaling==False:
            print('building tSNE of all '+str(num_kmers)+' dimensions, scaling done')
            tsnedf_main = make_tsne_from_df(kmerdf_norm,contigdf,statsdf,maindir,savename)
        end = time.time()
        plot_tsnedf_metrics(tsnedf_main,maindir,savename+'_'+str(num_kmers)+'_dimensions')
        print('finished building main tSNE, this took {:.2f} seconds'.format(end - start_time))
        print('performing cluster sweep of tSNE of all '+str(num_kmers)+' dimensions')
    ##################################################
    ##################################################

    start = time.time()
    hdbsweep = minCS_sweep(tsnedf_main,maindir,savename,savename)
    end = time.time()
    print('finished cluster sweep of main tSNE, this took {:.2f} seconds'.format(end - start))
    print('Total elapsed time is {:.2f} minutes'.format((end-start_time)/60))

    keys = [0,1];values = ['x_'+str(kmer_length)+'mers','y_'+str(kmer_length)+'mers']
    newnames = dict(zip(keys,values))
    tsnedf_main.rename(index=str,columns=newnames,inplace=True)
    
    optimaldf.loc[num_kmers,'max'] = hdbsweep['val_leaf'].max()
    optimaldf.loc[num_kmers,'idxmax'] = hdbsweep['val_leaf'].idxmax()
    
    keys = hdbsweep.columns;values = [f+str(kmer_length)+'mers' for f in keys]
    lut = dict(zip(keys,values))
    hdbsweep.rename(index=int,columns=lut,inplace=True)
     
    firstround=='NO'
    ##################################################
    ##################################################
    if os.path.isfile(maindir+'PCAdf_'+savename+'.pickle'):
        print('PCA performed earlier, loading file:\n'+maindir+'PCAdf_'+savename+'.pickle')
        pcdf = pd.read_pickle(maindir+'PCAdf_'+savename+'.pickle')
        if os.path.isfile(maindir+savename.split('_')[0]+str(kmer_length)+'mers_all_tSNEs_temp'):
            tsnedf_main = pd.read_pickle(maindir+savename.split('_')[0]+str(kmer_length)+'mers_all_tSNEs_temp')
            optimaldf = pd.read_pickle(maindir+savename.split('_')[0]+'_optimalValues_perPC_temp')
            hdbsweep = pd.read_pickle(maindir+savename.split('_')[0]+'_allPCs_clustersweep_Quality_temp')
            cols = tsnedf_main.columns
            pcs_done = list(set([int(f[4:]) for f in cols[4:]]))
            [pcs_to_reduce.remove(f) for f in pcs_done]
            print('Some tSNEs already calculated, resuming at dim reduction of '+str(pcs_to_reduce[0])+' PCs')
    else:
        if scaling==True:
            print('Performing PCA...')
            pcdf = perform_PCA(kmerdf_norm)
            pcdf.to_pickle(maindir+'PCAdf_'+savename+'.pickle')
        elif scaling==False:
            print('Performing PCA...')
            pcdf = perform_PCA_ns(kmerdf_norm)
            pcdf.to_pickle(maindir+'PCAdf_unscaled_'+savename+'.pickle')
        print('PCA done and saved.')
    ##################################################
    ##################################################
    ##################################################
    ##################################################
    for numpcs in pcs_to_reduce:
        print('building tSNE of '+str(numpcs)+' PCs')
        start = time.time()
        tsnedf_temp = make_tsne_from_df_ns(pcdf.iloc[:,:numpcs+1],contigdf,statsdf,maindir,savename)
        end = time.time()
        print('finished building tSNE of '+str(numpcs)+' PCs, this took {:.2f} seconds'.format(end - start))
        
        
        savetemp = savename.split('_')[0]+'_'+str(numpcs)+'PCs'
        print('performing cluster sweep of tSNE of '+str(numpcs)+' PCs')
        start = time.time()
        hdb_temp = minCS_sweep(tsnedf_temp,maindir,savetemp,savetemp)
        end = time.time()
        print('finished cluster sweep of tSNE with'+str(numpcs)+'PCs, this took {:.2f} seconds'.format(end - start))
        print('Total elapsed time is {:.2f} minutes'.format((end-start_time)/60))

        keys = [0,1];values = ['x_PC'+str(numpcs),'y_PC'+str(numpcs)]
        newnames = dict(zip(keys,values))
        tsnedf_main = tsnedf_main.join(tsnedf_temp[[0,1]])
        tsnedf_main.rename(index=str,columns=newnames,inplace=True)
        tsnedf_main.to_pickle(maindir+savename.split('_')[0]+str(kmer_length)+'mers_all_tSNEs_temp')
        
        optimaldf.loc[numpcs,'max'] = hdb_temp['val_leaf'].max()
        optimaldf.loc[numpcs,'idxmax'] = hdb_temp['val_leaf'].idxmax()
        optimaldf.to_pickle(maindir+savename.split('_')[0]+'_optimalValues_perPC_temp')
        
        keys = hdb_temp.columns;values = [f+'_PC'+str(numpcs) for f in keys]
        lut = dict(zip(keys,values))
        hdb_temp.rename(index=int,columns=lut,inplace=True)
        hdbsweep = hdbsweep.join(hdb_temp)
        hdbsweep.to_pickle(maindir+savename.split('_')[0]+'_allPCs_clustersweep_Quality_temp')
    ##################################################
    ##################################################
            
    end = time.time()
    print('Finished everything.')
    print('Total elapsed time is {:.2f} minutes'.format((end-start_time)/60))
    tsnedf_main.to_pickle(maindir+savename.split('_')[0]+'_all_tSNEs')
    optimaldf.to_pickle(maindir+savename.split('_')[0]+'_optimalValues_perPC')
    hdbsweep.to_pickle(maindir+savename.split('_')[0]+'_allPCs_clustersweep_Quality')
    return tsnedf_main,optimaldf,hdbsweep

def make_Opt_tSNE(final_df,maindir,alltsnes_df_pickle,savename):
    """
    Makes the tSNE dataframe with optimal number of PCs and HDBscan params
    These are calculated by the "perform_complete_analysis" function
    final_df = optimaldf
    alltsnes_df_pickle = pickle of tsnedf_main
    """
    tsne_all = pd.read_pickle(maindir+alltsnes_df_pickle)
    max_v_meas = final_df['max'].max()
    numPCsOpt = final_df['max'].astype(float).idxmax()
    minCS_Opt = final_df.loc[numPCsOpt,'idxmax']
    print('Max V-measure = '+str(np.round(max_v_meas,3))+', with '+str(numPCsOpt)+' PCs and minimum cluster size of '+str(minCS_Opt)+' contigs.')
    #cols = [f+'_leaf_PC'+str(numPCsOpt) for f in ['hom','com','val']]
    tsne_x,tsne_y = 'x_PC'+str(numPCsOpt),'y_PC'+str(numPCsOpt)
    tsne_opt = tsne_all[[tsne_x,tsne_y,'Sequence length','GC','genome']].copy()
    lut = dict(zip([tsne_x,tsne_y],[0,1]))
    tsne_opt.rename(index=str,columns=lut,inplace=True)
    keys = ['min_cluster_size','min_samples','cluster_selection_method','allow_single_cluster','expt_name']
    values = [int(minCS_Opt),1,'leaf',True,savename]
    kwds = dict(zip(keys,values))
    tsne_opt, statsOpt, Qmeas = cluster_main_tsne(tsne_opt,maindir,**kwds)
    
    """compute completeness and heterogeneity (#contigs and sequence length)"""
    for binn in statsOpt.index:
        if binn!=-1:
            statsOpt.loc[binn,'seq_len'] = tsne_opt[tsne_opt.DBclusternum==binn].groupby('genome').sum()['Sequence length'].max()
            statsOpt.loc[binn,'seq_len_idx'] = tsne_opt[tsne_opt.DBclusternum==binn].groupby('genome').sum()['Sequence length'].idxmax()
            statsOpt.loc[binn,'num_contigs'] = tsne_opt[tsne_opt.DBclusternum==binn].groupby('genome').count()['Sequence length'].max()
            statsOpt.loc[binn,'num_contigs_idx'] = tsne_opt[tsne_opt.DBclusternum==binn].groupby('genome').count()['Sequence length'].idxmax()
            genome = statsOpt.loc[binn,'num_contigs_idx']
            genome2 = statsOpt.loc[binn,'seq_len_idx']
            statsOpt.loc[binn,'completeness'] = np.round(statsOpt.loc[binn,'num_contigs']/len(tsne_opt[tsne_opt.genome==genome])*100,2)
            statsOpt.loc[binn,'completeness(seqlen)'] = np.round(statsOpt.loc[binn,'seq_len']/tsne_opt[tsne_opt.genome==genome2]['Sequence length'].sum()*100,2)
            statsOpt.loc[binn,'GTgenome_in_other_cluster(seqlen)'] = tsne_opt[(tsne_opt.genome==genome2)&(tsne_opt.DBclusternum!=binn)&(tsne_opt.DBclusternum!=-1)]['Sequence length'].sum()
            statsOpt.loc[binn,'GTgenome_unclustered(seqlen)'] = tsne_opt[(tsne_opt.genome==genome2)&(tsne_opt.DBclusternum==-1)]['Sequence length'].sum()
            statsOpt.loc[binn,'GTgenome_in_other_cluster'] = len(tsne_opt[(tsne_opt.genome==genome)&(tsne_opt.DBclusternum!=binn)&(tsne_opt.DBclusternum!=-1)])
            statsOpt.loc[binn,'GTgenome_unclustered'] = len(tsne_opt[(tsne_opt.genome==genome)&(tsne_opt.DBclusternum==-1)])
            statsOpt.loc[binn,'%genome_unclus'] = np.round(statsOpt.loc[binn,'GTgenome_unclustered']/len(tsne_opt[tsne_opt.genome==genome])*100,2)
            statsOpt.loc[binn,'%genome_inother'] = np.round(statsOpt.loc[binn,'GTgenome_in_other_cluster']/len(tsne_opt[tsne_opt.genome==genome])*100,2)
            statsOpt.loc[binn,'%genome_unclus(seqlen)'] = np.round(statsOpt.loc[binn,'GTgenome_unclustered(seqlen)']/tsne_opt[tsne_opt.genome==genome2]['Sequence length'].sum()*100,2)
            statsOpt.loc[binn,'%genome_inother(seqlen)'] = np.round(statsOpt.loc[binn,'GTgenome_in_other_cluster(seqlen)']/tsne_opt[tsne_opt.genome==genome2]['Sequence length'].sum()*100,2)


    statsOpt['homogeneity'] = np.round(statsOpt['num_contigs'].divide(statsOpt['# contigs'])*100,2)
    statsOpt['contamination'] = np.round(100 - statsOpt['homogeneity'],2)
    statsOpt['homogeneity(seqlen)'] = np.round(statsOpt['seq_len'].divide(statsOpt['total length'])*100,2)
    statsOpt['contamination(seqlen)'] = np.round(100 - statsOpt['homogeneity(seqlen)'],2)
    # add number of contigs and seqlen of genome in other clusters, as well as unclustered 
    genomes_unrepresented = list(set(tsne_opt.genome).difference(set(statsOpt.seq_len_idx.dropna())))
    unclust_idx = ['unclus_'+str(f) for f in range(len(genomes_unrepresented))]
    unclustered = pd.DataFrame(index=unclust_idx,columns=['seq_len_idx','%genome_unclus','%genome_unclus(seqlen)','%genome_inother','%genome_inother(seqlen)'])
    unclustered['seq_len_idx'] = genomes_unrepresented
    for genome in unclustered.seq_len_idx:
        idx = unclustered[unclustered.seq_len_idx==genome].index
        unclustered.loc[idx,'GTgenome_unclustered(seqlen)'] = tsne_opt[(tsne_opt.genome==genome)&(tsne_opt.DBclusternum==-1)]['Sequence length'].sum()
        unclustered.loc[idx,'GTgenome_in_other_cluster(seqlen)'] = tsne_opt[(tsne_opt.genome==genome)&(tsne_opt.DBclusternum!=-1)]['Sequence length'].sum()
        unclustered.loc[idx,'GTgenome_unclustered'] = len(tsne_opt[(tsne_opt.genome==genome)&(tsne_opt.DBclusternum==-1)])
        unclustered.loc[idx,'GTgenome_in_other_cluster'] = len(tsne_opt[(tsne_opt.genome==genome)&(tsne_opt.DBclusternum!=-1)])
        unclustered.loc[idx,'%genome_unclus(seqlen)'] = np.round(unclustered.loc[idx,'GTgenome_unclustered(seqlen)']/tsne_opt[(tsne_opt.genome==genome)]['Sequence length'].sum()*100,2)
        unclustered.loc[idx,'%genome_inother(seqlen)'] = np.round(unclustered.loc[idx,'GTgenome_in_other_cluster(seqlen)']/tsne_opt[(tsne_opt.genome==genome)]['Sequence length'].sum()*100,2)
        unclustered.loc[idx,'%genome_unclus'] = np.round(unclustered.loc[idx,'GTgenome_unclustered']/len(tsne_opt[(tsne_opt.genome==genome)])*100,2)
        unclustered.loc[idx,'%genome_inother'] = np.round(unclustered.loc[idx,'GTgenome_in_other_cluster']/len(tsne_opt[(tsne_opt.genome==genome)])*100,2)
    statsOpt = statsOpt.append(unclustered,sort=False,ignore_index=False)
    
    # make adjusted rand index
    keys = list(statsOpt['seq_len_idx']);values = list(range(-1,len(keys)))
    mapper = dict(zip(keys,values))
    counts = statsOpt.seq_len_idx.value_counts()
    doubles = counts[counts>1].index.values # check for genomes split over multiple clusters (take genome that is longest as true)
    if len(doubles)>0:
        for dob in doubles:
            print(dob)
            cluster = tsne_opt[tsne_opt.genome==dob].groupby('DBclusternum').sum()['Sequence length'].idxmax()
            mapper[dob] = cluster
    statsOpt['True_label'] = statsOpt['seq_len_idx'].map(mapper)
    tsne_opt['True_label'] = tsne_opt['genome'].map(mapper)
    tsne_opt['True_label'] = tsne_opt['True_label'].astype(int)
    from sklearn.metrics.cluster import adjusted_rand_score
    ARI = adjusted_rand_score(list(tsne_opt['True_label']),list(tsne_opt['DBclusternum']))


    """
    PLOTTING
    """
    f,ax = plt.subplots(figsize=(5,13))
    idx = statsOpt.drop(-1,axis=0)[['seq_len_idx','completeness(seqlen)','%genome_unclus(seqlen)','%genome_inother(seqlen)']].sort_values('completeness(seqlen)',ascending=False).index
    statsOpt.drop(-1,axis=0)[['seq_len_idx','completeness(seqlen)','%genome_unclus(seqlen)','%genome_inother(seqlen)']].sort_values('completeness(seqlen)',ascending=False).plot.barh(stacked=True,ax=ax,cmap='Paired')
    tix = list(statsOpt.loc[idx,'seq_len_idx'])
    loc,label = plt.yticks()
    combined = [str(y)+',  '+str(x) for x,y in zip(tix,idx)]
    plt.yticks(loc,combined)
    plt.gca().invert_yaxis()
    plt.legend(['Correct call (TP)','Unclustered (FN)','Incorrect call (FP)'],loc='upper right',bbox_to_anchor=(1.7,1))
    plt.title('Cluster-genome quality, ARI = '+str(np.round(ARI,3)))
    plt.xlabel('Assembled sequence length (%)')
    f.savefig(maindir+'plots/'+savename+'_clusterGTgenomeQuality.png')
    

    tsne_opt.to_pickle(maindir+'stats/'+savename+'_tSNE_OptimalClustering')
    statsOpt.to_pickle(maindir+'stats/'+savename+'_OptimalClustering_stats')
    print('----------------------------------------------------')
    print('Sequence-based average genome recovery is {:.1f}%'.format(statsOpt['completeness(seqlen)'].drop(-1).fillna(0).mean()))
    print('Median genome recovery is {:.1f}%'.format(statsOpt['completeness(seqlen)'].drop(-1).fillna(0).median()))
    print('Average recovery of clustered genomes is {:.2f}%'.format(statsOpt['completeness(seqlen)'].dropna().mean()))
    print('Median recovery of clustered genomes is {:.2f}%'.format(statsOpt['completeness(seqlen)'].dropna().median()))
    perc_unseen = len(genomes_unrepresented)/len(statsOpt)*100
    print('Adjusted rand index = '+str(np.round(ARI,3)))
    print('Number of unseen genomes: '+str(int(len(genomes_unrepresented)))+'/'+str(int(len(statsOpt)))+', or '+str(np.round(perc_unseen,2))+'%')
    print('----------------------------------------------------')    

    return tsne_opt,statsOpt


def perform_PCA(kmer_df):
    x = StandardScaler().fit_transform(kmer_df)
    pca = PCA(n_components=num_kmers)
    principalComp = pca.fit_transform(x)
    princdf = pd.DataFrame(principalComp)
    princdf.index =kmer_df.index
    return(princdf)

def perform_PCA_ns(kmer_df): #no scaling version
    #x = StandardScaler().fit_transform(kmer_df)
    pca = PCA(n_components=num_kmers)
    principalComp = pca.fit_transform(kmer_df)
    princdf = pd.DataFrame(principalComp)
    princdf.index =kmer_df.index
    return(princdf)

def make_tsne_from_df(df,contig_df,stats_df,maindir,savename):
    # compute tsne
    x = StandardScaler().fit_transform(df)
    x_emb = TSNE(n_components=2,perplexity=40,random_state=23944).fit_transform(x)
    tsnedf = pd.DataFrame(x_emb,index=df.index)
    tsnedf = tsnedf.join(contig_df['Sequence length'])
    tsnedf = tsnedf.join(contig_df['GC'])
    if firstround=='YES' and complete_analysis=='YES':
        tsnedf['genome'] = [f.split('_')[0] for f in tsnedf.index]
        tsnedf.to_pickle(maindir+'tsnedf_'+savename+'.pickle')
    if complete_analysis!='YES':
        # calculate cluster quality
        QCdf = make_QCdf_from_tsnedf(tsnedf)
        # plot some metrics
        plot_tsnedf_metrics(tsnedf,stats_df,QCdf)
        return tsnedf, QCdf
    else:
        return tsnedf

def make_tsne_from_df_ns(df,contig_df,stats_df,maindir,savename): #no scaling version
    # compute tsne
    #x = StandardScaler().fit_transform(df)
    x_emb = TSNE(n_components=2,perplexity=40,random_state=23944).fit_transform(df)
    tsnedf = pd.DataFrame(x_emb,index=df.index)
    tsnedf = tsnedf.join(contig_df['Sequence length'])
    tsnedf = tsnedf.join(contig_df['GC'])
    if firstround=='YES' and complete_analysis=='YES':
        tsnedf['genome'] = [f.split('_')[0] for f in tsnedf.index]
        tsnedf.to_pickle(maindir+'tsnedf_'+savename+'.pickle')
    if complete_analysis!='YES':
        # calculate cluster quality
        QCdf = make_QCdf_from_tsnedf(tsnedf)
        # plot some metrics
        plot_tsnedf_metrics(tsnedf,stats_df,QCdf)
        return tsnedf, QCdf
    else:
        return tsnedf


def plot_tsnedf_metrics(tsnedf,maindir,savename):
    f,ax = plt.subplots(1,1,figsize=(10,10))
    plt.scatter(tsnedf[0],tsnedf[1],s=tsnedf['Sequence length'].astype(float)/100,
                alpha=.05,c=tsnedf['GC'],cmap='RdBu_r')
    f.savefig(maindir+'plots/tSNE_GC_'+savename+'.png')
    plt.close(f)

    """f,ax = plt.subplots(figsize=(8,8))
    ax.scatter(stats_df['frag_GC'],QCdf['norm_mean_dist'],
               s=stats_df['frag_totlen'].divide(3e3),alpha=.4,
              c=stats_df['kmer_variance'],cmap='RdBu_r')
    plt.xlabel('GC-content')
    plt.ylabel('Mean contig distance (fraction of max contig distance)')
    f.savefig(maindir+'plots/genome_clusterQuality_'+savename+'_GC.png')
    plt.close(f)

    f,ax = plt.subplots(figsize=(8,8))
    ax.scatter(stats_df['kmer_variance'],QCdf['norm_mean_dist'],
               s=stats_df['frag_totlen'].divide(3e3),alpha=.4,
              c=stats_df['frag_GC'],cmap='RdBu_r')
    plt.xlabel('Normalized genome kmer variance')
    plt.ylabel('Mean contig distance (fraction of max contig distance)')
    f.savefig(maindir+'plots/genome_clusterQuality_'+savename+'_kmerVar.png')
    plt.close(f)"""


def make_QCdf_from_tsnedf(tsnedf,maindir,savename):
    QC_df = pd.DataFrame(index=tsnedf.genome.unique(),columns=['total_distance','num_contigs'])
    for binn in QC_df.index:
        X = tsnedf[tsnedf.genome==binn][[0,1]]
        dist = euclidean_distances(X,X)
        distdf = pd.DataFrame(dist)
        medist = pd.DataFrame(index=range(len(dist)),columns=['medx','medy'])
        medist['medx']=X[0].median() #this is to compute distance to median
        medist['medy']=X[1].median()
        QC_df.loc[binn,'median_x'] = X[0].median() # median position of contigs in a genome
        QC_df.loc[binn,'median_y'] = X[1].median()
        QC_df.loc[binn,'mean_x'] = X[0].mean() # average position of contigs in a genome
        QC_df.loc[binn,'mean_y'] = X[1].mean()
        QC_df.loc[binn,'R_median'] =  pd.DataFrame(euclidean_distances(X,medist[['medx','medy']]))[0].median()
        QC_df.loc[binn,'R_mean'] = pd.DataFrame(euclidean_distances(X,medist[['medx','medy']]))[0].mean()
        QC_df.loc[binn,'total_distance'] = distdf.sum().sum()/2
        QC_df.loc[binn,'num_contigs'] = len(distdf)
    QC_df['dist_from_origin_mean'] = [f[0] for f in euclidean_distances(QC_df[['mean_x','mean_y']],[[0,0]])] # distance from origin for mean(x,y)
    QC_df['dist_from_origin_median'] = [f[0] for f in euclidean_distances(QC_df[['median_x','median_y']],[[0,0]])] # distance from origin for median(x,y)
    #maxdist = np.max(euclidean_distances(tsnedf[[0,1]],tsnedf[[0,1]]))
    maxdist_fromorigin = QC_df['dist_from_origin_median'].max()

    QC_df['mean_dist'] = QC_df['total_distance'].divide(QC_df['num_contigs']**2/2) #mean distance of all contigs belonging to a genome
    QC_df['norm_mean_dist'] = QC_df['mean_dist'].divide(maxdist_fromorigin) # mean distance scaled to max (median) distance (Rmax) ###### formerly: the max distance between any two contigs in tSNE

    maxdist_fromorigin = np.max(euclidean_distances(tsnedf[[0,1]],[[0,0]]))

    QC_df['med_mean_diff'] = np.diagonal(euclidean_distances(QC_df[['median_x','median_y']],QC_df[['mean_x','mean_y']])) # distance between mean and median
    QC_df['frac_med_mean_diff'] = QC_df['med_mean_diff'].divide(maxdist_fromorigin)
    QC_df['frac_mean_orig'] = QC_df['dist_from_origin_mean'].divide(maxdist_fromorigin) # distance from origin scaled to max (median) distance (Rmax)
    QC_df['frac_median_orig'] = QC_df['dist_from_origin_median'].divide(maxdist_fromorigin)

    # find nearest neighbor
    x = QC_df[['median_x','median_y']]
    dists = euclidean_distances(x,x)
    np.fill_diagonal(dists,np.inf)
    idx = pd.DataFrame(dists).idxmin()
    QC_df['NN_name'] = QC_df.iloc[idx].index
    QC_df['NN_dist'] = list(pd.DataFrame(dists).min()) # distance of the nearest neigboring GT-genome
    QC_df['NN_dist_norm'] = QC_df['NN_dist'].divide(maxdist_fromorigin) # scaled to Rmax
    QC_df['NN_mindist'] = QC_df['R_median']+list(QC_df.loc[QC_df['NN_name'],'R_median']) # sum of two median radii (needs to be smaller than NN_dist to classify as non-overlapping)
    QC_df['non_overlapping'] = QC_df['NN_dist']>QC_df['NN_mindist']

    #compute quality score
    QC_df['QS'] = QC_df['frac_median_orig']*.1+(1-QC_df['norm_mean_dist'])+QC_df['NN_dist_norm']
    QC_df['QS'] = (1-QC_df['frac_med_mean_diff'])*(1-QC_df['norm_mean_dist'])*QC_df['NN_dist_norm']
    QC_df['QS_rank'] = QC_df.rank(axis=0,ascending=False)['QS']

    #save and return
    QC_df.to_pickle(maindir+'QCdf_'+savename+'.pickle')
    return(QC_df)



def circle(x,y,r):
    t = np.arange(0,2*np.pi,.01)
    cx = x+r*np.sin(t)
    cy = y+r*np.cos(t)
    return(cx,cy)

def plot_ground_truth_circles(tsne_df,QC_df,genome_ID,maindir,savename):
    info = QC_df.loc[genome_ID]
    x,y = circle(info['median_x'],info['median_y'],info['R_median'])
    x2,y2 = circle(info['median_x'],info['median_y'],info['R_mean'])
    f,ax = plt.subplots(figsize=(13,11))
    tsne_df[tsne_df.genome!=genome_ID].plot.scatter(0,1,alpha=.05,s=tsne_df['Sequence length'].astype(float).divide(2e2),c='gray',ax=ax,label=None)
    tsne_df[tsne_df.genome==genome_ID].plot.scatter(0,1,alpha=.2,s=tsne_df['Sequence length'].astype(float).divide(2e2),c='orange',ax=ax,label='genome '+genome_ID)
    plt.plot(info['median_x'],info['median_y'],'o',label='median contig pos')
    plt.plot(info['mean_x'],info['mean_y'],'or',label='mean contig pos')
    plt.plot(x,y,label='R=median dist')
    plt.plot(x2,y2,label='R=mean dist')
    plt.legend(bbox_to_anchor=(1.4, 1), loc='upper right')
    plt.gcf().subplots_adjust(right=0.75)
    f.savefig(maindir+'plots/ground_truth_circles_'+genome_ID+'_'+savename+'.png')
    plt.close(f)


def make_gif(image_path,save_name):
    #image_path = maindir+'plots/'
    #image_path = 'Permafrost/SagMAT/bins/fasta/originals/plots/'
    files = glob.glob(image_path+'*.png')
    files = np.sort(files)
    images = []
    for file in files:
        # imageio.imread(file) creates a numpy matrix array
        # In this case a 200 x 200 matrix for every file, since the files are 200 x 200 pixels.
        images.append(imageio.imread(file))
        print(file)
    imageio.mimsave(image_path+save_name+'.gif', images)

def plot_all_circles(tsne_df,QC_df,savedir,savename):
    f,ax = plt.subplots(figsize=(12,12))
    tsne_df.plot.scatter(0,1,alpha=.05,s=tsne_df['Sequence length'].astype(float).divide(2e2),
                         c='gray',ax=ax,label=None)
    plt.plot([QC_df['median_x'],QC_df['mean_x']],[QC_df['median_y'],QC_df['mean_y']],
             '-om',label='median contig pos')
    ax.plot(QC_df['median_x'],QC_df['median_y'],'o',label='median contig pos')
    ax.plot(QC_df['mean_x'],QC_df['mean_y'],'or',label='mean contig pos')
    for genome in QC_df.index:
        dat = QC_df.loc[genome]
        x,y = circle(dat['median_x'],dat['median_y'],dat['R_median'])
        x2,y2 = circle(dat['median_x'],dat['median_y'],dat['R_mean'])
        ax.plot(x,y,label='R=median dist',c='r')
        ax.plot(x2,y2,label='R=mean dist',c='b')
    f.savefig(savedir+savename+'_all_circles.png')

def plot_all_circles_thres(tsne_df,QC_df,thres,savedir,savename):
    QC_df['radrat'] = QC_df['R_median'].divide(QC_df['R_mean'])
    QC_df['thres'] = QC_df['frac_mean_orig']-QC_df['frac_med_mean_diff']
    indf = QC_df[QC_df.thres>=thres]
    oudf = QC_df[QC_df.thres<thres]
    f,ax = plt.subplots(figsize=(12,12))
    tsne_df.plot.scatter(0,1,alpha=.05,s=tsne_df['Sequence length'].astype(float).divide(2e2),
                         c='gray',ax=ax,label=None)
    #plt.plot([QC_df['median_x'],QC_df['mean_x']],[QC_df['median_y'],QC_df['mean_y']],
    #         '-om',label='median contig pos')
    ax.plot(QC_df['median_x'],QC_df['median_y'],'o',label='median contig pos')
    #ax.plot(QC_df['mean_x'],QC_df['mean_y'],'or',label='mean contig pos')
    for genome in QC_df.index:
        dat = QC_df.loc[genome]
        x,y = circle(dat['median_x'],dat['median_y'],dat['R_median'])
        x2,y2 = circle(dat['median_x'],dat['median_y'],dat['R_mean'])
        if genome in indf.index:
            ax.plot(x,y,label='R=median dist',c='r')
            ax.plot(x2,y2,label='R=mean dist',c='b')
        elif genome in oudf.index:
            ax.plot(x,y,label='R=median dist',c='gray')
            ax.plot(x2,y2,label='R=mean dist',c='gray')
        f.savefig(savedir+savename+'_all_circles_somethres.png')

def plot_all_circles_nonOverlap(tsne_df,QC_df,savedir,savename):

    indf = QC_df[QC_df.non_overlapping==True]
    oudf = QC_df[QC_df.non_overlapping==False]
    f,ax = plt.subplots(figsize=(12,12))
    tsne_df.plot.scatter(0,1,alpha=.05,s=tsne_df['Sequence length'].astype(float).divide(2e2),
                         c='gray',ax=ax,label=None)
    #plt.plot([QC_df['median_x'],QC_df['mean_x']],[QC_df['median_y'],QC_df['mean_y']],
    #         '-om',label='median contig pos')
    ax.plot(QC_df['median_x'],QC_df['median_y'],'o',label='median contig pos')
    #ax.plot(QC_df['mean_x'],QC_df['mean_y'],'or',label='mean contig pos')
    for genome in QC_df.index:
        dat = QC_df.loc[genome]
        x,y = circle(dat['median_x'],dat['median_y'],dat['R_median'])
        x2,y2 = circle(dat['median_x'],dat['median_y'],dat['R_mean'])
        ax.text(dat['median_x'],dat['median_y'],int(dat['QS_rank']),fontsize=20)
        if genome in indf.index:
            ax.plot(x,y,label='R=median dist',c='r')
            ax.plot(x2,y2,label='R=mean dist',c='b')
        elif genome in oudf.index:
            ax.plot(x,y,label='R=median dist',c='gray')
            ax.plot(x2,y2,label='R=mean dist',c='gray')
        f.savefig(savedir+savename+'_all_circles_nonOverlap.png')

def cluster_main_tsne(maindf,maindir,**kwargs):
    # set variables
    #if len(maindf)<1000: #this is a hack, as in usually you provide a df that goes in, but now it can also be a path to pickle
    #    df = pd.read_pickle(maindf)
    """ hdb kwargs:'min_cluster_size','min_samples','cluster_selection_method':'leaf','alpha':1.0, allow_single_cluster=False (default)} # default cluster_selection_method = 'eom' (excess of mass)
        other kwargs: 'perplexity_na','expt_name','write_fasta','write_df','is_this_a_series'
    """
    minCS = kwargs['min_cluster_size']
    minS = kwargs['min_samples']
    CSM = kwargs['cluster_selection_method']
    ASC = kwargs['allow_single_cluster']
    expt_name = kwargs['expt_name']
    maindf['Sequence length'] = maindf['Sequence length'].astype(int)


    xcol = 0; ycol = 1;
    temp = maindf[[xcol,ycol]]
    size = maindf['Sequence length']

    labels = hdbscan.HDBSCAN(min_cluster_size=minCS,min_samples=minS,cluster_selection_method=CSM,allow_single_cluster=ASC).fit_predict(temp)

    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    maindf.loc[maindf.iloc[0:len(maindf)].index,'DBclusternum'] = labels[0:len(maindf)] #add cluster numbers to main df
    maindf['Bin'] = ["{0:0=3d}".format(f) for f in maindf['DBclusternum']] # convert into 3-digit string
    maindf[[xcol,ycol]] = maindf[[xcol,ycol]].astype(float)
    stats = maindf.groupby('DBclusternum').mean()[[xcol,ycol]] #mean position in tSNE space


    stats = stats.join(maindf.groupby('DBclusternum').sum()['Sequence length']) # sequence length
    stats.rename(index=int,columns={'Sequence length':'total length'},inplace=True)
    stats = stats.join(maindf.groupby('DBclusternum').count()['Bin']) # 
    stats.rename(index=int,columns={'Bin':'# contigs'},inplace=True)
    params = pd.DataFrame.from_dict(kwargs,orient='index')
    params.rename(index=str,columns={0:'parameter values'},inplace=True)
    
    #if complete_analysis!='YES':
    # plot # (with selection constraints as text on fig)
    f, ax = plt.subplots(figsize=(12,12))
    maindf.plot.scatter(xcol,ycol,s=size.divide(3e2).astype(int),alpha=.05,ax=ax,c=colors)
    for cluster in stats.index:
        ax.annotate(str(cluster), (stats.loc[cluster,xcol],stats.loc[cluster,ycol]))

    plt.xlabel('tSNE 1');plt.ylabel('tSNE 2')
    if '/' in expt_name:
        plt.title(expt_name.split('/')[1].split('_')[1]+', min cluster size = '+str(minCS)+', cluster method = '+CSM)
    else:
        plt.title(expt_name+', min cluster size = '+str(minCS)+', cluster method = '+CSM)

    f.savefig(maindir+'plots/'+expt_name+'hdbscan.png')
    plt.close(f)

    h,c,v = cluster_quality(maindf,maindir,expt_name)
    Qmeas = [h,c,v]
    return(maindf,stats,Qmeas)

# make function that computes hdb clustering quality (e.g. confusion matrix, correlation of cluster with GT-genome, etc.)
# make another function that adds coverage to a set of given contigs (with known GT)
def hdb_clustering_QC(tsnedf_withBins,statsdf):
    return 0

def plot_optimal_clustering(tsne_opt,stats_opt,maindir,subfolder,savename):
    color = sns.color_palette('Paired',15)
    #subfolder = 'feb15_5mers_optimal'
    #savename = 'feb15_5mers'
    if not os.path.isdir(maindir+'plots/'+subfolder):
        os.mkdir(maindir+'plots/'+subfolder)
    for genomeclu in stats_opt.index:
        if genomeclu!=-1:
            genome = stats_opt.loc[genomeclu,'num_contigs_idx']
            cluster = genomeclu
            f,ax = plt.subplots(figsize=(10,10))
            if len(str(cluster))<5:
                tsne_opt[(tsne_opt.genome!=genome)&(tsne_opt.DBclusternum!=cluster)].plot.scatter(0,1,alpha=.1,ax=ax,color='gray',label=None)
                tsne_opt[(tsne_opt.genome==genome)&(tsne_opt.DBclusternum==cluster)].plot.scatter(0,1,alpha=.4,ax=ax,color='b',label='Correct call')
                try:
                    tsne_opt[(tsne_opt.genome==genome)&(tsne_opt.DBclusternum==-1)].plot.scatter(0,1,alpha=.3,ax=ax,color=color[6],label='Unclustered, but part of genome')
                    tsne_opt[(tsne_opt.genome!=genome)&(tsne_opt.DBclusternum==cluster)].plot.scatter(0,1,alpha=.3,ax=ax,color=color[11],label='Incorrect call')
                except ValueError:
                    print('no incorrect calls for cluster '+str(cluster))
            else:
                tsne_opt[(tsne_opt.genome!=genome)&(tsne_opt.DBclusternum!=cluster)].plot.scatter(0,1,alpha=.1,ax=ax,color='gray',label=None)
                tsne_opt[(tsne_opt.genome==genome)&(tsne_opt.DBclusternum==-1)].plot.scatter(0,1,alpha=.3,ax=ax,color=color[6],label='Unclustered, but part of genome')
                tsne_opt[(tsne_opt.genome==genome)&(tsne_opt.DBclusternum!=-1)].plot.scatter(0,1,alpha=.3,ax=ax,color=color[11],label='Incorrect call')
            if savename=='feb15_5mers':
                plt.xlim(right=30)
            plt.title(savename+' genome '+str(genome)+' cluster '+str(cluster))
            plt.legend(loc='lower center',bbox_to_anchor=(0.5,-.1),ncol=3)
            plt.xlabel('');plt.ylabel('')
            f.savefig(maindir+'plots/'+subfolder+'/'+savename+'_genome_'+str(genome)+'_cluster_'+str(cluster)+'.png')
            plt.close(f)

def cluster_quality(tsne,maindir,savename):
    matrix = pd.DataFrame(index=tsne.Bin.unique(),columns=tsne.genome.unique())
    for genome in matrix.columns:
        matrix[genome] = tsne[tsne.genome==genome].groupby('Bin').count()[0]
    try:
        matrix.drop('-01',inplace=True) 
    except KeyError:
        print('No bin -01 in tSNE'+savename.split('_')[-1])
    genome_centric = matrix.divide(matrix.sum())
    cluster_centric = matrix.T.divide(matrix.sum(axis=1))

    homogen = homogeneity(matrix)
    complet = completeness(matrix)
    v_measure = 2*homogen*complet/(homogen+complet)

    """
    f,ax = plt.subplots(1,2,figsize=(10,5))
    cluster_centric.max().sort_values().plot(style='-o',ax=ax[0])
    ax[0].set_xticks([])
    ax[0].set_xlabel('Clusters')
    ax[0].set_ylabel('Largest single-genome fraction')

    genome_centric.max().sort_values().plot(style='-o',ax=ax[1])
    ax[1].set_xticks([])
    ax[1].set_xlabel('Genomes')
    ax[1].set_ylabel('Largest single-cluster fraction')
    f.savefig(maindir+'plots/'+savename+'_cluster_qual.png')
    plt.close(f)
    """
    return(homogen,complet,v_measure)

def homogeneity(matrix):
    n = matrix.sum().sum()
    ncn = matrix.T.sum(axis=1).divide(n)
    HC = -np.sum(ncn*np.log(ncn))
    nk = matrix.T.sum()
    HCK = 0
    for genome in matrix.columns:
        nc = -np.sum(matrix[genome].divide(n)*np.log(matrix[genome].divide(nk)))
        HCK = HCK+nc
    homogeneity=1-HCK/HC
    return homogeneity

def completeness(matrix):
    n = matrix.sum().sum()
    nkn= matrix.sum(axis=1).divide(n)
    HK = -np.sum(nkn*np.log(nkn))
    nc = matrix.sum()
    HKC = 0
    for cluster in matrix.index:
        nk = -np.sum(matrix.T[cluster].divide(n)*np.log(matrix.T[cluster].divide(nc)))
        HKC = HKC+nk
    completeness = 1-HKC/HK
    return completeness

def minCS_sweep(tsne,maindir,subfolder,savename):
    if not os.path.isdir(maindir+'plots/'+subfolder):
        os.mkdir(maindir+'plots/'+subfolder)
    min_GTgenome_contigs = tsne.groupby('genome').count()[0].min()
    sweep_pct = np.linspace(.1,2,20) # sweep minimum cluster size from 10-200% of #contigs of smallest GT-genome
    minCS_sw = [int(round(f*min_GTgenome_contigs)) for f in sweep_pct]
    leafdf = pd.DataFrame(index=minCS_sw,columns=['hom_leaf','com_leaf','val_leaf'])
    eomdf = pd.DataFrame(index=minCS_sw,columns=['hom_eom','com_eom','val_eom'])
    for mcs in minCS_sw:
        #perc_int = int(perc*1e2)
        exptna = subfolder+'/'+savename+'_eom_minCS_'+str(mcs)
        keys = ['min_cluster_size','min_samples','cluster_selection_method','allow_single_cluster',
           'expt_name']
        values = [mcs,1,'eom',True,exptna]
        kwargs = dict(zip(keys,values))
        tsne,stats,QC_eom = cluster_main_tsne(tsne,maindir,**kwargs)
        eomdf.loc[mcs] = QC_eom

        exptna = subfolder+'/'+savename+'_leaf_minCS_'+str(mcs)
        kwargs['cluster_selection_method'] = 'leaf'
        kwargs['expt_name'] = exptna
        tsne,stats,QC_leaf = cluster_main_tsne(tsne,maindir,**kwargs)
        leafdf.loc[mcs] = QC_leaf
    clusterQualdf = eomdf.join(leafdf)
    clusterQualdf = clusterQualdf.astype(float)
    return(clusterQualdf)
