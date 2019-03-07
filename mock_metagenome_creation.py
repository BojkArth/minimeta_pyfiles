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
    kmers = [''.join(i) for i in itertools.product('ACTG',repeat=kmernum)] #dim = 1024 5mers, 256 4mers, 4096 6mers
    kmerdf = pd.DataFrame(columns=kmers)
    contigdf = pd.DataFrame(columns=['Sequence length', 'GC','description'])
    #counts = khmer.Counttable(kmernum,1e7,kmernum+10)
    for s in HTSeq.FastaReader(fastapath):
        seqstring = s.seq.decode('utf-8')
        length = len(seqstring)
        GC = (seqstring.count('G')+seqstring.count('C'))/len(seqstring)
        contigdf.loc[s.name] = {'Sequence length':length,'GC':GC,'description':s.descr}
        #counts.consume(seqstring)
        for kmer in kmers:
            #kmerdf.loc[seq.name,kmer] = sum(newcontig[i:].startswith(kmer) for i in range(len(newcontig)))
            kmerdf.loc[s.name,kmer] = len(re.findall('(?='+kmer+')',seqstring))
            #kmerdf.loc[s.name,kmer] = counts.get(kmer)
    entire_end = time.time()
    print('Constructing '+str(kmernum)+'-mer table from '+fastapath.split('/')[-1]+' took {:.2f} s'.format(entire_end - entire_start))
    contigdf.to_pickle(savedir+'contigdf_from_metagenome_fasta_'+str(kmernum)+'mer.pickle')
    kmerdf.to_pickle(savedir+'kmerdf_from_metagenome_fasta_'+str(kmernum)+'mer.pickle')
    return(contigdf,kmerdf)

def distance_calc(p1,p2):
    x1 = p1[0];y1 = p1[1]
    x2 = p2[0];y2 = p2[1]
    dist = np.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

def distance_matrix_from_tsne(tSNE_df):
    dist_matrix = pd.DataFrame(index=tSNE_df.index,columns=tSNE_df.index)
    for contig in dist_matrix.columns:
        dist_matrix[contig] = distance_calc(tSNE_df.loc[contig],tSNE_df)
    return dist_matrix

def make_tsne_from_df(df,contig_df,stats_df,maindir,savename):
    # compute tsne
    x = StandardScaler().fit_transform(df)
    x_emb = TSNE(n_components=2,perplexity=40,random_state=23944).fit_transform(x)
    tsnedf = pd.DataFrame(x_emb,index=df.index)
    tsnedf = tsnedf.join(contig_df['Sequence length'])
    tsnedf = tsnedf.join(contig_df['GC'])
    tsnedf['genome'] = [f.split('_')[0] for f in tsnedf.index]
    tsnedf.to_pickle(maindir+'tsnedf_'+savename+'.pickle')

    # calculate cluster quality
    QCdf = make_QCdf_from_tsnedf(tsnedf,maindir,savename)

    # plot some metrics
    f,ax = plt.subplots(1,1,figsize=(10,10))
    plt.scatter(tsnedf[0],tsnedf[1],s=tsnedf['Sequence length'].astype(float)/100,
                alpha=.05,c=tsnedf['GC'],cmap='RdBu_r')
    f.savefig(maindir+'plots/tSNE_GC_'+savename+'.png')
    plt.close(f)

    f,ax = plt.subplots(figsize=(8,8))
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
    plt.close(f)
    return tsnedf, QCdf


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

    # plot # (with selection constraints as text on fig)
    f, ax = plt.subplots(figsize=(12,12))
    maindf.plot.scatter(xcol,ycol,s=size.divide(3e2).astype(int),alpha=.05,ax=ax,c=colors)
    for cluster in stats.index:
        ax.annotate(str(cluster), (stats.loc[cluster,xcol],stats.loc[cluster,ycol]))

    plt.xlabel('tSNE 1');plt.ylabel('tSNE 2')
    f.savefig(maindir+'plots/'+expt_name+'hdbscan.png')

    return(maindf,stats)

# make function that computes hdb clustering quality (e.g. confusion matrix, correlation of cluster with GT-genome, etc.)
# make another function that adds coverage to a set of given contigs (with known GT)
def hdb_clustering_QC(tsnedf_withBins,statsdf):
    return 0
