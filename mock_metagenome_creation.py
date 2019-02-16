#!/usr/bin/env python

import itertools
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import HTSeq
import os
import re

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
"""


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
                        for kmer in kmers:
                            #kmerdf.loc[seq.name,kmer] = sum(newcontig[i:].startswith(kmer) for i in range(len(newcontig)))
                            kmerdf.loc[seq.name,kmer] = len(re.findall('(?='+kmer+')',newcontig))
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
    for s in HTSeq.FastaReader(fastapath):
        seqstring = s.seq.decode('utf-8')
        length = len(seqstring)
        GC = (seqstring.count('G')+seqstring.count('C'))/len(seqstring)
        contigdf.loc[s.name] = {'Sequence length':length,'GC':GC,'description':s.descr}
        for kmer in kmers:
            kmerdf.loc[s.name,kmer] = len(re.findall('(?='+kmer+')',newcontig))
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
