#!/user/bin/env python
# this is for the snakemake reassembly pipeline on Sherlock
# 201812

def make_genome_cluster_file(bins):
    nums = list(range(bins))
    nums2 = ['genome_cluster']
    for n in nums:
        nums2.append("{0:0=3d}".format(n))
    with open('genome_cluster_file.txt', 'w') as f:
        for item in nums2:
            f.write("%s\n" % item)
