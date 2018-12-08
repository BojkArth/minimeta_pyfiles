#!/usr/bin/env python


def build_annotation_dfs(indir)
    #mypath = '/home/datastorage/IMG_ANNOTATION_DATA/Obsidian_MAGs/'
    foldernames = [f for f in listdir(indir) if ('Knumbers') not in f and ('NN_MAGs') not in f]

    IMG_ID = pd.DataFrame(index=range(len(foldernames)),columns=['folder_name','number','genome'])
    #foldername['folder_name'] = [f for f in foldernames]
    #foldername['number'] = [f[-10:] for f in foldernames]
    #foldername['genome'] = [f[0:15] for f in foldernames]
    ktab = pd.DataFrame(index=range(10000),columns=IMG_ID)
    ctab = pd.DataFrame(index=range(10000),columns=IMG_ID)
    ptab = pd.DataFrame(index=range(10000),columns=IMG_ID)
    ctab_thr = pd.DataFrame(index=range(10000),columns=IMG_ID) # threshold hit value

    # make KO-list table and write to txt file in Knumbers folder
    for i in range(0,len(IMG_ID)):
        kofile = mypath+foldername[i]+'/IMG_Data/'+foldername.loc[i,'number']+'/'+foldername.loc[i,'number']+'.ko.tab.txt'
        ktab[foldername.loc[i,'genome']] = pd.read_csv(kofile,'\t')['ko_id'].str[3:]
        ktab[foldername.loc[i,'genome']].to_csv(mypath+'Knumbers/'+foldername.loc[i,'genome']+'.txt',sep='\t')
        # make COG-list table
        cofile = mypath+foldername.loc[i,'folder_name']+'/IMG_Data/'+foldername.loc[i,'number']+'/'+foldername.loc[i,'number']+'.cog.tab.txt'
        ctab[foldername.loc[i,'genome']] = pd.read_csv(cofile,'\t')['cog_id']
        ctab_thr[foldername.loc[i,'genome']] = pd.read_csv(cofile,'\t')['percent_identity']
        #ctab[foldername.loc[i,'genome']].to_csv(mypath+'Knumbers/'+foldername.loc[i,'genome']+'.txt',sep='\t')
        # make pfam-list table
        pffile = mypath+foldername.loc[i,'folder_name']+'/IMG_Data/'+foldername.loc[i,'number']+'/'+foldername.loc[i,'number']+'.pfam.tab.txt'
        pfamtemp =  pd.read_csv(pffile,'\t')[['pfam_id','pfam_name']]
        pfamtemp['namenum'] = pfamtemp['pfam_id']+', '+pfamtemp['pfam_name']
        ptab[foldername.loc[i,'genome']] = pfamtemp['namenum']


    # COG dfs with different percent identity threshold values
    ctab_30 = ctab[ctab_thr>30]
    ctab_40 = ctab[ctab_thr>40]
    ctab_50 = ctab[ctab_thr>50]
    ctab_60 = ctab[ctab_thr>60]
    ctab_70 = ctab[ctab_thr>70]
    ctab_80 = ctab[ctab_thr>80]

