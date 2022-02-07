import glob
import os
import pandas as pd
import random
from tqdm import tqdm

orig_dir = 'storage/train/'
aug_dir = 'storage/output/220204_backtranslation/'
out_dir = 'storage/output/220205_aug_discourse_df_test/'
out_text_dir = os.path.join(out_dir,'train/')

def write_text_file(tid,text):
    f = open(os.path.join(out_text_dir,tid+'.txt'),'w')
    f.write(text)
    f.close()

os.makedirs(out_text_dir,exist_ok=True)
tids = [os.path.basename(d).replace('.csv','') for d in glob.glob(os.path.join(aug_dir,'*.csv'))]
df_data = []
for tid in tqdm(tids):
    orig_text = open(os.path.join(orig_dir,tid+'.txt'),'r').read()
    aug_df = pd.read_csv(os.path.join(aug_dir,tid+'.csv'))

    curr_text = orig_text
    disc_starts = [int(row['discourse_start']) for i,row in aug_df.iterrows()]
    disc_ends = [int(row['discourse_end']) for i,row in aug_df.iterrows()]
    aug_texts = [row['aug_discourse_text'] for i,row in aug_df.iterrows()]
    discs = [row['discourse_type'] for i,row in aug_df.iterrows()]
    disc_starts_ends = set([(start,end) for start,end in zip(disc_starts,disc_ends)])

    seg_idxs = disc_starts + disc_ends
    seg_idxs.sort()

    aug_segments,disc_segments = [],[]
    if seg_idxs[0] != 0:
        aug_segments.append(orig_text[:seg_idxs[0]])
        disc_segments.append('O')
    i = 0
    for start,end in zip(seg_idxs[:-1],seg_idxs[1:]):
        if (start,end) in disc_starts_ends:
            aug_segments.append(aug_texts[i].lstrip().rstrip())
            disc_segments.append(discs[i])
            i += 1
        else:
            aug_segments.append(orig_text[start+1:end].lstrip().rstrip())
            disc_segments.append('O')
    if seg_idxs[-1] != len(orig_text)-1:
        aug_segments.append(orig_text[seg_idxs[-1]+1:])
        disc_segments.append('O')

    count = 0
    for text,disc in zip(aug_segments,disc_segments):
        if disc != 'O':
            df_data.append(
                    (tid,count,count+len(text),disc,text,aug_df.kfold.iloc[0].item())
                    )
        count += len(text) + 1

    write_text_file(tid,' '.join(aug_segments))
out_df = pd.DataFrame(df_data)
out_df.columns = ['id','discourse_start','discourse_end','discourse_type','discourse_text','kfold']
out_df.to_csv(os.path.join(out_dir,'train.csv'),index=False)
