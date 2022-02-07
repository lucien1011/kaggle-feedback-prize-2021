import os
import pandas as pd
import random
import torch
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True ;
from tqdm import tqdm

header = '='*100
subheader = '-'*100

out_dir = 'storage/output/220204_backtranslation/'

en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', useGPU=True)
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', useGPU=True)

en2de.eval()
en2de.cuda()
de2en.eval()
de2en.cuda()

disc_df = pd.read_csv('storage/train_folds.csv')
tids = disc_df.id.unique().tolist()
random.shuffle(tids)
os.makedirs(out_dir,exist_ok=True)
with tqdm(total=len(tids)) as pbar:
    for tid in tids:
        out_df_path = os.path.join(out_dir,tid+'.csv')
        if os.path.exists(out_df_path):
            pbar.update(1)
            continue
        temp_df = disc_df[disc_df['id']==tid]
        aug_discourse_text = de2en.translate(
                en2de.translate([row['discourse_text'] for i,row in temp_df.iterrows()],beam=1),
                beam=1,
                )
        #aug_discourse_text = []
        #for i,row in temp_df.iterrows():
        #    orig_text = row['discourse_text']
        #    aug_text = de2en.translate(en2de.translate(orig_text, beam=1), beam=1)
        #    aug_discourse_text.append(aug_text)
        temp_df['aug_discourse_text'] = aug_discourse_text
        temp_df.to_csv(out_df_path,index=False)
        pbar.update(1)
