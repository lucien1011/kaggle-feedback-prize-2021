import argparse
import pandas as pd

from comp import evaluate_score_from_df

header = '='*100

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_df',action='store')
    parser.add_argument('--true_df',action='store',default='storage/train_folds.csv')
    parser.add_argument('--fold',action='store',default=0)
    return parser.parse_args()

def count(first_disc_name,next_disc_name,df,flagname='discourse_type'):
    count = 0
    ids = df.id.unique().tolist()
    for id in ids:
        temp_df = df[df.id==id]
        prev_disc_type,prev_predstr = None,None
        rows = [(int(row['predictionstring'].split()[0]),int(row['predictionstring'].split()[-1]),row[flagname]) for i,row in temp_df.iterrows()]
        rows.sort()
        for start,end,disc in rows:
            if disc == next_disc_name:
                if prev_disc_type == first_disc_name and abs(prev_end-start) < 5:
                    count += 1
            prev_disc_type = disc
            prev_end = end
    return count

def count_df(df,name):
    names = df[name].unique().tolist()
    names.sort()
    for first_disc_name in names:
        print('-'*100)
        print(first_disc_name)
        print('-'*100)
        for next_disc_name in names:
            print(next_disc_name,count(first_disc_name,next_disc_name,df,name))

def count_discourse_type_df(df,name):
    discs = df[name].unique().tolist()
    discs.sort()
    print(header)
    for disc in discs:
        temp_df = df[df[name]==disc]
        ndisc = len(temp_df[temp_df[name]==disc])
        nword = temp_df['predictionstring'].apply(lambda x: len(x.split())).sum()
        print("{:20s} | {:10d} | {:10d} | {:4.2f}".format(
            disc,ndisc,nword,nword/ndisc
            ))

def run(args):
    true_df = pd.read_csv(args.true_df)
    true_df = true_df[true_df['kfold']==args.fold]    
    pred_df = pd.read_csv(args.pred_df)

    #count_df(pred_df,'class')
    #count_df(true_df,'discourse_type')
    
    count_discourse_type_df(pred_df,'class')
    count_discourse_type_df(true_df,'discourse_type')
    
    evaluate_score_from_df(true_df,pred_df)

if __name__ == '__main__':

    args = parse_arguments()
    run(args)
