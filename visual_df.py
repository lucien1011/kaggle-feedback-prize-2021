import pandas as pd
from termcolor import colored

color_map = {
        'Lead': 'red',
        'Position': 'red',
        'Claim': 'cyan',
        'Evidence': 'yellow',
        'Rebuttal': 'cyan',
        'Counterclaim': 'cyan',
        'Concluding Statement': 'red',
        'O': 'white',
        }
header = '='*100
subheader = '-'*100

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--true_csv',action='store',default='storage/train.csv')
    parser.add_argument('--pred_csv',action='store',default='storage/output/220203_baseline+stance+lstm_cvfold0_longformer-large/NERPredictionString/submission_df.csv')
    parser.add_argument('--id',action='store',default='')
    return parser.parse_args()

def visualize_id(tid,train_df,dname='discourse_type',hname=''):
    with open('storage/train/'+tid+'.txt','r') as file: data = file.read()
    words = data.split()
    print(header)
    print(hname,tid)
    print(header)
    items = []
    for i, row in train_df[train_df['id'] == tid].iterrows():
        preds = row['predictionstring'].split()
        start = int(preds[0])
        end = int(preds[-1])
        disc = row[dname]
        items.append(
            (start,disc,colored(' '.join(words[start:end+1]),color_map[disc]))
        )
    items.sort()
    for start,disc,text in items:
        print("{:20s} | ".format(disc),text)

def run(args):
    true_df = pd.read_csv(args.true_csv)
    pred_df = pd.read_csv(args.pred_csv)
    
    tid = pred_df.sample().id.item() if not args.id else args.id
    visualize_id(tid,true_df,hname='True')
    visualize_id(tid,pred_df,'class',hname='Pred')

if __name__ == '__main__':

    args = parse_arguments()
    run(args)
