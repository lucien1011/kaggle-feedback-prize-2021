import pandas as pd

from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',action='store',default=5)
    parser.add_argument('--csv',action='store',default='storage/train.csv')
    parser.add_argument('--out',action='store',default='storage/train_folds.csv')
    parser.add_argument('--cluster',action='store',default='')
    return parser.parse_args()

def run(args):

    df = pd.read_csv(args.csv)
    
    dfx = pd.get_dummies(df, columns=["discourse_type"]).groupby(["id"], as_index=False).sum()
    cols = [c for c in dfx.columns if c.startswith("discourse_type_") or c == "id" and c != "discourse_type_num"]
    if args.cluster: cols.append(args.cluster)
    dfx = dfx[cols]
    
    mskf = MultilabelStratifiedKFold(n_splits=args.n, shuffle=True, random_state=42)
    labels = [c for c in dfx.columns if c != "id"]
    dfx_labels = dfx[labels]
    dfx["kfold"] = -1
    
    for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
        dfx.loc[val_, "kfold"] = fold
    
    df = df.merge(dfx[["id", "kfold"]], on="id", how="left")
    df.to_csv(args.out, index=False)

if __name__ == '__main__':

    args = parse_arguments()
    run(args)
