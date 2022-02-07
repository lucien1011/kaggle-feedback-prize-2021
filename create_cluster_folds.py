import pandas as pd

from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

df = pd.read_csv('storage/train.csv')
dfx = pd.read_csv("storage/train_cluster.csv")
dfx = dfx[['id','cluster']]
dfx['cluster'] = dfx['cluster'].astype('category')

mskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
labels = [c for c in dfx.columns if c != "id"]
dfx_labels = dfx[labels]
dfx["kfold"] = -1

for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
    print(len(trn_), len(val_))
    dfx.loc[val_, "kfold"] = fold

df = df.merge(dfx[["id", "kfold"]], on="id", how="left")
print(df.kfold.value_counts())
print(df)
df.to_csv("storage/train_cluster_folds.csv", index=False)
