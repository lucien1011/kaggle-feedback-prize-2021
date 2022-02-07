import pickle
import random

ts1 = pickle.load(open('storage/output/220126_baseline_preprocess_bi_mskfold/NERPreprocessKFold/train_samples_fold0.p','rb'))
ts2 = pickle.load(open('storage/output/220205_baseline+aug_preprocess_bi_mskfold/NERPreprocessKFold/train_samples_fold0.p','rb'))

print(len(ts1),len(ts2))
ts = ts1 + random.sample(ts2,int(0.5*len(ts2)))
print(len(ts))

pickle.dump(ts,open('storage/output/220205_baseline+aug_preprocess_bi_mskfold/NERPreprocessKFold/all_train_samples_fold0.p','wb'))
