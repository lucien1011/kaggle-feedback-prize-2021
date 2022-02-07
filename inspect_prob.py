import numpy as np 
import random

from baseline.Preprocess import id_target_map

input_path = 'storage/output/220128_baseline+cvfold0_longformer/NERInfer/probs.npy'

def print_segment(labels,maxprobs):
    subheader = '-'*100
    n = len(labels)
    i = 1
    prev = labels[0]
    temp = [maxprobs[0]]
    while i < n:
        if prev == labels[i]:
            temp.append(maxprobs[i])
        else:
            print(subheader)
            print(prev)
            print(subheader)
            print(temp)
            temp = [maxprobs[i]]
            prev = labels[i]
        i += 1

probs = np.load(input_path,allow_pickle=True)

idx = random.randint(0,len(probs))
labels = probs[idx].argmax(axis=-1)
maxprobs = probs[idx].max(axis=-1)

print_segment(
    [id_target_map[e] for i,e in enumerate(labels[0])],
    [maxprobs[0][i] for i,e in enumerate(labels[0])],
    )

