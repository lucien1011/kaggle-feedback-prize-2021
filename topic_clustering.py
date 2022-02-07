import pandas as pd, os
from tqdm import tqdm
import numpy as np

train_names, train_texts = [], []
for f in tqdm(list(os.listdir('storage/train'))):
    train_names.append(f.replace('.txt', ''))
    train_texts.append(open('storage/train/' + f, 'r').read())
train_text_df = pd.DataFrame({'id': train_names, 'text': train_texts})
train_text_df.head()

from sklearn.feature_extraction.text import TfidfVectorizer
print('tfidf')
tfidf = TfidfVectorizer(stop_words='english', binary=True, max_features=25000)
text_embeddings = tfidf.fit_transform( train_text_df.text ).toarray()

from umap import UMAP
print('umap')
umap = UMAP()
embed_2d = umap.fit_transform(text_embeddings)

from sklearn.cluster import KMeans
print('kmeans')
kmeans = KMeans(n_clusters=15)
kmeans.fit(embed_2d)
train_text_df['cluster'] = kmeans.labels_

train_text_df.to_csv('storage/train_cluster.csv',index=False)

import matplotlib.pyplot as plt

centers = kmeans.cluster_centers_

plt.figure(figsize=(10,10))
plt.scatter(embed_2d[:,0], embed_2d[:,1], s=1, c=kmeans.labels_)
plt.title('UMAP Plot of Train Text using Tfidf features\nRAPIDS Discovers the 15 essay topics!',size=16)

for k in range(len(centers)):
    mm = np.mean( text_embeddings[train_text_df.cluster.values==k],axis=0 )
    ii = np.argmax(mm)
    top_word = tfidf.vocabulary_.iloc[ii]
    plt.text(centers[k,0]-1,centers[k,1]+0.75,f'{k+1}-{top_word}',size=16)

plt.show()
