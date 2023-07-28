from scipy.cluster.vq import kmeans, vq
import joblib
import numpy as np
from numpy.linalg import norm
import os

k = 200
iters = 1

all_descriptors = np.load("output/all_descriptors.npy", allow_pickle=True)

print(all_descriptors.shape)

if not os.path.exists("output/bow_codebook.plk"):
    all_descriptors_ = all_descriptors[0].astype("float")

    for i, descriptors in enumerate(all_descriptors):
        if i != 0:
            all_descriptors_ = np.vstack((descriptors.astype("float"), all_descriptors_))

    print(all_descriptors_.shape)
    print("Build Codebook")

    codebook, variance = kmeans(all_descriptors_, k, iters)

    joblib.dump((k, codebook), "output/bow_codebook.plk", compress=3)
    
k, codebook = joblib.load("output/bow_codebook.plk")

print("Build Pairs")

visual_words = []

for desciptors in all_descriptors:
    img_visual_words, distance = vq(desciptors.astype("float"), codebook)
    visual_words.append(img_visual_words)

frequency_vectors = []
for img_visual_words in visual_words:
    img_frequency_vector = np.zeros(k)
    for word in img_visual_words:
        img_frequency_vector[word] += 1
    frequency_vectors.append(img_frequency_vector)

frequency_vectors = np.stack(frequency_vectors)

print(frequency_vectors.shape)

N = frequency_vectors.shape[0]

df = np.sum(frequency_vectors > 0, axis = 0 )

idf = np.log(N/ df)

tfidf = frequency_vectors * idf

all_idx = []
all_score = []
top_k = 3
for i in range(N):
    a = tfidf[i]
    b = tfidf
    cosine_similarity = np.dot(a, b.T)/(norm(a) * norm(b, axis=1))
    idx = np.argsort(-cosine_similarity)[1:top_k]
    score = np.sort(-cosine_similarity)[1:top_k]
    all_idx.append(idx)
    all_score.append(score)

connection = [None]*N

for i in range(N):
    for j, id in enumerate(all_idx[i]):
        if not connection[i]:
            connection[i] = []
        if not connection[id]:
            connection[id] = []
        if -all_score[i][j] > 0.75:
            if not id in connection[i]:
                connection[i].append(id)
            if not i in connection[id]:
                connection[id].append(i)

start = 0
queue = [(start, start)]
visited = [False]*N
visited[0] = True
i = 0

while True:
    for id in connection[queue[i][1]]:
        if not visited[id]:
            queue.append((queue[i][1], id))
            visited[id] = True
    i += 1
    if i >= len(queue):
        break
print(queue, len(queue))

np.save('output/img_pairs.npy', queue[1:])

