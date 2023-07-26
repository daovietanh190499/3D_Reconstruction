import joblib
from scipy.cluster.vq import vq
import numpy as np
from numpy.linalg import norm

all_descriptors = np.load("output/all_descriptors.npy")
k, codebook = joblib.load("output/bow_codebook.plk")
visual_words = []

for desciptors in all_descriptors:
    img_visual_words, distance = vq(desciptors, codebook)
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
for i in range(N):
    a = tfidf[i]
    b = tfidf
    cosine_similarity = np.dot(a, b.T)/(norm(a) * norm(b, axis=1))
    top_k = 3
    idx = np.argsort(-cosine_similarity)[1:top_k]
    all_idx.append(idx)

connection = [None]*N

for i in range(N):
    for id in all_idx[i]:
        if not connection[i]:
            connection[i] = []
        if not connection[id]:
            connection[id] = []
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

