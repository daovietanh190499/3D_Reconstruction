from tqdm import tqdm
import torch
import numpy as np
from lightglue import LightGlue
from lightglue.utils import rbd

import joblib
from scipy.cluster.vq import vq
from numpy.linalg import norm

all_descriptors = np.load("output/all_descriptors.npy", allow_pickle=True)
all_points = np.load("output/all_points.npy", allow_pickle=True)
img_size = np.load("output/img_size.npy", allow_pickle=True)
k, codebook = joblib.load("output/bow_codebook.plk")

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matcher = LightGlue(features='disk').eval().to(device)

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
top_k = 10
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

print(connection)

max = 0
start = 0
# for i, c in enumerate(connection):
#     if len(c) > max:
#         max = len(c)
#         start = i

point3d_index = 0
all_matches = []
all_points3d = [None]*all_points.shape[0]
queue = [(start, start)]
visited = [False]*N
visited[start] = True
i = 0

with tqdm(total=N) as pbar:
    while True:
        for id in connection[queue[i][1]]:
            if not visited[id]:
                reference_id = queue[i][1]

                for id_ in connection[id]:
                    if id_ == queue[i][1]:
                        break
                    if visited[id_]:
                        reference_id = id_
                        break
                
                feats0 = {
                    "keypoints": torch.tensor(np.array([all_points[reference_id]], dtype=float), dtype=torch.float).to(device), 
                    "descriptors": torch.tensor(np.array([all_descriptors[reference_id]], dtype=float), dtype=torch.float).to(device), 
                    'image_size': torch.tensor(np.array([img_size[reference_id]], dtype=float), dtype=torch.float).to(device)
                }
                feats1 = {
                    "keypoints": torch.tensor(np.array([all_points[id]], dtype=float), dtype=torch.float).to(device), 
                    "descriptors": torch.tensor(np.array([all_descriptors[id]], dtype=float), dtype=torch.float).to(device), 
                    'image_size': torch.tensor(np.array([img_size[id]], dtype=float), dtype=torch.float).to(device)
                }

                matches01 = matcher({'image0': feats0, 'image1': feats1})
                feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
                kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
                m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
                idx0, idx1 = matches[..., 0].detach().cpu().numpy(), matches[..., 1].detach().cpu().numpy()

                interlaced_points = 0
                
                for p1, p2 in zip(idx0, idx1):
                    if not all_points3d[reference_id]:
                        all_points3d[reference_id] = [-1]*all_points[reference_id].shape[0]
                    if not all_points3d[id]:
                        all_points3d[id] = [-1]*all_points[id].shape[0]
                    if all_points3d[reference_id][p1] == -1 and all_points3d[id][p2] == -1:
                        continue
                    elif all_points3d[reference_id][p1] != -1:
                        interlaced_points += 1
                    elif all_points3d[id][p1] != -1:
                        interlaced_points += 1

                if  len(idx0) >= 500 and (i == start or (i != start and interlaced_points/len(idx0) >= 0.3)): 
                    point3d_indexes = []
                    for p1, p2 in zip(idx0, idx1):
                        if all_points3d[reference_id][p1] == -1 and all_points3d[id][p2] == -1:
                            all_points3d[reference_id][p1] = point3d_index
                            all_points3d[id][p2] = point3d_index
                            point3d_index += 1
                        elif all_points3d[reference_id][p1] != -1:
                            all_points3d[id][p2] = all_points3d[reference_id][p1]
                        elif all_points3d[id][p1] != -1:
                            all_points3d[reference_id][p2] = all_points3d[id][p1]
                        
                        point3d_indexes.append(all_points3d[reference_id][p1])

                    all_matches.append([idx0, idx1, np.array(point3d_indexes)])
                    queue.append((reference_id, id))
                    visited[id] = True
                else:
                    continue

        i += 1
        pbar.update(1)
        if i >= len(queue):
            break

print(queue, len(queue))
np.save('output/img_pairs.npy', queue[1:])
np.save('output/all_matches.npy', np.array(all_matches, dtype=object))
