from scipy.cluster.vq import kmeans, vq
import joblib
import numpy as np
from numpy.linalg import norm
import os

k = 200
iters = 1

all_descriptors = np.load("output/descriptors.npy", allow_pickle=True)

print(all_descriptors.shape)

all_descriptors_ = all_descriptors[0].astype("float")

for i, descriptors in enumerate(all_descriptors):
    if i != 0:
        all_descriptors_ = np.vstack((descriptors.astype("float"), all_descriptors_))

print(all_descriptors_.shape)
print("Build Codebook")

codebook, variance = kmeans(all_descriptors_, k, iters)

joblib.dump((k, codebook), "output/bow_codebook.plk", compress=3)