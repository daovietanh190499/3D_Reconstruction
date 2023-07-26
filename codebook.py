from scipy.cluster.vq import kmeans
import joblib
import numpy as np

k = 200
iters = 1

all_descriptors = np.load("output/all_descriptors.npy")

print(all_descriptors.shape)

all_descriptors_ = all_descriptors[0]

for i, descriptors in enumerate(all_descriptors):
    if i != 0:
        all_descriptors_ = np.vstack((descriptors, all_descriptors_))

print(all_descriptors_.shape)

codebook, variance = kmeans(all_descriptors_, k, iters)

joblib.dump((k, codebook), "output/bow_codebook.plk", compress=3)
