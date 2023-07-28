from tqdm import tqdm
import torch
import numpy as np
from lightglue import LightGlue
from lightglue.utils import rbd

img_pairs = np.load("output/img_pairs.npy")
all_descriptors = np.load("output/all_descriptors.npy", allow_pickle=True)
all_points = np.load("output/all_points.npy", allow_pickle=True)
img_size = np.load("output/img_size.npy", allow_pickle=True)

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matcher = LightGlue(features='disk').eval().to(device)

point3d_index = 0

all_matches = []

all_points3d = [None]*all_points.shape[0]

for index, (i, j) in enumerate(tqdm(img_pairs)):
    feats0 = {
        "keypoints": torch.tensor(np.array([all_points[i]], dtype=float), dtype=torch.float).to(device), 
        "descriptors": torch.tensor(np.array([all_descriptors[i]], dtype=float), dtype=torch.float).to(device), 
        'image_size': torch.tensor(np.array([img_size[i]], dtype=float), dtype=torch.float).to(device)
    }
    feats1 = {
        "keypoints": torch.tensor(np.array([all_points[j]], dtype=float), dtype=torch.float).to(device), 
        "descriptors": torch.tensor(np.array([all_descriptors[j]], dtype=float), dtype=torch.float).to(device), 
        'image_size': torch.tensor(np.array([img_size[j]], dtype=float), dtype=torch.float).to(device)
    }
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    idx0, idx1 = matches[..., 0].detach().cpu().numpy(), matches[..., 1].detach().cpu().numpy()

    point3d_indexes = []
    for p1, p2 in zip(idx0, idx1):
        if not all_points3d[i]:
            all_points3d[i] = [-1]*all_points[i].shape[0]
        if not all_points3d[j]:
            all_points3d[j] = [-1]*all_points[j].shape[0]
        if all_points3d[i][p1] == -1 and all_points3d[j][p2] == -1:
            all_points3d[i][p1] = point3d_index
            all_points3d[j][p2] = point3d_index
            point3d_index += 1
        elif all_points3d[i][p1] != -1:
            all_points3d[j][p2] = all_points3d[i][p1]
        elif all_points3d[j][p1] != -1:
            all_points3d[i][p2] = all_points3d[j][p1]
        
        point3d_indexes.append(all_points3d[i][p1])

    all_matches.append([idx0, idx1, np.array(point3d_indexes)])

np.save('output/all_matches.npy', np.array(all_matches, dtype=object))

    
    