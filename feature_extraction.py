import os
import cv2
import torch
from lightglue import DISK
from tqdm import tqdm
import numpy as np

torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
extractor = DISK(max_num_keypoints=2048).eval().to(device)  # load the extractor

img_dir = './ystad_kloster/'
images = sorted( filter( lambda x: os.path.isfile(os.path.join(img_dir, x)), os.listdir(img_dir) ) )

text_file = open("output/img_list.txt", "wt")

all_feats = np.array([None]*len(images), dtype=object)
all_points = np.array([None]*len(images), dtype=object)
all_colors = np.array([None]*len(images), dtype=object)
img_size = np.array([None]*len(images), dtype=object)

for i in tqdm(range(len(images))):
    if images[i].split('.')[-1] in ['JPG', 'jpg', 'PNG', 'png', 'RAW', 'raw', 'TIF', 'tif']:
        text_file.write(images[i] + '\n')
        img = cv2.imread(img_dir + images[i])
        img_ = img.transpose((2, 0, 1))
        img_tensor = torch.tensor(img_ / 255., dtype=torch.float)
        feats = extractor.extract(img_tensor.to(device))
        feats_ = feats['descriptors'].cpu().detach().numpy()
        points = feats['keypoints'].cpu().detach().numpy()
        size = feats['image_size'].cpu().detach().numpy()
        colors = [img[int(point[1]), int(point[0]), :] for point in points[0]]

        points_ = np.array([np.array([pt[0] - size[0][0]/2, -(pt[1] - size[0][1]/2)]) for pt in points[0]])

        img_size[i] = size[0]
        all_points[i] = points_
        all_feats[i] = feats_[0]
        all_colors[i] = colors

np.save('output/all_points.npy', all_points)
del all_points

np.save('output/img_size.npy', img_size)
del img_size

np.save('output/all_colors.npy', all_colors)
del all_colors

np.save('output/all_descriptors.npy', all_feats)
del all_feats

text_file.close()