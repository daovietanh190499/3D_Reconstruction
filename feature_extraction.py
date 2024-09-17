import os
import cv2
from tqdm import tqdm
import numpy as np

# import torch
# from lightglue import DISK
# torch.set_grad_enabled(False)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# extractor = DISK(max_num_keypoints=2048).eval().to(device)  # load the extractor

# Create ORB object
orb = cv2.ORB_create(nfeatures=2048)  # Set max number of features to 2048

img_dir = '.\\ystad_kloster\\'
images = sorted(filter(lambda x: os.path.isfile(os.path.join(img_dir, x)), os.listdir(img_dir)))

text_file = open("output/img_list.txt", "wt")

all_feats = []
all_points = []
all_colors = []
img_size = []

for i in tqdm(range(len(images))):
    if images[i].split('.')[-1].lower() in ['jpg', 'png', 'raw', 'tif']:
        text_file.write(images[i] + '\n')
        img = cv2.imread(img_dir + images[i])

        # img_ = img.transpose((2, 0, 1))
        # img_tensor = torch.tensor(img_ / 255., dtype=torch.float)
        # feats = extractor.extract(img_tensor.to(device))
        # feats_ = feats['descriptors'].cpu().detach().numpy()
        # points = feats['keypoints'].cpu().detach().numpy()
        # size = feats['image_size'].cpu().detach().numpy()
        # colors = [img[int(point[1]), int(point[0]), :] for point in points[0]]
        
        # img_size.append(size[0])
        # all_points.append(points[0])
        # all_feats.append(feats_[0])
        # all_colors.append(colors)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        points = np.array([kp.pt for kp in keypoints])
        colors = [img[int(point[1]), int(point[0])] for point in points]
        
        img_size.append(img.shape[:2])
        all_points.append(points)
        all_feats.append(descriptors)
        all_colors.append(colors)

all_descriptors = np.array(all_feats, dtype=object)
all_points = np.array(all_points, dtype=object)
img_size = np.array(img_size, dtype=object)
all_colors = np.array(all_colors, dtype=object)

text_file.close()
np.save('output/all_descriptors.npy', all_descriptors)
np.save('output/all_points.npy', all_points)
np.save('output/all_colors.npy', all_colors)
np.save('output/img_size.npy', img_size)
