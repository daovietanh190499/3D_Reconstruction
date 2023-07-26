import os
import cv2
import numpy as np
from tqdm import tqdm
import exifread

from lightglue import LightGlue
from lightglue.utils import rbd

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import torch
torch.set_grad_enabled(False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matcher = LightGlue(features='disk').eval().to(device)

img_path = "ystad_kloster/"
point_cloud = []
point_color = []
img_list = []
with open("output/img_list.txt") as f:
    img_list = f.readlines()
img_list = [l.strip() for l in img_list]
img_pairs = np.load("output/img_pairs.npy")
all_descriptors = np.load("output/all_descriptors.npy")
all_points = np.load("output/all_points.npy")
cameras = [None]*len(img_list)

class Camera:
    def __init__(self, id, img, kp, desc, match2d3d):
        self.id = id
        self.img = img
        self.kp = kp
        self.desc = desc 
        self.match2d3d = match2d3d
        self.Rt = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.reconstruct = False

    def setRt(self, R, t):
        self.Rt = np.hstack((R, t))
        self.reconstruct = True
    
    def getRt(self):
        return self.Rt[:3,:3], self.Rt[:3, 3]

    def getRelativeRt(self, cam2):
        return cam2.Rt[:3,:3].T.dot(self.Rt[:3,:3]), cam2.Rt[:3, :3].T.dot(self.Rt[:3, 3] - cam2.Rt[:3, 3])
    
    def getP(self, K):
        return np.matmul(K, self.Rt)
    
    def getPos(self):
        pts = np.array([[0,0,0]]).T
        pts = self.Rt[:3,:3].T.dot(pts)- self.Rt[:3,3][:,np.newaxis]
        return pts[:,0]
    
    def getFeature(self):
        return (self.kp, self.desc)

def get_camera_intrinsic_params(images_dir):
    K = []
    h, w, c = cv2.imread(images_dir + os.listdir(images_dir)[1]).shape
    img = open(images_dir + os.listdir(images_dir)[1], 'rb')
    exif = exifread.process_file(img, details=False)
    exif = exif if 'EXIF FocalLengthIn35mmFilm' in exif else {'EXIF FocalLengthIn35mmFilm': exifread.classes.IfdTag(True, 'focal', list, [29], 1, 32)}
    image_width, image_height = (w, h) if w > h else (h, w)
    focal_length = (exif['EXIF FocalLengthIn35mmFilm'].values[0]/35)*image_width
    K.append([focal_length, 0, image_width/2])
    K.append([0, focal_length, image_height/2])
    K.append([0, 0, 1])
    return {'width': image_width, 'height': image_height}, np.array(K, dtype=float)

def triangulate(cam1, cam2, idx0, idx1, K):
    points_3d = cv2.triangulatePoints(cam1.getP(K), cam2.getP(K), cam1.kp[idx0].T, cam2.kp[idx1].T)
    points_3d = points_3d / points_3d[3]
    points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
    points_3d = points_3d[:, 0, :]
    point2d_ind = idx1[np.where(cam1.match2d3d[idx0] ==  -1)]
    for w, i in enumerate(idx0):
        if cam1.match2d3d[i] == -1:
            point_cloud.append(points_3d[w])
            point_color.append(cam1.img[int(cam1.kp[i][1]), int(cam1.kp[i][0]), :])
            cam1.match2d3d[i] = len(point_cloud) - 1
        cam2.match2d3d[idx1[w]] = cam1.match2d3d[i]
    point3d_ind = cam2.match2d3d[point2d_ind]
    x = np.hstack((cv2.Rodrigues(cam2.getRt()[0])[0].ravel(), cam2.getRt()[1].ravel(), np.array(point_cloud)[point3d_ind].ravel()))
    A = ba_sparse(point3d_ind, x)
    res = least_squares(calculate_reprojection_error, x, jac_sparsity=A, x_scale='jac', ftol=1e-8, args=(K, cam2.kp[point2d_ind]))
    R, t, point_3D = cv2.Rodrigues(res.x[:3])[0], res.x[3:6], res.x[6:].reshape((len(point3d_ind), 3))
    for i, j in enumerate(point3d_ind): point_cloud[j] = point_3D[i]
    cam2.setRt(R, t.reshape((3,1)))

def to_ply(img_dir, point_cloud, colors, subfix = "_sparse.ply"):
    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    print(out_colors.shape, out_points.shape)
    verts = np.hstack([out_points, out_colors])
    mean = np.mean(verts[:, :3], axis=0)
    temp = verts[:, :3] - mean
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    indx = np.where(dist < np.mean(dist) + 300)
    verts = verts[indx]
    ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar blue
		property uchar green
		property uchar red
		end_header
		'''
    print(img_dir + '/Point_Cloud/' + img_dir.split('/')[-2] + subfix)
    if not os.path.exists(img_dir + '/Point_Cloud/'):
        os.makedirs(img_dir + '/Point_Cloud/')
    with open(img_dir + '/Point_Cloud/' + img_dir.split('/')[-2] + subfix, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

def ba_sparse(point3d_ind, x):
    A = lil_matrix((len(point3d_ind)*2, len(x)), dtype=int)
    A[np.arange(len(point3d_ind)*2), :6] = 1
    for i in range(3):
        A[np.arange(len(point3d_ind))*2, 6 + np.arange(len(point3d_ind))*3 + i] = 1
        A[np.arange(len(point3d_ind))*2 + 1, 6 + np.arange(len(point3d_ind))*3 + i] = 1
    return A

def calculate_reprojection_error(x, K, point_2D):
    R, t, point_3D = x[:3], x[3:6], x[6:].reshape((len(point_2D), 3))
    reprojected_point, _ = cv2.projectPoints(point_3D, R, t, K, distCoeffs=None)
    reprojected_point = reprojected_point[:, 0, :]
    return (point_2D - reprojected_point).ravel()

exif, K = get_camera_intrinsic_params(img_path)

for index, (i, j) in tqdm(enumerate(img_pairs)):
    update = False
    if not cameras[i]:
        update = True
        img = cv2.imread(img_path + img_list[i])
        if img.shape[1] < img.shape[0]:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cameras[i] = Camera(img_list[i], img, all_points[i], all_descriptors[i], np.ones((len(all_points[i]),), dtype='int32')*-1)
    
    if not cameras[j]:
        update = True
        img = cv2.imread(img_path + img_list[j])
        if img.shape[1] < img.shape[0]:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cameras[j] = Camera(img_list[j], img, all_points[j], all_descriptors[j], np.ones((len(all_points[j]),), dtype='int32')*-1)
    
    feats0 = {"keypoints": torch.tensor([all_points[i]]), "descriptors": torch.tensor([all_descriptors[i]]), 'image_size': torch.tensor([[img.shape[1],  img.shape[0]]])}
    feats1 = {"keypoints": torch.tensor([all_points[j]]), "descriptors": torch.tensor([all_descriptors[j]]), 'image_size': torch.tensor([[img.shape[1],  img.shape[0]]])}
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    pts0_, pts1_, idx0, idx1 = m_kpts0.detach().cpu().numpy(), m_kpts1.detach().cpu().numpy(), matches[..., 0].detach().cpu().numpy(), matches[..., 1].detach().cpu().numpy()

    if update:
        E, mask = cv2.findEssentialMat(pts0_, pts1_, K, method=cv2.RANSAC, prob=0.999, threshold=1)
        idx0, idx1 = idx0[mask.ravel() == 1], idx1[mask.ravel() == 1]
        if index != 0:
            match = np.int32(np.where(cameras[i].match2d3d[idx0] != -1)[0])
            if len(match) < 8: continue
            ret, rvecs, t, inliers = cv2.solvePnPRansac(np.float32(point_cloud)[cameras[i].match2d3d[idx0[match]]], cameras[j].kp[idx1[match]], K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)
            R, _ = cv2.Rodrigues(rvecs)
        else:
            _, R, t, _ = cv2.recoverPose(E, pts0_[mask.ravel() == 1], pts1_[mask.ravel() == 1], K)
        cameras[j].setRt(R, t)

    triangulate(cameras[i], cameras[j], idx0, idx1, K)
    

to_ply(img_path, np.array(point_cloud), np.array(point_color))
to_ply(img_path, np.array([cam.getPos() if cam else np.array([0,0,0]) for cam in cameras]), np.ones_like(np.array([cam.getPos() for cam in cameras]))*255, '_campos.ply')