import os
import cv2
import numpy as np
from tqdm import tqdm

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

img_list = []
with open("output/img_list.txt") as f:
    img_list = f.readlines()
img_list = [l.strip() for l in img_list]

img_pairs = np.load("output/img_pairs.npy", allow_pickle=True)
all_points = np.load("output/all_points.npy", allow_pickle=True)
all_colors = np.load("output/all_colors.npy", allow_pickle=True)
all_matches = np.load("output/all_matches.npy", allow_pickle=True)
img_size = np.load("output/img_size.npy", allow_pickle=True)
all_point3ds = [[None]*(np.max(np.hstack(all_matches[:,2])) + 1),[None]*(np.max(np.hstack(all_matches[:,2])) + 1)]
cameras = [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]) for _ in range(len(img_list))]
# focal_length = [2378.98]*len(img_list)
focal_length = 3340.8
focal_length = 2378.98

def triangulate(i, j, pts0, pts1, idx0, idx1, idx3d, K):
    points_3d = cv2.triangulatePoints(np.matmul(K, cameras[i]), np.matmul(K, cameras[j]), pts0.T, pts1.T)
    points_3d = points_3d / points_3d[3]
    points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
    points_3d = points_3d[:, 0, :]

    for w, f in enumerate(idx3d):
        all_point3ds[0][f] = points_3d[w]
        all_point3ds[1][f] = all_colors[i][idx0[w]]

    x = np.hstack((np.array([K[0][0]]), cv2.Rodrigues(cameras[j][:3, :3])[0].ravel(), cameras[j][:3, 3].ravel(), np.stack(np.array(all_point3ds[0], dtype=object)[idx3d]).ravel()))
    A = ba_sparse(idx3d, x)
    res = least_squares(calculate_reprojection_error, x, jac_sparsity=A, x_scale='jac', ftol=1e-8, args=(K, pts1))
    focal_length, R, t, point_3D = res.x[0], cv2.Rodrigues(res.x[1:4])[0], res.x[4:7], res.x[7:].reshape(len(idx3d), 3)

    for w, f in enumerate(idx3d): 
        all_point3ds[0][f] = point_3D[w]

    cameras[j] = np.hstack((R, t.reshape((3,1))))

    return focal_length

def to_ply(img_dir, point_cloud, colors):
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
    with open(img_dir, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

def ba_sparse(point3d_ind, x):
    A = lil_matrix((len(point3d_ind)*2, len(x)), dtype=int)
    A[np.arange(len(point3d_ind)*2), :7] = 1
    for i in range(3):
        A[np.arange(len(point3d_ind))*2, 7 + np.arange(len(point3d_ind))*3 + i] = 1
        A[np.arange(len(point3d_ind))*2 + 1, 7 + np.arange(len(point3d_ind))*3 + i] = 1
    return A

def calculate_reprojection_error(x, K, point_2D):
    focal_length, R, t, point_3D = x[0], x[1:4], x[4:7], x[7:].reshape((len(point_2D), 3))
    K[0,0] = focal_length
    K[1,1] = focal_length
    reprojected_point, _ = cv2.projectPoints(point_3D, R, t, K, distCoeffs=None)
    reprojected_point = reprojected_point[:, 0, :]
    return (point_2D - reprojected_point).ravel()

for index, (i, j) in enumerate(tqdm(img_pairs)):
    idx0, idx1, idx3d = all_matches[index][0], all_matches[index][1], all_matches[index][2] 
    pts0, pts1, point3ds = all_points[i][idx0].astype('float64'), all_points[j][idx1].astype('float64'), np.array(all_point3ds[0], dtype=object)[idx3d]
    K =  np.array([[focal_length, 0, img_size[i, 0]/2], [0, focal_length, img_size[i, 1]/2], [0, 0, 1]])

    # print(pts0, pts1)

    E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1)
    # print(E, mask, pts0, pts1, K, i, j, img_list[i], img_list[j])
    idx0, idx1, idx3d = idx0[mask.ravel() == 1], idx1[mask.ravel() == 1], idx3d[mask.ravel() == 1]
    pts0, pts1, point3ds = pts0[mask.ravel() == 1], pts1[mask.ravel() == 1], point3ds[mask.ravel() == 1]

    mask_ = np.array([pt is None for pt in point3ds])

    if index != 0:
        ret, rvecs, t, inliers = cv2.solvePnPRansac(np.stack(point3ds[mask_ == 0]), pts1[mask_ == 0], K, np.zeros((5, 1), dtype=np.float32), cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(rvecs)
    else:
        _, R, t, _ = cv2.recoverPose(E, pts0, pts1, K)
    
    cameras[j] = np.hstack((R, t))

    focal_length = triangulate(i, j, pts0[mask_ == 1], pts1[mask_ == 1], idx0[mask_ == 1], idx1[mask_ == 1], idx3d[mask_ == 1], K)

mask = np.array([pt is None for pt in all_point3ds[0]])

np.save("output/cameras_extrinsic.npy", np.array(cameras))
to_ply("output/result.ply", np.stack(np.array(all_point3ds[0], dtype=object)[mask == 0]).astype(float), np.stack(np.array(all_point3ds[1], dtype=object)[mask == 0]).astype(float))
to_ply("output/campos.ply", np.array([(cam[:3,:3].T.dot(np.array([[0,0,0]]).T) - cam[:3,3][:,np.newaxis])[:,0] for cam in cameras]), np.array([ np.array([1, 1, 1]) for cam in cameras])*255)