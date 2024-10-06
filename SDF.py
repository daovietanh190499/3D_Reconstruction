from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.optimize import least_squares

class PointOptimizer(nn.Module):
    def __init__(self, num_points, initial_points=None):
        super().__init__()
        self.num_points = num_points
        
        if initial_points is None:
            # Initialize random points if not provided
            initial_points = torch.randn(num_points, 3)
        
        # Set a scale factor to allow more flexible exploration of the point cloud
        self.scale_factor = 10  # Adjusted to a larger range
        scaled_points = initial_points / self.scale_factor
        
        # Create parameters
        self.points_params = nn.Parameter(scaled_points)
        
    def forward(self):
        # Directly return the point parameters, no sigmoid activation
        return torch.tanh(self.points_params)
    
    def get_points(self):
        # Rescale points to their original scale
        return self.points_params * self.scale_factor

def rodrigues_to_rotation_matrix(rodrigues):
    theta = torch.norm(rodrigues)
    if theta < 1e-6:
        return torch.eye(3, dtype=rodrigues.dtype, device=rodrigues.device)
    
    r = rodrigues / theta
    r_cross = torch.tensor([
        [0, -r[2], r[1]],
        [r[2], 0, -r[0]],
        [-r[1], r[0], 0]
    ], dtype=rodrigues.dtype, device=rodrigues.device)
    
    R = torch.eye(3, dtype=rodrigues.dtype, device=rodrigues.device) + \
        torch.sin(theta) * r_cross + \
        (1 - torch.cos(theta)) * torch.matmul(r_cross, r_cross)
    
    return R

def reproject_points(points_3d, K_params, R_rodrigues, t, eps=1e-6):
    fx, fy, cx, cy, skew = K_params
    K = torch.tensor([
        [fx, skew, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=points_3d.dtype, device=points_3d.device)
    
    R = rodrigues_to_rotation_matrix(R_rodrigues)
    
    points_3d_homogeneous = F.pad(points_3d, (0, 1), value=1)
    points_camera = torch.matmul(points_3d_homogeneous, torch.cat((R, t), dim=1).T)
    points_image = torch.matmul(points_camera, K.T)

    # z_coords = points_image[:, 2:3].clamp(min=eps)
    points_2d = points_image[:, :2] / points_image[:, 2:3]
    
    return points_2d

def reprojection_loss(points_3d, points_2d_gt, K_params, R_rodrigues, t):
    points_2d_pred = reproject_points(points_3d, K_params, R_rodrigues, t)

    point_3D = points_3d.detach().cpu().numpy().astype(np.float32)
    R_rodrigues_ = R_rodrigues.detach().cpu().numpy().astype(np.float32)
    t = t.detach().cpu().numpy().astype(np.float32)
    fx, fy, cx, cy, skew = K_params
    K = np.array([
        np.array([fx, skew, cx]),
        np.array([0, fy, cy]),
        np.array([0, 0, 1])
    ], dtype=np.float32)
    # R, _ = cv2.Rodrigues(R_rodrigues_)
    # print(type(R), type(t), type(K), type(point_3D), R.shape, t.shape, K.shape, point_3D.shape)

    reprojected_point, _ = cv2.projectPoints(point_3D, R_rodrigues_, t.ravel(), K, distCoeffs=None)
    reprojected_point = reprojected_point[:, 0, :]

    # points_2d_pred = torch.from_numpy(reprojected_point)

    # print(points_2d_pred[100], points_2d_gt[100], reprojected_point[100])
    loss = F.mse_loss(points_2d_pred, points_2d_gt)
    return loss

def reprojection_error(cameras_params, points_3d, points_2d_gt):
    K_params, R_rodrigues, t = cameras_params[:5], cameras_params[5:8], cameras_params[8:].reshape(3, 1)
    K_params = torch.from_numpy(K_params).float()
    R_rodrigues = torch.from_numpy(R_rodrigues).float()
    t = torch.from_numpy(t).float()

    points_2d_pred = reproject_points(points_3d, K_params, R_rodrigues, t)
    error = points_2d_pred - points_2d_gt
    error = error.view(-1).detach().numpy().ravel()
    return error

def least_square(points_3d, points_2d, K_params, R_rodrigues, t, ftol=1e-5):
    initial_params = np.concatenate([
        K_params.detach().cpu().numpy().ravel(),
        R_rodrigues.detach().cpu().numpy().ravel(),
        t.detach().cpu().numpy().ravel()
    ])

    res = least_squares(reprojection_error, initial_params, ftol=ftol, args=(points_3d, points_2d), method="lm")
    
    optimized_params = res.x[:5]
    R_rodrigues = res.x[5:8]
    t = res.x[8:]
    return optimized_params, R_rodrigues, t

# Training loop
keypoints = np.load("output/keypoints.npy", allow_pickle=True)
keypoints = [torch.from_numpy(keypoint).float() for keypoint in keypoints]

corresponding_2d_keypoints = np.load("output/corresponding_2d_keypoints.npy", allow_pickle=True)
num_points = len(corresponding_2d_keypoints)
print(f"Number of 3d point: {num_points}")
initial_points = torch.randn(num_points, 3)

images_to_points = np.load("output/corresponding_image_points.npy", allow_pickle=True)
images_to_points = [np.array([np.array(list(keypoint)) for keypoint in image]) for image in images_to_points]

image_size = np.load("output/img_size.npy", allow_pickle=True)

Ks = [torch.tensor([max(image), max(image), image[1] / 2, image[0] / 2, 0], dtype=torch.float32) for image in image_size]
Rs = [torch.randn(3, dtype=torch.float32) for _ in images_to_points]  # Randomized rotations
ts = [torch.tensor([[0], [0], [0]], dtype=torch.float32) for _ in images_to_points]

point_model = PointOptimizer(num_points, initial_points)
optimizer_params = torch.optim.Adam(point_model.parameters(), lr=0.001)  # Reduced learning rate

ftol = 1e-8

# Optimization loop
for epoch in tqdm(range(1000)):
    for j in tqdm(range(len(images_to_points))):
        optimizer_params.zero_grad()

        K_params, R_rodrigues, t = Ks[j], Rs[j], ts[j]

        points_2d_indices = images_to_points[j][:, 0].ravel()
        points_3d_indices = images_to_points[j][:, 1].ravel()

        points_2d_gt = keypoints[j][points_2d_indices]
        points_3d = point_model.get_points()[points_3d_indices]

        optimized_params = least_square(points_3d, points_2d_gt, K_params, R_rodrigues, t, ftol=ftol)
        
        optimized_K_params = torch.from_numpy(optimized_params[0]).float()
        optimized_R_rodrigues = torch.from_numpy(optimized_params[1]).float()
        optimized_t = torch.from_numpy(optimized_params[2]).float().reshape(3, 1)

        Ks[j] = optimized_K_params
        Rs[j] = optimized_R_rodrigues
        ts[j] = optimized_t

        loss = reprojection_loss(points_3d, points_2d_gt, optimized_K_params, optimized_R_rodrigues, optimized_t)

        loss.backward()
        optimizer_params.step()

    if epoch in [10, 25, 45]:
        ftol /= 100
        optimized_points = point_model.get_points()
        np.save(f'output/point_cloud_{str(epoch).zfill(4)}.npy', optimized_points.detach().cpu().numpy())

    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# Get the optimized points in the original scale
optimized_points = point_model.get_points()