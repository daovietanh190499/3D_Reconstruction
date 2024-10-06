

from tqdm import tqdm
import numpy as np
import cv2

from scipy.optimize import least_squares

def least_square(points_3d, points_2d, K_params, R_rodrigues, t, ftol=1e-5):
    fx, fy, cx, cy, skew = K_params
    K = np.array([
        [fx, skew, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    _, R_rodrigues, t, _ = cv2.solvePnPRansac(points_3d, points_2d, K, cv2.SOLVEPNP_ITERATIVE)

    initial_params = np.concatenate([
        K_params.detach().cpu().numpy().ravel(),
        R_rodrigues.ravel(),
        t.ravel()
    ])

    res = least_squares(reprojection_error, initial_params, ftol=ftol, args=(points_3d, points_2d), method="lm")

    optimized_params = res.x

    return optimized_params

# Training loop
keypoints = np.load("output/keypoints.npy", allow_pickle=True)

corresponding_2d_keypoints = np.load("output/corresponding_2d_keypoints.npy", allow_pickle=True)
points_3ds = np.zeros((len(corresponding_2d_keypoints),3))

images_to_points = np.load("output/corresponding_image_points.npy", allow_pickle=True)
images_to_points = [np.array([np.array(list(keypoint)) for keypoint in image]) for image in images_to_points]

image_size = np.load("output/img_size.npy",allow_pickle=True)

Ks = [np.array([max(image), max(image), image[1]/2, image[0]/2, 0], dtype=np.float32) for image in image_size]
Rs = [np.array([0.1, 0.2, 0.3], dtype=np.float32) for _ in images_to_points]
ts = [np.array([[0], [0], [0]], dtype=np.float32) for _ in images_to_points]

ftol=1e-8

for j in tqdm(range(len(images_to_points))):

    K_params, R_rodrigues, t = Ks[j], Rs[j], ts[j]

    points_2d_indices = images_to_points[j][:, 0].ravel()
    points_3d_indices = images_to_points[j][:, 1].ravel()

    points_2d_gt = keypoints[j][points_2d_indices]
    points_3d = points_3ds[points_3d_indices]

    

    optimized_params = least_square(points_3d, points_2d_gt, K_params, R_rodrigues, t, ftol=ftol)
    
    optimized_K_params = optimized_params[:5]
    optimized_R_rodrigues = optimized_params[5:8]
    optimized_t = optimized_params[8:]

    Ks[j] = optimized_K_params
    Rs[j] = optimized_R_rodrigues
    ts[j] = optimized_t
