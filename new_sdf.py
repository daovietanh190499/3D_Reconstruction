import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import os
from matplotlib.image import imread

from tqdm import tqdm

class SceneHelper:
    def __init__(self, data_path, point_cloud_path, camera_extrinsics_path):
        point_cloud = np.load(point_cloud_path)
        camera_extrinsics = np.load(camera_extrinsics_path)

        camera_intrinsics = torch.ones(1, device=device)*2378.98305085
        camera_extrinsics = torch.from_numpy(np.array([np.hstack((cv2.Rodrigues(cam[:,:3])[0].ravel(), cam[:, 3].ravel())) for cam in camera_extrinsics])).float().to(device)

        self.data_path = data_path
        self.point_cloud = point_cloud
        self.camera_extrinsics = camera_extrinsics
        self.camera_intrinsics = camera_intrinsics
        self.images, self.num_img = self.load_images()
        self.min_bound, self.max_bound, self.resolution = self.get_grid_resolution()

    def load_images(self):
        img_list = []
        images = []
        image = None
        with open("output/reconstructed_img.txt") as f:
            img_list = f.readlines()
        img_list = [l.strip() for l in img_list]
        for i in range(len(img_list)):
            image_path = img_list[i]
            image = imread(os.path.join(self.data_path, image_path))
            images.append(image)
        return images, len(img_list)

    def get_grid_resolution(self):
        minx, miny, minz, maxx, maxy, maxz = \
            np.min(self.point_cloud[:,0]), np.min(self.point_cloud[:,1]), np.min(self.point_cloud[:,2]), \
            np.max(self.point_cloud[:,0]), np.max(self.point_cloud[:,1]), np.max(self.point_cloud[:,2])

        minx, miny, minz, maxx, maxy, maxz = \
            int(minx*1.5), int(miny*1.5), int(minz*1.5), int(maxx*1.5), int(maxy*1.5), int(maxz*1.5)

        x_length = maxx - minx
        y_length = maxy - miny
        z_length = maxz - minz
        grid_size = np.array([x_length, y_length, z_length])

        arg_sort = np.argsort(grid_size)

        resolution = 255
        grid_resolution = np.array([255, 255, 255])

        size_box = grid_size[arg_sort[2]] / 255

        grid_resolution[arg_sort[0]] = np.ceil(grid_size[arg_sort[0]]/size_box)
        grid_resolution[arg_sort[1]] = np.ceil(grid_size[arg_sort[1]]/size_box)

        grid_size[arg_sort[0]] = grid_resolution[arg_sort[0]]*size_box
        grid_size[arg_sort[1]] = grid_resolution[arg_sort[1]]*size_box

        return (minx, miny, minz), (maxx, maxy, maxz), grid_resolution

    def sample_batch(self, batch_size, img_index=0, sample_all=False):
        image = self.images[img_index]
        H, W = image.shape[:2]

        if sample_all:
            image_indices = (torch.zeros(W * H) + img_index).type(torch.long)
            u, v = np.meshgrid(np.linspace(0, W - 1, W, dtype=int), np.linspace(0, H - 1, H, dtype=int))
            u = torch.from_numpy(u.reshape(-1)).to(self.camera_intrinsics.device)
            v = torch.from_numpy(v.reshape(-1)).to(self.camera_intrinsics.device)
        else:
            image_indices = (torch.zeros(batch_size) + img_index).type(torch.long)  # Sample random images
            u = torch.randint(W, (batch_size,), device=self.camera_intrinsics.device)  # Sample random pixels
            v = torch.randint(H, (batch_size,), device=self.camera_intrinsics.device)

        focal = self.camera_intrinsics[0]
        t = self.camera_extrinsics[img_index, :3]
        r = self.camera_extrinsics[img_index, -3:]

        # Creating the c2w matrix, Section 4.1 from the paper
        phi_skew = torch.stack([torch.cat([torch.zeros(1, device=r.device), -r[2:3], r[1:2]]),
                                torch.cat([r[2:3], torch.zeros(1, device=r.device), -r[0:1]]),
                                torch.cat([-r[1:2], r[0:1], torch.zeros(1, device=r.device)])], dim=0)
        alpha = r.norm() + 1e-15
        R = torch.eye(3, device=r.device) + (torch.sin(alpha) / alpha) * phi_skew + (
                (1 - torch.cos(alpha)) / alpha ** 2) * (phi_skew @ phi_skew)
        c2w = torch.cat([R, t.unsqueeze(1)], dim=1)
        c2w = torch.cat([c2w, torch.tensor([[0., 0., 0., 1.]], device=c2w.device)], dim=0)

        rays_d_cam = torch.cat([((u.to(self.camera_intrinsics.device) - .5 * W) / focal).unsqueeze(-1),
                                (-(v.to(self.camera_intrinsics.device) - .5 * H) / focal).unsqueeze(-1),
                                - torch.ones_like(u).unsqueeze(-1)], dim=-1)
        rays_d_world = torch.matmul(c2w[:3, :3].view(1, 3, 3), rays_d_cam.unsqueeze(2)).squeeze(2)
        rays_o_world = c2w[:3, 3].view(1, 3).expand_as(rays_d_world).to(self.camera_intrinsics.device)

        gt_px_values = torch.from_numpy(image[v.cpu(), u.cpu()]).to(self.camera_intrinsics.device)

        return rays_o_world, F.normalize(rays_d_world, p=2, dim=1), gt_px_values

class SDFGrid(nn.Module):
    def __init__(self, resolution, min_bound, max_bound, device):
        super().__init__()
        self.resolution = resolution
        self.min_bound = torch.tensor(min_bound).to(device)
        self.max_bound = torch.tensor(max_bound).to(device)
        self.grid = nn.Parameter(torch.ones(1, 27 + 1, *resolution) / 100)
    
    def get_sdf(self, points):
        # Normalize points to [-1, 1] range
        normalized_points = (points - self.min_bound) / (self.max_bound - self.min_bound) * 2 - 1
        
        # Reshape for grid_sample
        normalized_points = normalized_points.view(1, -1, 1, 1, 3)
        
        # Trilinear interpolation
        sdf_values = F.grid_sample(self.grid[:, 0:1, ...], normalized_points, 
                                   align_corners=True, mode='bilinear')

        return sdf_values.view(points.shape[:-1])

    def get_sdf_sh(self, points):
        # Normalize points to [-1, 1] range
        normalized_points = (points - self.min_bound) / (self.max_bound - self.min_bound) * 2 - 1
        
        # Reshape for grid_sample
        normalized_points = normalized_points.view(1, -1, 1, 1, 3)
        
        # Trilinear interpolation
        sdf_values = F.grid_sample(self.grid[:, 0:1, ...], normalized_points, 
                                   align_corners=True, mode='bilinear')

        sh_values = F.grid_sample(self.grid[:, 1:, ...], normalized_points, 
                            align_corners=True, mode='bilinear')
        
        return sdf_values.view(points.shape[:-1]), sh_values.view(points.shape[0], 27)

    def get_sdf_gradient(self, points):
        points.requires_grad_(True)
        sdf = self.get_sdf(points)
        grad = torch.autograd.grad(sdf.sum(), points, create_graph=True)[0]
        return grad

class GradientBasedSampler:
    def __init__(self, num_samples=64, num_importance=32, perturb=True):
        self.num_samples = num_samples
        self.num_importance = num_importance
        self.perturb = perturb

    def ray_aabb_intersection(self, rays_o, rays_d, min_bound, max_bound):
        inv_d = 1.0 / rays_d
        t_min = (min_bound - rays_o) * inv_d
        t_max = (max_bound - rays_o) * inv_d
        
        t_near = torch.max(torch.min(t_min, t_max), dim=-1).values
        t_far = torch.min(torch.max(t_min, t_max), dim=-1).values

        t_near = torch.max(t_near, torch.tensor(0.0, device=t_near.device))
        
        valid = t_far > t_near  # Ensures t_far > t_near and t_near >= 0
        return t_near, t_far, valid
    
    def sample_uniform(self, rays_o, rays_d, t_near, t_far):
        # Uniform sampling along the ray between t_near and t_far
        t = torch.linspace(0, 1, self.num_samples, device=rays_o.device)
        z = t_near[:, None] * (1 - t) + t_far[:, None] * t
        
        if self.perturb:
            mids = 0.5 * (z[:, 1:] + z[:, :-1])
            upper = torch.cat([mids, z[:, -1:]], dim=-1)
            lower = torch.cat([z[:, :1], mids], dim=-1)
            t_rand = torch.rand_like(z)
            z = lower + (upper - lower) * t_rand

        pts = rays_o[:, None, :] + rays_d[:, None, :] * z[:, :, None]
        return pts, z
    
    def sample_importance(self, rays_o, rays_d, z_uniform, weights):
        z_vals = self.sample_pdf(z_uniform, weights, self.num_importance, det=False)
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
        return pts, z_vals

    # Helper function for importance sampling
    def sample_pdf(self, bins, weights, N_samples, det=False):
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)
        bins = torch.cat([bins[...,:1], bins], -1)
        
        if det:
            u = torch.linspace(0., 1., steps=N_samples).to(weights.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(weights.device)

        u = u.contiguous()
        
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)
        
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
        # bins_g = torch.gather(bins.unsqueeze(1).expand(inds_g.shape[0], inds_g.shape[1], bins.shape[-1]), 2, inds_g)
        
        denom = (cdf_g[...,1] - cdf_g[...,0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[...,0])/denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])
        
        return samples
    
    def __call__(self, sdf_grid, rays_o, rays_d):
        # Compute t_near and t_far from ray-box intersection
        t_near, t_far, valid = self.ray_aabb_intersection(rays_o, rays_d, sdf_grid.min_bound, sdf_grid.max_bound)
        
        # Only sample rays that intersect the grid
        if not valid.any():
            raise ValueError("No valid rays intersect the grid.")
        
        rays_o = rays_o[valid]
        rays_d = rays_d[valid]
        t_near = t_near[valid]
        t_far = t_far[valid]

        # Initial uniform sampling based on t_near and t_far
        pts_uniform, z_uniform = self.sample_uniform(rays_o, rays_d, t_near, t_far)

        # Compute SDF and gradients
        sdf_uniform = sdf_grid.get_sdf(pts_uniform.reshape(-1, 3)).reshape(pts_uniform.shape[:-1])
        gradients = sdf_grid.get_sdf_gradient(pts_uniform.reshape(-1, 3)).reshape(pts_uniform.shape)

        # Compute weights based on SDF gradient magnitude
        gradient_mag = gradients.norm(dim=-1)
        weights = F.softmax(gradient_mag, dim=-1)

        # Importance sampling
        pts_importance, z_importance = self.sample_importance(rays_o, rays_d, z_uniform, weights)

        # Combine uniform and importance samples
        pts = torch.cat([pts_uniform, pts_importance], dim=1)
        z_vals = torch.cat([z_uniform, z_importance], dim=-1)

        # Sort samples by depth
        z_vals, indices = torch.sort(z_vals, dim=-1)
        pts = torch.gather(pts, 1, indices.unsqueeze(-1).expand_as(pts))

        # Do not calculate grad of these variables
        pts = pts.detach()
        z_vals = z_vals.detach()
        valid = valid.detach()

        # Compute SDF for all points
        sdf_values, sh_values = sdf_grid.get_sdf_sh(pts.reshape(-1, 3)) #.reshape(pts.shape[:-1])
        sdf_values = sdf_values.reshape(pts.shape[:-1])
        sh_values = sh_values.reshape(*pts.shape[:-1], 27)

        return sdf_values, sh_values, pts, z_vals, valid


class SDFToNeRF(nn.Module):
    def __init__(self, resolution=(64, 64, 64), bounds=((-1, -1, -1), (1, 1, 1)), device="cuda"):
        super().__init__()
        self.sdf_grid = SDFGrid(resolution, bounds[0], bounds[1], device).to(device)
        self.sampler = GradientBasedSampler(num_samples=160, num_importance=32)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def eval_spherical_function(self, k, d):
        x, y, z = d[..., 0:1], d[..., 1:2], d[..., 2:3]

        # Modified from https://github.com/google/spherical-harmonics/blob/master/sh/spherical_harmonics.cc
        return 0.282095 * k[..., 0] + \
            - 0.488603 * y * k[..., 1] + 0.488603 * z * k[..., 2] - 0.488603 * x * k[..., 3] + \
            (1.092548 * x * y * k[..., 4] - 1.092548 * y * z * k[..., 5] + 0.315392 * (2.0 * z * z - x * x - y * y) * k[
                ..., 6] + -1.092548 * x * z * k[..., 7] + 0.546274 * (x * x - y * y) * k[..., 8])

    def compute_accumulated_transmittance(self, alphas):
        accumulated_transmittance = torch.cumprod(alphas, 1)
        return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                        accumulated_transmittance[:, :-1]), dim=-1)
    
    def sdf_to_density(self, sdf):
        return 1 / (1 + torch.exp(-self.alpha * (sdf + self.beta)))
    
    def forward(self, rays_o, rays_d):
        sdf_values, sh_values, pts, z_vals, valid = self.sampler(self.sdf_grid, rays_o, rays_d)
        
        rays_o = rays_o[valid]
        rays_d = rays_d[valid]

        rays_d = rays_d.expand(z_vals.shape[1], rays_d.shape[0], 3).transpose(0, 1)
        colors = self.eval_spherical_function(sh_values.reshape(-1, 3, 9), rays_d.reshape(-1, 3)).reshape(rays_d.shape)
        sigma = self.sdf_to_density(sdf_values.reshape(-1)).reshape(sdf_values.shape)
        delta = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.tensor([1e10], device=colors.device).expand(rays_o.shape[0], 1)), -1)

        alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
        weights = self.compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
        c = (weights * colors).sum(dim=1)  # Pixel values
        weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background\
        return c + 1 - weight_sum.unsqueeze(-1)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nb_epochs = int(1e3)
    batch_size = 512

    scene_helper = SceneHelper('ystad_kloster', "output/points_3d.npy", "output/cameras_extrinsic.npy")

    sdf_model = SDFToNeRF(scene_helper.resolution, (scene_helper.min_bound, scene_helper.max_bound), device)
    sdf_model.to(device)

    optimizer = torch.optim.Adam(sdf_model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)

    training_loss = []
    for epoch in range(nb_epochs):
        for i in tqdm(range(scene_helper.num_img)):
            torch.cuda.empty_cache()

            rays_o, rays_d, gt_px_values = scene_helper.sample_batch(batch_size, img_index=i, sample_all=False)
            pred_px_values = sdf_model(rays_o, rays_d)

            loss = torch.nn.functional.mse_loss(gt_px_values.float(), pred_px_values.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
        scheduler.step()