import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import cv2
import os
from matplotlib.image import imread

from tqdm import tqdm


class AppearanceModel(nn.Module):
    def __init__(self, embedding_dim_pos=20, hidden_dim=128):
        super(AppearanceModel, self).__init__()

        self.block1 = nn.Sequential(nn.Linear(embedding_dim_pos * 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )

        self.block3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), )
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 75), )

        self.embedding_dim_pos = embedding_dim_pos
        self.relu = nn.ReLU()

        self.bandwidth = nn.Parameter(torch.zeros((1, 25)))
        self.p = nn.Parameter(torch.randn((25, 2)))

    @staticmethod
    def positional_encoding(x, L):
        out = torch.empty(x.shape[0], x.shape[1] * 2 * L, device=x.device)
        for i in range(x.shape[1]):
            for j in range(L):
                out[:, i * (2 * L) + 2 * j] = torch.sin(2 ** j * x[:, i])
                out[:, i * (2 * L) + 2 * j + 1] = torch.cos(2 ** j * x[:, i])
        return out

    def forward(self, o):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos // 2)
        h = self.block1(emb_x)
        h = self.block3(h)
        k = self.block4(h).reshape(o.shape[0], 75)
        return k


class SceneHelper:
    def __init__(self, data_path, point_cloud_path, camera_extrinsics_path, max_resolution):
        point_cloud = np.load(point_cloud_path)
        camera_extrinsics = np.load(camera_extrinsics_path)

        camera_intrinsics = torch.ones(1, device=device)*2378.98305085
        camera_extrinsics = torch.from_numpy(np.array([np.hstack((cv2.Rodrigues(cam[:,:3])[0].ravel(), cam[:, 3].ravel())) for cam in camera_extrinsics])).float().to(device)

        self.data_path = data_path
        self.point_cloud = self.filter_point_cloud(point_cloud)
        self.camera_extrinsics = camera_extrinsics
        self.camera_intrinsics = camera_intrinsics
        self.images, self.num_img = self.load_images()
        self.min_bound, self.max_bound, self.resolution = (-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), (max_resolution, max_resolution, max_resolution)
        self.min_bound, self.max_bound, self.resolution = self.get_grid_resolution(max_resolution)
        self.bounds = (self.min_bound, self.max_bound)

        x = np.linspace(0, 1, 10)*((self.max_bound[0] - self.min_bound[0])) + self.min_bound[0]
        y = np.linspace(0, 1, 10)*((self.max_bound[1] - self.min_bound[1])) + self.min_bound[1]
        z = np.linspace(0, 1, 10)*((self.max_bound[2] - self.min_bound[2])) + self.min_bound[2]
        grid = np.meshgrid(x, y, z, indexing='ij')
        self.rectangular = np.stack(grid, axis=-1).reshape(-1, 3)
    
    def filter_point_cloud(self, verts):
        verts = verts * 200
        mean = np.mean(verts, axis=0)
        temp = verts - mean
        dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        return verts/200

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

    def get_grid_resolution(self, max_resolution):
        min_coords = np.min(self.point_cloud, axis=0)
        max_coords = np.max(self.point_cloud, axis=0)
        
        min_coords = (min_coords * 1.5).astype(int)
        max_coords = (max_coords * 1.5).astype(int)
        
        grid_size = max_coords - min_coords
        
        size_box = np.max(grid_size) / max_resolution
        grid_resolution = np.ceil(grid_size / size_box).astype(int)
        
        grid_size = grid_resolution * size_box
        
        return tuple(min_coords), tuple(max_coords), grid_resolution

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

        pts = pts_uniform
        z_vals = z_uniform

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
        # sh_values = sh_values.reshape(*pts.shape[:-1], 75)

        return sdf_values, sh_values, pts, z_vals, valid

class SDFGrid(nn.Module):
    def __init__(self, resolution, min_bound, max_bound, device):
        super().__init__()
        self.sampler = GradientBasedSampler(num_samples=160, num_importance=32)
        self.resolution = resolution
        self.min_bound = torch.tensor(min_bound).to(device)
        self.max_bound = torch.tensor(max_bound).to(device)
        self.grid = nn.Parameter(torch.ones(1, 27 + 1, *resolution) / 100)
        # self.grid = nn.Parameter(torch.ones(1, 1, *resolution) / 100)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        # self.appearance_model = AppearanceModel()
    
    def get_sdf(self, points):
        sdf = torch.zeros((points.shape[0]), device=points.device)
        mask = self.test_points(points)

        points = points[mask]

        # Normalize points to [-1, 1] range
        normalized_points = (points - self.min_bound) / (self.max_bound - self.min_bound) * 2 - 1
        
        # Reshape for grid_sample
        normalized_points = normalized_points.view(1, 1, 1, -1, 3)
        
        # Trilinear interpolation
        sdf_values = F.grid_sample(self.grid[:, 0:1, ...], normalized_points, 
                                   align_corners=True, mode='bilinear')

        sdf_values = sdf_values.permute([0, 2, 3, 4, 1]).squeeze(0).squeeze(0).squeeze(0)

        sdf[mask] = sdf_values.squeeze(1)

        return sdf

    def get_sdf_sh(self, points):
        sh = torch.zeros((points.shape[0], 27), device=points.device)
        # sh = torch.zeros((points.shape[0], 75), device=points.device)
        sdf = torch.zeros((points.shape[0]), device=points.device)
        mask = self.test_points(points)

        old_point = points
        points = points[mask]

        # Normalize points to [-1, 1] range
        normalized_points = (points - self.min_bound) / (self.max_bound - self.min_bound) * 2 - 1
        
        # Reshape for grid_sample
        normalized_points = normalized_points.view(1, 1, 1, -1, 3)
        
        # Trilinear interpolation
        sdf_values = F.grid_sample(self.grid[:, 0:1, ...], normalized_points, 
                                   align_corners=True, mode='bilinear')

        sh_values = F.grid_sample(self.grid[:, 1:, ...], normalized_points, 
                            align_corners=True, mode='bilinear')

        sdf_values = sdf_values.permute([0, 2, 3, 4, 1]).squeeze(0).squeeze(0).squeeze(0)
        sh_values = sh_values.permute([0, 2, 3, 4, 1]).squeeze(0).squeeze(0).squeeze(0)

        # sh_values = self.appearance_model(points)

        sdf[mask] = sdf_values.squeeze(1)
        sh[mask] = sh_values
        # sh[mask] = sh_values

        # print(torch.sum(self.grid[:, :1, ...] != 0.01))
        # print(torch.sum(torch.isnan(points)), torch.sum(torch.isnan(old_point)))
        # print(torch.sum(torch.isnan(sdf)), torch.sum(torch.isnan(normalized_points)), torch.sum(torch.isnan(self.grid[:, :1, ...])), torch.sum(torch.isnan(self.grid[:, 1:, ...])), torch.isnan(self.grid[:, :1, ...]).nonzero())
        # print(self.alpha, self.beta)
        
        return sdf, sh

    def get_sdf_gradient(self, points):
        points.requires_grad_(True)
        sdf = self.get_sdf(points)
        grad = torch.autograd.grad(sdf.sum(), points, create_graph=True)[0]
        return grad

    def to_cartesian(self, theta_phi):
        return torch.stack([torch.sin(theta_phi[:, 0]) * torch.cos(theta_phi[:, 1]),
                            torch.sin(theta_phi[:, 0]) * torch.sin(theta_phi[:, 1]),
                            torch.cos(theta_phi[:, 0])], axis=1)

    def eval_spherical_gaussian_function(self, k, d):
        k = k.reshape(-1, 25, 3)
        c = (k * torch.exp(
            (self.appearance_model.bandwidth.unsqueeze(-1) * self.to_cartesian(self.appearance_model.p).unsqueeze(0) * d.unsqueeze(1)))).sum(1)
        return torch.sigmoid(c)

    def eval_spherical_function(self, k, d):
        k = k.reshape(-1, 3, 9)
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
        return torch.nn.functional.relu(sdf)
        # return 1 / (1 + torch.exp(-self.alpha * (sdf + self.beta)))

    def test_camera(self, rays_o, rays_d):
        try:
            sdf_values, sh_values, pts, z_vals, valid = self.sampler(self, rays_o, rays_d)
            return True
        except:
            return False
    
    def test_points(self, x):
        return (x[:, 0] >= self.min_bound[0]) & (x[:, 1] >= self.min_bound[1]) & (x[:, 2] >= self.min_bound[2]) & \
        (x[:, 0] <= self.max_bound[0]) & (x[:, 1] <= self.max_bound[1]) & (x[:, 2] <= self.max_bound[2])
    
    def forward(self, rays_o, rays_d):
        sdf_values, sh_values, pts, z_vals, valid = self.sampler(self, rays_o, rays_d)
        
        rays_o = rays_o[valid]
        rays_d = rays_d[valid]

        rays_d = rays_d.expand(z_vals.shape[1], rays_d.shape[0], 3).transpose(0, 1)
        colors = self.eval_spherical_function(sh_values, rays_d.reshape(-1, 3)).reshape(rays_d.shape)
        sigma = self.sdf_to_density(sdf_values.reshape(-1)).reshape(rays_d.shape[:-1])
        delta = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.tensor([1e10], device=colors.device).expand(rays_o.shape[0], 1)), -1)

        alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
        weights = self.compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
        c = (weights * colors).sum(dim=1)  # Pixel values
        weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background\
        return c + 1 - weight_sum.unsqueeze(-1), pts, valid


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nb_epochs = int(1e0)
    batch_size = 1024

    scene_helper = SceneHelper('ystad_kloster', "output/points_3d.npy", "output/cameras_extrinsic.npy", 250)

    sdf_model = SDFGrid(scene_helper.resolution, scene_helper.min_bound, scene_helper.max_bound, device).to(device)
    optimizer = torch.optim.Adam(sdf_model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)

    training_dataset = torch.from_numpy(np.load('/home/coder/psrnet/nerf_datasets/training_data.pkl', allow_pickle=True))
    data_loader = DataLoader(training_dataset, batch_size=2048, shuffle=True)
    total_loss = []
    for _ in range(nb_epochs):
        training_loss = []
        for batch in tqdm(data_loader):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)

            regenerated_px_values, pts, valid = sdf_model(ray_origins, ray_directions)
            loss = torch.nn.functional.mse_loss(ground_truth_px_values[valid], regenerated_px_values)
            
            # print("=====", regenerated_px_values[-4:], (ground_truth_px_values[valid])[-4:])
            # print("losss", loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            if(len(training_loss) % 170 == 0):
                if len(total_loss) > 0:
                    print(training_loss)
                total_loss.append(training_loss)
                training_loss = []
        scheduler.step()

    # available_img = []
    # for i in tqdm(range(scene_helper.num_img)):
    #     torch.cuda.empty_cache()
    #     rays_o, rays_d, gt_px_values = scene_helper.sample_batch(batch_size, img_index=i, sample_all=False)
    #     is_valid = sdf_model.test_camera(rays_o, rays_d)
    #     if is_valid:
    #         available_img.append(i)
    
    # print(available_img)
    # test_pts = np.array([[0, 0, 0]])

    # total_loss = []
    # for epoch in tqdm(range(nb_epochs)):
    #     training_loss = []
    #     for index, i in tqdm(enumerate(available_img)):
    #         torch.cuda.empty_cache()

    #         rays_o, rays_d, gt_px_values = scene_helper.sample_batch(batch_size, img_index=i, sample_all=False)
    #         pred_px_values, pts, valid = sdf_model(rays_o, rays_d)

    #         loss = torch.nn.functional.mse_loss(gt_px_values[valid].float()/255, pred_px_values.float())

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         training_loss.append(loss.item())

    #         if index > 20 and index <= 25:
    #             print("SHAPE", pts.shape)
    #             test_pts = np.concatenate((test_pts, pts.reshape(-1, 3).detach().cpu().numpy()), axis=0)
    #             np.save('output/test_points.npy', np.concatenate((scene_helper.rectangular, test_pts)))
    #     if epoch > 0:
    #         print([0 if l1 - l2 > 0 else 1 for l1, l2 in zip(training_loss,total_loss[-1])])
    #     total_loss.append(training_loss)
    #     scheduler.step()

