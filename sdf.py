import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.image import imread
from tqdm import tqdm

def compute_polynomial_coefficients(sdf_values, corners, origin, direction):
    # Unpack the sdf values
    s000, s100, s010, s110, s001, s101, s011, s111 = sdf_values
    p000, p100, p010, p110, p001, p101, p011, p111 = corners
    px000, py000, pz000 = p000[0][0], p000[0][1], p000[0][2]
    px100 = p100[0][0]
    py010 = p010[0][1]
    pz001 = p001[0][2]
    ox, oy, oz = origin
    dx, dy, dz = direction

    ox = (ox - px000) / (px100 - px000)
    oy = (oy - py000) / (py010 - py000)
    oz = (oz - pz000) / (pz001 - pz000)
    dx = dx / (px100 - px000)
    dy = dy / (py010 - py000)
    dz = dz / (pz001 - pz000)

    # Calculate constants ki based on equation (3)
    k0 = s000
    k1 = s100 - s000
    k2 = s010 - s000
    k3 = s110 - s010 - k1
    k4 = k0 - s001
    a = s101 - s001
    k5 = k1 - a
    k6 = k2 - (s011 - s001)
    k7 = k3 - (s111 - s011 - a)

    # Calculate intermediate terms (m0 to m5)
    m0 = ox * oy
    m1 = dx * dy
    m2 = ox * dy + oy * dx
    m3 = k5 * oz - k1
    m4 = k6 * oz - k2
    m5 = k7 * oz - k3

    # Calculate c0, c1, c2, c3 based on equation (6)
    c0 = (k4 * oz - k0) + ox * m3 + oy * m4 + m0 * m5
    c1 = dx * m3 + dy * m4 + m2 * m5 + dz * (k4 + k5 * ox + k6 * oy + k7 * m0)
    c2 = m1 * m5 + dz * (k5 * dx + k6 * dy + k7 * m2)
    c3 = k7 * m1 * dz

    return c3, c2, c1, c0

def compute_quadratic_real_roots(c2, c1, c0):
    if c2 == 0 or c2 <= 10e-5:
        if c1 == 0 or c1 <= 10e-20:
            return None
        else:
            return -c0/c1

    # Compute the discriminant Δ = c1^2 - 4 * c2 * c0
    discriminant = c1 ** 2 - 4 * c2 * c0

    if discriminant > 0:
        # Check for two real roots (Δ > 0), one real root (Δ == 0), or no real roots (Δ < 0)
        sqrt_discriminant = torch.sqrt(discriminant)
        # Compute the roots
        root1 = (-c1 + sqrt_discriminant) / (2 * c2)
        root2 = (-c1 - sqrt_discriminant) / (2 * c2)
        
        if root1 > root2:
            return root2
        else:
            return root1
    else:
        return None

def compute_real_roots(c3, c2, c1, c0):
    if c3 == 0 or c3 <= 10e-5:
        return compute_quadratic_real_roots(c2, c1, c0)

    # Compute a, b, and c based on the provided coefficients
    a = c2 / c3
    b = c1 / c3
    c = c0 / c3

    # Compute Q and R
    Q = (a ** 2 - 3 * b) / 9
    R = (2 * a ** 3 - 9 * a * b + 27 * c) / 54
    
    roots = []
    
    # Case 1: Three real roots when R^2 < Q^3
    if R ** 2 < Q ** 3:
        theta = torch.acos(R / torch.sqrt(Q ** 3))
        sqrt_Q = torch.sqrt(Q)

        t1 = -2 * sqrt_Q * torch.cos(theta / 3) - a / 3
        t2 = -2 * sqrt_Q * torch.cos((theta - 2 * torch.pi) / 3) - a / 3
        t3 = -2 * sqrt_Q * torch.cos((theta + 2 * torch.pi) / 3) - a / 3

        # Sort roots in increasing order
        roots = t1
    
    # Case 2: One real root when R^2 >= Q^3
    else:
        A = -torch.sign(R) * (abs(R) + torch.sqrt(R ** 2 - Q ** 3)) ** (1/3)
        B = 0 if A == 0 else Q / A
        
        t1 = A + B - a / 3
        roots = t1

    return roots


def lerp(u, a, b):
    return a + u * (b - a)

def compute_normal(sdf, corners, point):
    p000, p100, p010, p110, p001, p101, p011, p111 = corners
    px000, py000, pz000 = p000[0][0], p000[0][1], p000[0][2]
    px100 = p100[0][0]
    py010 = p010[0][1]
    pz001 = p001[0][2]
    x, y, z = point

    # sdf = [s000, s100, s010, s110, s001, s101, s011, s111]
    x = (x - px000) / (px100 - px000)
    y = (y - py000) / (py010 - py000)
    z = (z - pz000) / (pz001 - pz000)

    # Partial derivative w.r.t x
    y0 = lerp(y, sdf[1] - sdf[0], sdf[3] - sdf[2])
    y1 = lerp(y, sdf[5] - sdf[4], sdf[7] - sdf[6])
    df_dx = lerp(z, y0, y1)

    # Partial derivative w.r.t y
    x0 = lerp(x, sdf[2] - sdf[0], sdf[3] - sdf[1])
    x1 = lerp(x, sdf[6] - sdf[4], sdf[7] - sdf[5])
    df_dy = lerp(z, x0, x1)

    # Partial derivative w.r.t z
    x0 = lerp(x, sdf[4] - sdf[0], sdf[5] - sdf[1])
    x1 = lerp(x, sdf[6] - sdf[2], sdf[7] - sdf[3])
    df_dz = lerp(y, x0, x1)

    # Normal vector
    normal = torch.cat((df_dx, df_dy, df_dz), dim=0)
    return normal

class SDFGrid(nn.Module):
    def __init__(self, minx, miny, minz, maxx, maxy, maxz, nx, ny, nz, device):
        super(SDFGrid, self).__init__()
        self.device = device
        # Define the grid bounds
        self.min_bound = torch.tensor([minx, miny, minz]).to(device)
        self.max_bound = torch.tensor([maxx, maxy, maxz]).to(device)
        
        # Define the learnable SDF grid as a parameter (nx * ny * nz grid)
        self.sdf_values = nn.Parameter(torch.randn(nx, ny, nz))  # Random initialization
        
        # Grid size (number of voxels)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        # Voxel size (distance between grid points)
        self.voxel_size = (self.max_bound - self.min_bound) / torch.tensor([nx - 1, ny - 1, nz - 1]).to(device)

    def query_voxels(self, points):
        # Batch process of voxel queries
        idx = ((points - self.min_bound) / self.voxel_size).long()

        # Ensure points are within bounds
        valid_mask = torch.all((idx >= 0) & (idx < torch.tensor([self.nx, self.ny, self.nz]).to(self.device) - 1), dim=-1)

        if not valid_mask.any():
            return None  # No valid points

        # Get 8 corner indices for each point
        corners = [
            idx,
            idx + torch.tensor([1, 0, 0]).to(self.device),
            idx + torch.tensor([0, 1, 0]).to(self.device),
            idx + torch.tensor([1, 1, 0]).to(self.device),
            idx + torch.tensor([0, 0, 1]).to(self.device),
            idx + torch.tensor([1, 0, 1]).to(self.device),
            idx + torch.tensor([0, 1, 1]).to(self.device),
            idx + torch.tensor([1, 1, 1]).to(self.device),
        ]

        # Get the 8 SDF values and their positions for all rays
        sdf_vals = [self.sdf_values[c[..., 0], c[..., 1], c[..., 2]] for c in corners]
        positions = [self.min_bound + c * self.voxel_size for c in corners]

        return sdf_vals, positions, valid_mask

    def rays_box_intersection(self, origins, directions):
        inv_dir = 1.0 / directions

        t1 = (self.min_bound - origins) * inv_dir
        t2 = (self.max_bound - origins) * inv_dir
        zeros = torch.zeros_like(t1)

        t_near = torch.max(torch.min(t1, t2), dim=-1)[0]
        t_far = torch.min(torch.max(t1, t2), dim=-1)[0]

        t_near = t_near*(t_near > 0)

        valid_mask = t_near <= t_far
        t_near = torch.where(valid_mask, t_near, torch.tensor(float('inf')))
        t_far = torch.where(valid_mask, t_far, torch.tensor(-float('inf')))

        return t_near, t_far, valid_mask

    def forward(self, origins, directions):
        import time
        start = time.time()
        batch_size = origins.size(0)
        t_near, t_far, valid_mask = self.rays_box_intersection(origins, directions)
        
        ray_data = [None for _ in range(batch_size)]

        print("generate_t_infor", time.time() - start)

        for i in range(batch_size):
            if not valid_mask[i]:
                continue

            current_pos = origins[i] + t_near[i] * directions[i]
            idx = torch.floor((current_pos - self.min_bound) / self.voxel_size).long()
            
            step = torch.sign(directions[i]).long()
            t_delta = self.voxel_size / directions[i].abs()
            
            next_voxel_boundary = (idx + torch.maximum(step, torch.tensor(0))) * self.voxel_size + self.min_bound
            t_next = (next_voxel_boundary - origins[i]) / directions[i]

            while torch.all((idx >= 0) & (idx < torch.tensor([self.nx, self.ny, self.nz]).to(self.device) - 1)):
                query_result = self.query_voxels(current_pos.unsqueeze(0))
                if query_result is not None:
                    sdf_vals, positions, _ = query_result
                    coeffs = compute_polynomial_coefficients(sdf_vals, positions, origins[i], directions[i])
                    t = compute_real_roots(*coeffs)

                    if t is not None:
                        intersection = origins[i] + t * directions[i]
                        normal = compute_normal(sdf_vals, positions, intersection)
                        
                        if ray_data[i] is None: 
                            ray_data[i] = [intersection, normal]
                            break

                mask = (t_next < t_far[i]) & (t_next == torch.min(t_next))
                if not mask.any():
                    break

                idx += step * mask
                t_min = torch.min(t_next)
                current_pos = origins[i] + t_min * directions[i]
                t_next += t_delta * mask
        
        print("stop_t_infor", time.time() - start)

        return ray_data


class AppearanceModel(nn.Module):
    def __init__(self, embedding_dim_pos=20, embedding_dim_direction=8, embedding_dim_normal=8, hidden_dim=128):
        super(AppearanceModel, self).__init__()

        self.block1 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + 3 + embedding_dim_pos * 6 + 3 + embedding_dim_normal * 6 + 3, hidden_dim),  nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),)
        self.block2 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(), )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.embedding_dim_normal = embedding_dim_normal

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d, n):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction)
        emb_n = self.positional_encoding(n, self.embedding_dim_normal)

        c = self.block2(self.block1(torch.cat((emb_x, emb_d, emb_n), dim=1)))
        return c


def sample_batch(camera_extrinsics, camera_intrinsics, batch_size, H, W, img_index=0, sample_all=False):
    if sample_all:  
        image_indices = (torch.zeros(W * H) + img_index).type(torch.long)
        u, v = np.meshgrid(np.linspace(0, W - 1, W, dtype=int), np.linspace(0, H - 1, H, dtype=int))
        u = torch.from_numpy(u.reshape(-1)).to(camera_intrinsics.device)
        v = torch.from_numpy(v.reshape(-1)).to(camera_intrinsics.device)
    else:
        image_indices = (torch.zeros(batch_size) + img_index).type(torch.long)  # Sample random images
        u = torch.randint(W, (batch_size,), device=camera_intrinsics.device)  # Sample random pixels
        v = torch.randint(H, (batch_size,), device=camera_intrinsics.device)

    focal = camera_intrinsics[0]
    t = camera_extrinsics[img_index, :3]
    r = camera_extrinsics[img_index, -3:]

    # Creating the c2w matrix, Section 4.1 from the paper
    phi_skew = torch.stack([torch.cat([torch.zeros(1, device=r.device), -r[2:3], r[1:2]]),
                            torch.cat([r[2:3], torch.zeros(1, device=r.device), -r[0:1]]),
                            torch.cat([-r[1:2], r[0:1], torch.zeros(1, device=r.device)])], dim=0)
    alpha = r.norm() + 1e-15
    R = torch.eye(3, device=r.device) + (torch.sin(alpha) / alpha) * phi_skew + (
            (1 - torch.cos(alpha)) / alpha ** 2) * (phi_skew @ phi_skew)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)
    c2w = torch.cat([c2w, torch.tensor([[0., 0., 0., 1.]], device=c2w.device)], dim=0)

    rays_d_cam = torch.cat([((u.to(camera_intrinsics.device) - .5 * W) / focal).unsqueeze(-1),
                            (-(v.to(camera_intrinsics.device) - .5 * H) / focal).unsqueeze(-1),
                            - torch.ones_like(u).unsqueeze(-1)], dim=-1)
    rays_d_world = torch.matmul(c2w[:3, :3].view(1, 3, 3), rays_d_cam.unsqueeze(2)).squeeze(2)
    rays_o_world = c2w[:3, 3].view(1, 3).expand_as(rays_d_world).to(camera_intrinsics.device)
    return rays_o_world, F.normalize(rays_d_world, p=2, dim=1), (image_indices, v.cpu(), u.cpu())

def render_rays(sdf_grid, appearance_model, rays_o, rays_d):
    import time
    start = time.time()
    
    ray_data = sdf_grid(rays_o, rays_d)

    print("generate_sdf", time.time() - start)

    intersections = []
    normals = []
    directions = []
    mask = []
    for i, data in enumerate(ray_data):
        mask.append(data is not None)
        if data is not None:
            [intersection, normal] = data
            directions.append(rays_d[i])
            intersections.append(intersection)
            normals.append(normal)

    directions = torch.stack(directions, dim=0)
    intersections = torch.stack(intersections, dim=0)
    normals = torch.stack(normals, dim=0)

    colors = appearance_model(intersections, directions, normals)

    return colors, intersections, mask

def load_images(data_path):
    img_list = []
    images = []
    image = None
    with open("output/reconstructed_img.txt") as f:
        img_list = f.readlines()
    img_list = [l.strip() for l in img_list]
    for i in range(len(img_list)):
        image_path = img_list[i]
        image = imread(os.path.join(data_path, image_path))
        images.append(image)
    return images, len(img_list)

def get_grid_resolution(point_cloud):
    minx, miny, minz, maxx, maxy, maxz = \
        np.min(point_cloud[:,0]), np.min(point_cloud[:,1]), np.min(point_cloud[:,2]), \
        np.max(point_cloud[:,0]), np.max(point_cloud[:,1]), np.max(point_cloud[:,2])

    minx, miny, minz, maxx, maxy, maxz = \
        int(minx*1.5), int(miny*1.5), int(minz*1.5), int(maxx*1.5), int(maxy*1.5), int(maxz*1.5)

    x_length = maxx - minx
    y_length = maxy - miny
    z_length = maxz - minz
    grid_size = np.array([x_length, y_length, z_length])

    arg_sort = np.argsort(grid_size)

    resolution = 255
    grid_resolution = np.array([255, 255, 255])

    size_box = grid_size[arg_sort[1]] / 255

    grid_resolution[arg_sort[0]] = np.ceil(grid_size[arg_sort[0]]/size_box)
    grid_resolution[arg_sort[2]] = np.ceil(grid_size[arg_sort[2]]/size_box)

    grid_size[arg_sort[0]] = grid_resolution[arg_sort[0]]*size_box
    grid_size[arg_sort[2]] = grid_resolution[arg_sort[2]]*size_box

    return minx, miny, minz, maxx, maxy, maxz, grid_resolution

def train(sdf_grid, appearance_model, optimizers, schedulers, num_img, camera_extrinsics, camera_intrinsics, batch_size,
          nb_epochs):
    import time
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):
        ids = np.arange(num_img)
        np.random.shuffle(ids)
        for img_index in ids:
            start = time.time()

            image = images[img_index]
            
            H, W = image.shape[:2]

            rays_o, rays_d, samples_idx = sample_batch(camera_extrinsics, camera_intrinsics,
                                                       batch_size, H, W, img_index=img_index)

            print(time.time() - start)

            gt_px_values = torch.from_numpy(image[samples_idx[1], samples_idx[2]]).to(camera_intrinsics.device)

            print(time.time() - start)

            regenerated_px_values, intersections, mask = render_rays(sdf_grid, appearance_model, rays_o, rays_d)

            loss = ((gt_px_values[mask]/255 - regenerated_px_values) ** 2).sum()

            print(time.time() - start)

            print("backward", loss.item())

            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            training_loss.append(loss.item())

            print(time.time() - start)
        for scheduler in schedulers:
            scheduler.step()

        if epoch % 5 == 0 and epoch != 0:
            torch.save(sdf_grid.state_dict(), "output/sdf_grid.pth")
            torch.save(appearance_model.state_dict(), "output/appearance_model.pth")

    return training_loss

images = []

if __name__ == "__main__":
    images, num_img = load_images('ystad_kloster')
    point_cloud = np.load("output/points_3d.npy")
    camera_extrinsics = np.load("output/cameras_extrinsic.npy")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nb_epochs = int(1e4)
    batch_size = 1024

    print(device)

    camera_intrinsics = torch.ones(1, device=device)*2378.98305085
    camera_extrinsics = torch.from_numpy(np.array([np.hstack((cv2.Rodrigues(cam[:,:3])[0].ravel(), cam[:, 3].ravel())) for cam in camera_extrinsics])).float().to(device)

    minx, miny, minz, maxx, maxy, maxz, grid_resolution = get_grid_resolution(point_cloud)

    sdf_grid = SDFGrid(minx, miny, minz, maxx, maxy, maxz, grid_resolution[0], grid_resolution[1], grid_resolution[2], device)
    appearance_model = AppearanceModel()

    sdf_grid.to(device)
    appearance_model.to(device)

    sdf_optimizer = torch.optim.Adam(sdf_grid.parameters(), lr=0.001)
    scheduler_sdf = torch.optim.lr_scheduler.MultiStepLR(
        sdf_optimizer, [10 * (i + 1) for i in range(nb_epochs // 10)], gamma=0.9954)

    appearance_optimizer = torch.optim.Adam(appearance_model.parameters(), lr=0.001)
    scheduler_appearance = torch.optim.lr_scheduler.MultiStepLR(
        appearance_optimizer, [10 * (i + 1) for i in range(nb_epochs // 10)], gamma=0.9954)

    train(sdf_grid, appearance_model, [sdf_optimizer, appearance_optimizer], [scheduler_sdf, scheduler_appearance], 
          num_img, camera_extrinsics, camera_intrinsics,
          batch_size, nb_epochs)