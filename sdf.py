import numpy as np

def compute_polynomial_coefficients(sdf_values, corners, origin, direction):
    # Unpack the sdf values
    s000, s100, s010, s110, s001, s101, s011, s111 = sdf_values
    p000, p100, p010, p110, p001, p101, p011, p111 = corners
    px000, py000, pz000 = p000[0], p000[1], p000[2]
    px100 = p100[0]
    py010 = p010[1]
    pz001 = p001[2]
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


import torch

def compute_real_roots(c3, c2, c1, c0):
    # Compute a, b, and c based on the provided coefficients
    a = c2 / c3
    b = c1 / c3
    c = c0 / c3

    # Compute Q and R
    Q = (a ** 2 - 3 * b) / 9
    R = (2 * a ** 3 - 9 * a * b + 27 * c) / 54

    # Initialize tensor for storing roots
    # roots = torch.zeros(3, dtype=torch.float32, requires_grad=True)

    roots = torch.zeros(1, dtype=torch.float32, requires_grad=True).to(c3.device)

    # Case 1: Three real roots when R^2 < Q^3
    # Using torch.where to keep the computation differentiable
    R2 = R ** 2
    Q3 = Q ** 3
    condition = R2 < Q3
    
    theta = torch.acos(torch.clamp(R / torch.sqrt(Q3), -1.0, 1.0))  # Clamp to avoid numerical issues
    sqrt_Q = torch.sqrt(Q)
    
    t1 = -2 * sqrt_Q * torch.cos(theta / 3) - a / 3
    t2 = -2 * sqrt_Q * torch.cos((theta - 2 * torch.pi) / 3) - a / 3
    t3 = -2 * sqrt_Q * torch.cos((theta + 2 * torch.pi) / 3) - a / 3

    # Compute the real root in case of one real root
    A = -torch.copysign(torch.abs(R) + torch.sqrt(R2 - Q3), R) ** (1/3)
    B = torch.where(A != 0, Q / A, torch.zeros_like(A))

    t_single = A + B - a / 3
    
    # # Select roots based on the condition
    # roots_case_1 = torch.stack([t1, t2, t3])
    # roots_case_1 = torch.sort(roots_case_1).values  # Sort roots in ascending order

    roots = torch.where(condition.unsqueeze(-1), t1, t_single)

    return roots

import torch
import torch.nn as nn

# Learnable SDF grid model
class LearnableSDFGrid(nn.Module):
    def __init__(self, minx, miny, minz, maxx, maxy, maxz, nx, ny, nz):
        super(LearnableSDFGrid, self).__init__()
        # Define the grid bounds
        self.min_bound = torch.tensor([minx, miny, minz])
        self.max_bound = torch.tensor([maxx, maxy, maxz])
        
        # Define the learnable SDF grid as a parameter (nx * ny * nz grid)
        self.sdf_values = nn.Parameter(torch.randn(nx, ny, nz))  # Random initialization
        
        # Grid size (number of voxels)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        # Voxel size (distance between grid points)
        self.voxel_size_x = (maxx - minx) / (nx - 1)
        self.voxel_size_y = (maxy - miny) / (ny - 1)
        self.voxel_size_z = (maxz - minz) / (nz - 1)
    
    # Query function to get 8 corner SDF values and their positions
    def query_voxels(self, point):
        # Compute voxel indices
        idx_x = ((point[0] - self.min_bound[0]) / self.voxel_size_x).long()
        idx_y = ((point[1] - self.min_bound[1]) / self.voxel_size_y).long()
        idx_z = ((point[2] - self.min_bound[2]) / self.voxel_size_z).long()

        if idx_x < 0 or idx_x >= self.nx - 1 or \
           idx_y < 0 or idx_y >= self.ny - 1 or \
           idx_z < 0 or idx_z >= self.nz - 1:
            return None  # Outside grid
        
        # Get 8 corner indices
        corners = [
            (idx_x, idx_y, idx_z),
            (idx_x + 1, idx_y, idx_z),
            (idx_x, idx_y + 1, idx_z),
            (idx_x + 1, idx_y + 1, idx_z),
            (idx_x, idx_y, idx_z + 1),
            (idx_x + 1, idx_y, idx_z + 1),
            (idx_x, idx_y + 1, idx_z + 1),
            (idx_x + 1, idx_y + 1, idx_z + 1)
        ]
        
        # Get the 8 SDF values and their positions
        sdf_vals = [self.sdf_values[c] for c in corners]
        positions = [(self.min_bound + torch.tensor(c) * torch.tensor([self.voxel_size_x, self.voxel_size_y, self.voxel_size_z])) for c in corners]
        
        return sdf_vals, positions

    # Ray-box intersection function
    def ray_box_intersection(self, origin, direction):
        inv_dir = 1.0 / direction
        tmin = (self.min_bound - origin) * inv_dir
        tmax = (self.max_bound - origin) * inv_dir
        
        tmin, _ = torch.max(torch.min(tmin, tmax), dim=0)
        tmax, _ = torch.min(torch.max(tmin, tmax), dim=0)
        
        if tmax < tmin:
            return None, None  # No intersection
        return tmin, tmax

    # Function to march through the grid along a ray
    def march_ray(self, origin, direction, t_near, t_far, step_size=0.1):
        t = t_near
        ray_data = []
        while t < t_far:
            point = origin + t * direction
            # Query voxel corners and their SDF values
            query_result = self.query_voxels(point)
            if query_result is not None:
                sdf_vals, positions = query_result
                ray_data.append((sdf_vals, positions))
            t += step_size
        return ray_data


point_cloud = np.load("output/points_3d.npy")

minx, miny, minz, maxx, maxy, maxz = \
    np.min(point_cloud[:,0]), np.min(point_cloud[:,1]), np.min(point_cloud[:,2]), \
    np.max(point_cloud[:,0]), np.max(point_cloud[:,1]), np.max(point_cloud[:,2])

print(minx*1.5, miny*1.5, minz*1.5, maxx*1.5, maxy*1.5, maxz*1.5)



