def voxel_traversal(rays, _bin_size):
    # rays.shape (N_rays, 8): origin(3) direction(3), smallest t(1), largest t(1)
    # _bin_size scaler
    # return: (N_rays, Max_steps, 3)
    
    rays_o, rays_d, near, far = rays[:, :3], rays[:, 3:6], rays[:, 6:7], rays[:, 7:8]
    _bin_size = float(_bin_size)
    voxel_visited = []
    
    ray_start = rays_o + torch.mul(rays_d, near)
    ray_end = rays_o + torch.mul(rays_d, far)

    current_voxel = torch.floor_divide(ray_start, _bin_size)
    last_voxel = torch.floor_divide(ray_end, _bin_size)

    step = torch.ones_like(rays_d)
    step[rays_d < 0] = -1

    next_voxel_boundary = torch.mul((current_voxel + step), _bin_size)

    tMax = torch.true_divide((next_voxel_boundary - ray_start), rays_d)
    tMax[rays_d == 0] = float('inf') # max double

    tDelta = torch.true_divide(torch.mul(step, _bin_size), rays_d)
    tDelta[rays_d == 0] = float('inf') # max double

    diff = torch.zeros_like(rays_d)
    mask = torch.logical_and((current_voxel != last_voxel), (rays_d < 0))
    diff[mask] -= 1

    voxel_visited += [torch.clone(current_voxel)]
    def get_maskt(current_voxel, last_voxel):
        test0 = torch.logical_and((current_voxel[:, 0] == current_voxel[:, 0]), (torch.mul(step[:, 0], current_voxel[:, 0]) < torch.mul(step[:, 0], last_voxel[:, 0]))) # not null and not equal to last voxel
        test1 = torch.logical_and((current_voxel[:, 1] == current_voxel[:, 1]), (torch.mul(step[:, 1], current_voxel[:, 1]) < torch.mul(step[:, 1], last_voxel[:, 1])))
        test2 = torch.logical_and((current_voxel[:, 2] == current_voxel[:, 2]), (torch.mul(step[:, 2], current_voxel[:, 2]) < torch.mul(step[:, 2], last_voxel[:, 2])))
        maskt = torch.logical_or(test0, torch.logical_or(test1, test2)) # true if xyz is different
        return maskt

    maskt = get_maskt(current_voxel, last_voxel)
    i = 0
    while torch.any(maskt):

        mask11 = torch.logical_and((tMax[:, 0] < tMax[:, 1]), (tMax[:, 0] < tMax[:, 2])) # X
        mask12 = torch.logical_and((tMax[:, 1] <= tMax[:, 0]), (tMax[:, 1] < tMax[:, 2])) # Y
        mask13 = torch.logical_and((tMax[:, 0] >= tMax[:, 1]), (tMax[:, 1] >= tMax[:, 2])) # Z
        mask14 = torch.logical_and((tMax[:, 0] >= tMax[:, 2]), (tMax[:, 1] > tMax[:, 0])) # Z
        maskx = mask11
        masky = mask12
        maskz = torch.logical_or(mask13, mask14)

        maskx = torch.logical_and(maskt, maskx)
        if torch.any(maskx):
            current_voxel[:, 0][maskx] += step[:, 0][maskx]
            tMax[:, 0][maskx] += tDelta[:, 0][maskx]

        masky = torch.logical_and(maskt, masky)
        if torch.any(masky):
            current_voxel[:, 1][masky] += step[:, 1][masky]
            tMax[:, 1][masky] += tDelta[:, 1][masky]

        maskz = torch.logical_and(maskt, maskz)
        if torch.any(maskz):
            current_voxel[:, 2][maskz] += step[:, 2][maskz]
            tMax[:, 2][maskz] += tDelta[:, 2][maskz]

        voxel_visited += [torch.clone(current_voxel)]
        maskt = get_maskt(current_voxel, last_voxel)
        current_voxel[torch.logical_not(maskt)] = float('nan')

        i += 1

    voxel_visited = torch.stack(voxel_visited).permute(1,0,2)
    return voxel_visited