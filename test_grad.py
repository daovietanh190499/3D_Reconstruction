import torch
input = torch.tensor([[[
    [[1., 2., 3.],
     [4., 5., 6.]]]]],
    dtype=torch.float64,
    requires_grad=True)

# 2x3x2
grid = torch.tensor([[[  # x,y
    [[ 1.,  1., 1.],  # 1
     [-1., -1., 1.],  # 6
     [ 0.,  1., 1.]], # 5

    [[ 0.,  0., 0.],     # between 2 and 5   == (2 + 5) / 2 == 3.5
     [ 0.,  0.5, 0.],    # between 3.5 and 5 == (3.5 + 5) / 2 == 4.25
     [-1., -1., 1.]]]]],  # etc
     dtype=torch.float64)

interpolation_mode = 'bilinear'
padding_mode = 'zeros'
align_corners = True

res = torch.nn.functional.grid_sample(input, grid, interpolation_mode, padding_mode, align_corners)
print(res)
res.mean().backward()
print(input.grad)