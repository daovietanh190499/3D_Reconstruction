import numpy as np

def filter_point_cloud(verts):
  verts = verts * 200
  mean = np.mean(verts, axis=0)
  temp = verts - mean
  dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
  indx = np.where(dist < np.mean(dist) + 300)
  verts = verts[indx]
  return verts/200

point_cloud = np.load("output/test_points.npy")
past_cloud = np.load("output/points_3d.npy")
past_cloud = filter_point_cloud(past_cloud)
point_cloud = np.concatenate((point_cloud, past_cloud))

colors = np.ones_like(point_cloud)*255
for i in range(8):
  colors[i] = np.array([0, 0, 255])

def to_ply(img_dir, point_cloud, colors):
    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    print(out_colors.shape, out_points.shape)
    verts = np.hstack([out_points, out_colors])
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

to_ply("output/test_2.ply", point_cloud, colors)