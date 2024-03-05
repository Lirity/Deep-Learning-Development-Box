import numpy as np
import open3d as o3d

def read_obj2pts(file_path, num_samples=1024):
    mesh = o3d.io.read_triangle_mesh(file_path)
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_samples)
    pts = np.asarray(point_cloud.points)

    return pts


if __name__ == "__main__":
    obj_file_path = "/data4/LJ/datasets/nocs/obj_models/real_test/bottle_red_stanford_norm.obj"
    pts = read_obj2pts(obj_file_path, num_samples=1024)

    print(pts.shape)
    
    import sys
    BASE_DIR = '/data4/lj/master'
    sys.path.append(BASE_DIR)
    from src.utils.visualization.pts_api import vis_numpy_pts

    vis_numpy_pts(pts)
