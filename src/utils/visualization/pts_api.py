import trimesh

def vis_numpy_pts(pts):
    '''
    将点云可视化为图像

    Parameters:
    -pts (numpy.array): 维度为[N, 3]的点云

    Returns:
    None: 直接可视化出来
    '''

    cloud_close = trimesh.points.PointCloud(pts)
    scene = trimesh.Scene(cloud_close)
    scene.show()