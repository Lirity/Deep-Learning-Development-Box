import cv2

import numpy as np

from PIL import Image

def vis_bbox(img, pred_size, RT, s, intrinsics, img_name, img_color):
    bbox_3d = get_3d_bbox(pred_size, 0)
    transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RT, s)
    projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
    img = draw_bboxes(img, projected_bbox, color=img_color).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(img_name + '.png')


def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar

    Returns:
        bbox_3d: [3, N]

    """
    # x 正负 前后 y 正负 上下 z 正负 左右
    bbox_3d = np.array([[0, 0, 0],
                        [+size[0] / 2, +size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, +size[1] / 2, -size[2] / 2], 
                        [-size[0] / 2, +size[1] / 2, +size[2] / 2], 
                        [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                        [-size[0] / 2, -size[1] / 2, -size[2] / 2]
                        ]) + shift
    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def transform_coordinates_3d(coordinates, RT, s):
    """
    Args:
        coordinates: [3, N]
        sRT: [4, 4]

    Returns:
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)]) * s
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
    return new_coordinates

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)
    print(projected_coordinates)
    return projected_coordinates

def draw_bboxes(img, img_pts, dir_pts=None, color=(0, 0, 255), width=3, cx=(0, 0, 255), cy=(0, 255, 0),
                cz=(255, 0, 0)):  # colorx, colory, colorz
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([3, 4, 8, 7], [4, 8, 7, 3]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, width)
    # draw pillars in minor darker color
    color_pillar = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
    for i, j in zip([1, 2, 5, 6], [3, 4, 7, 8]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, width)
    # draw top layer in original color
    for i, j in zip([1, 2, 6, 5], [2, 6, 5, 1]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, width)

    if dir_pts is None:
        img_pts = img_pts[1:]
        center = tuple(np.int32(img_pts.mean(0)))
        center_x = tuple(np.int32((img_pts[[0, 1, 4, 5]].mean(0) - img_pts.mean(0)) * 0.8 + img_pts.mean(0)))
        center_y = tuple(np.int32((img_pts[[0, 1, 2, 3]].mean(0) - img_pts.mean(0)) * 0.8 + img_pts.mean(0)))
        center_z = tuple(np.int32((img_pts[[0, 2, 4, 6]].mean(0) - img_pts.mean(0)) * 0.8 + img_pts.mean(0)))
    else:
        dir_pts = np.int32(dir_pts).reshape(-1, 2)
        center = dir_pts[0]
        center_x, center_y, center_z = dir_pts[1], dir_pts[2], dir_pts[3]
    img = cv2.line(img, center, center_x, cx, width)
    img = cv2.line(img, center, center_y, cy, width)
    img = cv2.line(img, center, center_z, cz, width)
    return img

if __name__ == "__main__":
    path = "/data4/lj/master/doc/0000.JPG"
    img = cv2.imread(path)
    pred_size = np.array([0.64684695, 0.41495627, 0.63984406])
    RT = np.array([[-0.27297553, -0.0794856,  -0.44678032, -0.03034966],
                   [ 0.334779,   -0.38725805, -0.13564841, -0.01541225],
                   [-0.30635542, -0.35236275,  0.2498662,   0.49667513],
                   [ 0.,          0.,          0.,          1.        ]])
    intrinsics = np.array([[435.30108642578125, 0 , 242.00372314453125],
                          [0, 435.30108642578125, 319.216796875],
                          [0, 0, 1]])
    intrinsics = np.array([[384.866, 0 , 326.371],
                          [0, 383.857, 243.675],
                          [0, 0, 1]])
    vis_bbox(img, pred_size, RT, np.linalg.norm(pred_size), intrinsics, "my_color", (0, 0, 0))