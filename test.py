import cv2

import numpy as np

from src.utils.document.pkl_api import read_pkl

# 点云 + 对应颜色

def get_bbox(bbox):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    img_width = 480
    img_length = 640
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

def load_depth(img_path):
    """ Load depth image from img_path. """
    depth_path = img_path + '_depth.png'
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16

if __name__ == "__main__":
    sample_num = 1024
    img_path = "/data2/lj/dataset/nocs/Real/test/scene_1/0000"
    label_path = "/data2/lj/dataset/nocs/Real/test/scene_1/0000_label.pkl"

    bbox = read_pkl(label_path)['bboxes'][2]
    rmin, rmax, cmin, cmax = get_bbox(bbox)

    rgb = cv2.imread(img_path + '_color.png')[rmin:rmax, cmin:cmax, :]
    depth = load_depth(img_path)
    mask = cv2.imread(img_path + '_mask.png')[:, :, 2]

    # choose
    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    if len(choose)<=0:
        print('Error!')
    elif len(choose) <= sample_num:
        choose_idx = np.random.choice(np.arange(len(choose)), sample_num)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), sample_num, replace=False)
    choose = choose[choose_idx]

    # pts
    pts2 = depth.copy()[rmin:rmax, cmin:cmax].reshape((-1))[choose] / 1000.0
    pts0 = (self.xmap[rmin:rmax, cmin:cmax].reshape((-1))[choose] - cam_cx) * pts2 / cam_fx
    pts1 = (self.ymap[rmin:rmax, cmin:cmax].reshape((-1))[choose] - cam_cy) * pts2 / cam_fy
    pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,0)).astype(np.float32)