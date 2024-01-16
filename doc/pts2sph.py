# import numpy as np
# from scipy.special import sph_harm

# def compute_spherical_harmonics_coefficients(points, l_max):
#     """
#     计算点云数据的球谐函数系数。
    
#     参数:
#     - points: 二维数组，形状为 (n, 3)，表示点云数据。
#     - l_max: 球谐函数的最大阶数。
    
#     返回:
#     - coefficients: 一个字典，包含球谐函数的系数。
#     """
#     # 将点云归一化，确保其位于单位球上
#     norm_points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
    
#     coefficients = {}
    
#     for l in range(l_max + 1):
#         for m in range(-l, l + 1):
#             # 计算每个点在球谐函数上的系数
#             coef = np.mean(sph_harm(m, l, norm_points[:, 1], norm_points[:, 0])) + \
#                    np.mean(sph_harm(m, l, norm_points[:, 2], np.sqrt(1 - norm_points[:, 2]**2)))
            
#             coefficients[(l, m)] = coef
            
#     return coefficients

# # 生成一个假设的点云数据（简化示例）
# points_cloud = np.random.rand(1024, 3)  # 这只是一个随机生成的点云，你可能有自己的真实数据

# # 计算球谐函数系数，假设我们只关心前5个阶数
# l_max = 5
# coefficients = compute_spherical_harmonics_coefficients(points_cloud, l_max)

# # 打印前几个系数作为示例
# for (l, m), coef in coefficients.items():
#     print(f"Coefficient for l={l}, m={m}: {coef}")


# import numpy as np
# from scipy.special import sph_harm

# # 生成模拟的1024x3的点云数据集（仅作为示例）
# num_points = 1024
# points = np.random.rand(num_points, 3)  # 随机生成点云数据

# def cartesian_to_spherical(xyz):
#     """将三维直角坐标转换为球坐标"""
#     r = np.linalg.norm(xyz, axis=1)
#     theta = np.arccos(xyz[:, 2] / r)
#     phi = np.arctan2(xyz[:, 1], xyz[:, 0])
#     return r, theta, phi

# # 将点云数据转换为球坐标
# r, theta, phi = cartesian_to_spherical(points)

# def compute_spherical_harmonics_coefficients(r, theta, phi, l_max):
#     """计算球谐函数的系数"""
#     coefficients = []
#     for l in range(l_max + 1):
#         for m in range(-l, l + 1):
#             coef = np.mean(sph_harm(m, l, phi, theta) * r**l)   # 归一化 r**l
#             coefficients.append(coef)
#     return coefficients

# l_max_value = 5  # 设定最大的l值
# coefficients = compute_spherical_harmonics_coefficients(r, theta, phi, l_max_value)

# # 输出前几个系数（这只是为了示例，实际应用中可能需要保存或进一步处理这些系数）
# print("Spherical Harmonics Coefficients:", coefficients[:10])

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

def cartesian_to_spherical(xyz):
    """将三维直角坐标转换为球坐标"""
    r = np.linalg.norm(xyz, axis=1)
    theta = np.arccos(xyz[:, 2] / r)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return r, theta, phi

pts = np.load('/data4/LJ/datasets/nocs/mean_shapes.npy')[4]

# 1. 生成笔记本电脑形状的简化点云（示例数据）
# 这里，我们使用一个简化的方法来生成一个简单的笔记本电脑形状的点云
# 实际上，你可能需要从一个真实的3D模型中提取点云数据
num_points = 1024
r, theta, phi = cartesian_to_spherical(pts)

# 2. 计算球谐函数的系数
l_max = 5
m_values = range(-l_max, l_max + 1)

coefficients = {}
for l in range(l_max + 1):
    for m in m_values:
        Ylm = sph_harm(m, l, phi, theta)
        real_part = np.real(Ylm)
        imag_part = np.imag(Ylm)
        
        # 计算系数时，考虑实部和虚部
        coefficients[(l, m)] = np.sum((real_part + imag_part) * r**l)

# 3. 使用球谐函数重建点云
reconstructed_points = np.zeros((3, num_points))
for l in range(l_max + 1):
    for m in m_values:
        Ylm = sph_harm(m, l, phi, theta)
        real_part = np.real(Ylm)
        imag_part = np.imag(Ylm)
        
        # 更新重建点云时，考虑实部和虚部
        reconstructed_points[0] += coefficients[(l, m)] * real_part
        reconstructed_points[1] += coefficients[(l, m)] * imag_part

print(coefficients)
a

# 4. 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reconstructed_points[0], reconstructed_points[1], reconstructed_points[2], c='b', marker='o')
# ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
