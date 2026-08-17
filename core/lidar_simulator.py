# -*- coding: utf-8 -*-
# @Time : 2022/12/16 23:20
# @Author : Junior_Jo
# @FileName: lidar_simulator.py
import open3d as o3d
import numpy as np
import math
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
import config
import os

"""
对mesh投射得到较为真实的点云
"""

def get_rays(horizontal_left,
             horizontal_right,
             vertical_down,
             vertical_up,
             horizontal_resolution,
             vertical_resolution,
             ):
    """
    Args:光线相关参数
        horizontal_left:       水平方向左侧最大角度
        horizontal_right:      水平方向右侧最大角度
        vertical_down:         垂直方向下方最大角度
        vertical_up:           垂直方向上方最大角度
        horizontal_resolution: 水平方向分辨率
        vertical_resolution:   垂直方向分辨率

    Returns:
        光线
    """
    points = ray_direction(horizontal_left,
                           horizontal_right,
                           vertical_down,
                           vertical_up,
                           horizontal_resolution,
                           vertical_resolution,
                           config.lidar_config.r)

    rays = create_rays(config.lidar_config.lidar_position, points)
    # print(rays)
    return rays


def create_rays(lidar_position, point_directions):
    assert len(lidar_position) == 3
    rays = []
    for point_direction in point_directions:
        ray = (lidar_position[0], lidar_position[1], lidar_position[2],
               point_direction[0], point_direction[1], point_direction[2])
        rays.append(ray)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    return rays


def ray_direction(horizontal_left,
                  horizontal_right,
                  vertical_down,
                  vertical_up,
                  horizontal_resolution,
                  vertical_resolution,
                  r):
    """
    Args: 获取光线的方向
        horizontal_left:       水平方向左侧最大角度
        horizontal_right:      水平方向右侧最大角度
        vertical_down:         垂直方向下方最大角度
        vertical_up:           垂直方向上方最大角度
        horizontal_resolution: 水平方向分辨率
        vertical_resolution:   垂直方向分辨率
        r:                     长度单位——默认值为 1

    Returns:
        光线的方向
    """
    points_list = []
    # 垂直方向圆弧个数
    circle_num = int((vertical_up - vertical_down) // vertical_resolution)  # 垂直方向圆弧个数

    for i in range(circle_num):
        degree = vertical_resolution * i + vertical_down
        rad_phi = degree * np.pi / 180  # phi的大小
        pts = ray_direction_circle_simulation(horizontal_left, horizontal_right, horizontal_resolution, r, rad_phi)
        points_list += pts

    return points_list


def ray_direction_circle_simulation(horizontal_left,
                                    horizontal_right,
                                    horizontal_resolution,
                                    r,
                                    rad_phi):
    """
    Args:
        horizontal_left:       水平方向左侧最大角度
        horizontal_right:      水平方向右侧最大角度
        horizontal_resolution: 水平方向分辨率
        r:                     长度单位——默认值为 1
        rad_phi:               光线方向与垂直方向的夹角

    Returns:

    """
    laster_num = int((horizontal_right - horizontal_left) // horizontal_resolution)  # 水平方向一个圆弧所包含的激光数量

    points_list = []  # 每个点代表从原点射出的激光的方向

    for i in range(laster_num):
        degree = horizontal_resolution * i + horizontal_left
        rad_theta = degree * np.pi / 180
        # TODO:
        if rad_phi > 90 * np.pi / 180:
            rad_phi = 90 * np.pi / 180
        elif rad_phi < -90 * np.pi / 180:
            rad_phi = -90 * np.pi / 180
        if rad_theta > 90 * np.pi / 180:
            rad_theta = 90 * np.pi / 180
        elif rad_theta < -90 * np.pi / 180:
            rad_theta = -90 * np.pi / 180

        x, y, z = spherical_to_cartesian(r, rad_phi, rad_theta)  # 球坐标系转化成笛卡尔坐标系
        points_list.append((x, y, z))

    return points_list


# render objet min lidar ray range args
def get_min_ray_args4render_by_obj(obj, extend_range, rays_args):
    """
    :param obj: mesh物体
    :param extend_range: 针对边框的扩充度大小——角度制
    :param rays_args: 分辨率
    :return:list-[horizontal_left,
            horizontal_right,
            vertical_down,
            vertical_up,
            horizontal_resolution,
            vertical_resolution]
    """
    horizontal_left, horizontal_right = rays_args[0], rays_args[1]
    vertical_down, vertical_up = rays_args[2], rays_args[3]
    horizontal_resolution, vertical_resolution = rays_args[4], rays_args[5]

    box_points = obj.get_oriented_bounding_box().get_box_points()  # TODO: fix a bug

    temp = np.asarray(box_points)  # box的八个角点坐标
    for point in temp:
        _, latitude, longitude = cartesian_to_spherical(*list(point))

        latitude, longitude = latitude.value, longitude.value
        # print(latitude,longitude)
        # 经度，以x轴向y轴为正方向，2π为量程，无负数
        if longitude > np.pi: longitude = -(np.pi * 2 - longitude)

        # 换为角度制
        latitude = math.degrees(latitude)
        longitude = math.degrees(longitude)

        # 更新边界
        if horizontal_left > longitude: horizontal_left = longitude
        if horizontal_right < longitude: horizontal_right = longitude
        if vertical_down > latitude: vertical_down = latitude
        if vertical_up < latitude: vertical_up = latitude

    # 稍微扩充一下范围
    horizontal_left -= extend_range
    horizontal_right += extend_range
    vertical_down -= extend_range
    vertical_up += extend_range

    return [horizontal_left, horizontal_right, vertical_down, vertical_up, horizontal_resolution, vertical_resolution]


def render_pcd(pointcloud_xyz, average, variance, severity, loss_rate):
    """
    gaussian_noise & random loss
    :param pointcloud_xyz:获取点云的xyz坐标 (N,3)
    :param average:       正太分布的平均值
    :param variance:      正太分布的方差
    :param severity:      正太分布的缩小值(严重性)
    :param loss_rate:     点云随机丢失比率
    :return:经过高斯渲染后的点云xyz
    """
    try:  # TODO:
        row, column = pointcloud_xyz.shape
    except:
        print(pointcloud_xyz)
        print(type(pointcloud_xyz))
        raise ValueError()
    jitter = np.random.normal(average, variance, size=(row, column)) * severity
    new_pc_xyz = (pointcloud_xyz + jitter).astype('float32')
    # 获取索引
    index = np.random.choice(row, size=int(row * (1 - loss_rate)), replace=False)
    return new_pc_xyz[index]


def get_obj_pcd(car, rays_args, render_args):
    """
    :param car        被投射的车辆mesh
    :param rays_args: 光线相关参数  horizontal_left,
                             horizontal_right,
                             vertical_down,
                             vertical_up,
                             horizontal_resolution,
                             vertical_resolution
    :param render_args: 渲染相关参数
                            average:       正太分布的平均值
                            variance:      正太分布的方差
                            severity:      正太分布的缩小值(严重性)
                            loss_rate:     点云随机丢失比率
    :return:  被光线扫描后的物体点云坐标——可以直接构成点云
    """
    car_t = o3d.t.geometry.TriangleMesh.from_legacy(car)  # 会把car的颜色信息也存储，所以放在颜色信息之前
    car.paint_uniform_color([1, 1, 0])  # 黄色

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(car_t)

    rays = get_rays(*rays_args)

    ans = scene.cast_rays(rays)  # 投射光线获取交点
    distance = ans["t_hit"].numpy()  # 距离的数组，以方向的三个值的平方和的平方根作为单位

    # 光线的后三位是光线的方向
    xyz_direction = rays.numpy()[:, 3:]
    # 实际的距离要用distance进行转换
    xyz_position = []  # 点云点的xyz

    # 现在有r,缺少phi,theta
    for i in range(len(xyz_direction)):
        r, phi, theta = cartesian_to_spherical(*list(xyz_direction[i, :]))
        if distance[i] == np.inf:  # 光线与车的mesh没有交点
            pass
        else:
            x, y, z = spherical_to_cartesian(distance[i], phi, theta)
            xyz_position.append([x, y, z])

    points_obj = render_pcd(np.array(xyz_position), *render_args)  # 高斯扰动加下采样

    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(points_obj)

    return pcd_obj


# lidar ray args
def get_ray_args():
    vertical_resolution = config.lidar_config.vertical_resolution
    horizontal_resolution = config.lidar_config.horizontal_resolution
    horizontal_left = config.lidar_config.horizontal_left
    horizontal_right = config.lidar_config.horizontal_right
    vertical_down = config.lidar_config.vertical_down
    vertical_up = config.lidar_config.vertical_up
    return [horizontal_left, horizontal_right, vertical_down, vertical_up, horizontal_resolution, vertical_resolution]


def lidar_simulation(mesh_obj):
    """
    :param mesh_obj: 用于生成点云的mesh原型
    :return:  返回依据mesh原型获得的点云数据
    """

    # 获得雷达基本配置
    rays_args = get_ray_args()

    # 根据网格范围获得雷达配置
    rays_args = get_min_ray_args4render_by_obj(mesh_obj,
                                               config.lidar_config.extend_range,
                                               rays_args)  # 获取光线参数

    render_args = [config.lidar_config.noise_average,
                   config.lidar_config.noise_variance,
                   config.lidar_config.noise_severity,
                   config.lidar_config.loss_rate]

    pcd_obj = get_obj_pcd(mesh_obj, rays_args, render_args)  # 光线模拟出的点云

    return pcd_obj


# TODO:
def _test_ref(point=(0, 0, 0)):  # lidar point

    alpha = 0  # extinction coefficient
    input_ref = 0  # reflection
    rmax = 200  # max range (m)
    dR = 0.09  # range accuracy(m)

    x, y, z = point
    ran = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    ref = input_ref * np.exp(-2 * alpha * ran)
    P0 = ref * np.exp(-2 * alpha * ran) / (ran ** 2)  # reflected power
    Pmin = 0.9 * rmax ** (-2)  # min measurable power (arb units)
    snr = P0 / Pmin  # signal noise ratio
    sig = dR / np.sqrt(2 * snr)  # std of range uncertainty
    if P0 < Pmin:
        # del points:
        ...
    else:
        # add noise
        # noise = N(0,sig)
        ...


def mix_pc(pcd_road, pcd_non_road, pc_objs):
    """
    :param pcd_road:       背景道路点云
    :param pcd_non_road:   背景非道路点云
    :param pc_objs:        插入物体点云
    :param data_bg         背景索引——用于生成文件名
    :param save_pc_dir     存储混合点云的文件夹
    """
    # 将路面，非路面，插入物体的点云转化为np数组
    np_pcd_road = np.asarray(pcd_road.points)
    np_pcd_non_road = np.asarray(pcd_non_road.points)
    flag = True
    for pc_obj in pc_objs:
        if flag:
            np_pc_objs = np.asarray(pc_obj.points)
            flag = False  # repair bugs
        else:
            np_pc_objs = np.concatenate([np_pc_objs, np.asarray(pc_obj.points)], axis=0)

    # 合并
    mixed_pc_three = np.concatenate([np_pcd_road, np_pcd_non_road, np_pc_objs], axis=0)

    return mixed_pc_three

    # mix = np.fromfile(mixed_pc_save_path, dtype=np.float32)
    # mix2 = np.fromfile(mixed_pc_save_path)
    # print(mix.shape,mix2.shape)
    # assert 1 == 2


def complet_pc(mixed_pc_three):
    assert mixed_pc_three.shape[1] == 3
    # 补足第四列信息——默认为0
    hang = mixed_pc_three.shape[0]
    b = np.zeros((hang, 1))
    mixed_pc = np.concatenate([mixed_pc_three, b], axis=1)
    return mixed_pc


if __name__ == '__main__':
    # KITTI
    lidar_height = 1.73
    verticle_view = 26.8
    horizontal_view = 360
    beam_num = 64
    horizontal_resolution = 0.09
    max_verticle_view = 15
    min_verticle_view = -11.8
    verticle_resolution = verticle_view / beam_num  # 0.41875
    print(verticle_resolution)
    # rotation_direction  逆时针
    # range: 120 m
    # 2 cm distance accuracy
    # refresh_rate = 10 HZ
    # 130万点/秒 points/second
    # sample_rate =horizontal_view / horizontal_resolution * refresh_rate * beam_num # 采样率
