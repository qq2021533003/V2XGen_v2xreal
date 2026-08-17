# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Transformation utils
"""

import numpy as np


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system
    NOTE: 局部坐标系下的位姿转换到世界坐标系下，并返回转换矩阵

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.
    NOTE: 获得变换矩阵

    Parameters
    ----------
    x1 : list
        The pose of x1 under world coordinates.
    x2 : list
        The pose of x2 under world coordinates.

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    x1_to_world = x_to_world(x1)
    x2_to_world = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)

    transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix



def x2_to_x1(x1_in_x2, x2_pose):
    """
    把 x2 局部坐标系下的点，变换回 x1 坐标系

    Parameters
    ----------
    x1_in_x2 : list
        The pose of x1 in x2 coordinates.
    x2_pose : list
        The pose of x2 under world coordinates.

    Returns
    -------
    T_obj2world : np.ndarray
        The transformation matrix.

    """
    # obj -> LiDAR matrix
    T_obj2x2 = x_to_world(x1_in_x2)

    # LiDAR -> world matrix
    T_x22world = x_to_world(x2_pose)
    
    # 逆向：物体世界位姿 = lidar到世界 × 物体到lidar
    obj_to_world = np.dot(T_x22world, T_obj2x2)
    
    return obj_to_world  # 这就是物体的世界位姿矩阵



def dist_to_continuous(p_dist, displacement_dist, res, downsample_rate):
    """
    Convert points discretized format to continuous space for BEV representation.
    Parameters
    ----------
    p_dist : numpy.array
        Points in discretized coorindates.

    displacement_dist : numpy.array
        Discretized coordinates of bottom left origin.

    res : float
        Discretization resolution.

    downsample_rate : int
        Dowmsamping rate.

    Returns
    -------
    p_continuous : numpy.array
        Points in continuous coorindates.

    """
    p_dist = np.copy(p_dist)
    p_dist = p_dist + displacement_dist
    p_continuous = p_dist * res * downsample_rate
    return p_continuous


def matrix_to_pose(mat):
    """
    4x4 变换矩阵 → [x, y, z, roll, yaw, pitch]
    """
    x = mat[0, 3]
    y = mat[1, 3]
    z = mat[2, 3]
    
    # 旋转矩阵转欧拉角
    roll = np.arctan2(mat[2, 1], mat[2, 2])
    pitch = np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1]**2 + mat[2, 2]**2))
    yaw = np.arctan2(mat[1, 0], mat[0, 0])
    
    return [x, y, z, np.rad2deg(roll), np.rad2deg(yaw), np.rad2deg(pitch)]