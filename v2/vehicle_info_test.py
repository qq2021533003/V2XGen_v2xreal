import sys
import config
import random
import numpy as np
import open3d as o3d
import core.occlusion_treatment as occ
import utils.common_utils as common
import utils.visual as vis
from logger import CLogger
from utils.v2x_object import V2XInfo
from utils.common_utils import pc_numpy_2_o3d, is_box_containing_position
from core.obj_insert import insert_obj
from core.occlusion_treatment import get_delete_points_idx
from core.lidar_simulation import lidar_simulation, lidar_intensity_convert

def loading_info_verification_test(ego_info, cp_info, car_id):
    # 测试对象框是否能覆盖点云对象
    # TODO: 可视化中心点，看看是不是位置有问题
    corner = ego_info.vehicles_info[car_id]["corner"]
    vis.show_pc_with_info(ego_info)

    # TODO: 验证一下 car_id 是否就是 vehicle

    # TODO: 验证 lidar simulation 配置仰角和俯角是否搞反了