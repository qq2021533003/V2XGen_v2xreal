import os
import sys
import yaml
import numpy as np
import open3d as o3d

from base_dataset import BaseDataset
from utils.v2x_file import read_Bin_PC, load_yaml, pcd_to_np
from utils.road_split import road_split
from build.mtest.utils import box_utils
from utils.common_utils import pc_numpy_2_o3d, pose_6d_to_4x4, rotation_matrix_to_euler


class V2VDataset(BaseDataset):
    """
    加载 V2V 点云帧信息 (适配 V2V4Real)
    """
    def __init__(self, bg_index: int, scene: str, dataset_config, is_ego=True):
        super().__init__(bg_index, scene, dataset_config, is_ego)

        self.pc = None
        self.param = {}  # 读取/保存的标签
        self.road_pc = None
        self.no_road_pc = None
        self.road_label = None
        self.vehicles_info = {}

        self.lidar_pose = []
        self.true_ego_pose = []

        # 区分 V2V4Real 和 V2X-Real 的数据和标签格式
        self.init_dataset()

        self.load_vehicles_info()  # 从标签中加载车辆信息

    def init_dataset(self):
        scene_dir = os.path.join(self.dataset_config.dataset_root, self.dataset_config.dataset, self.scene)

        if self.is_ego:
            data_dir = os.path.join(scene_dir, "0")
        else:
            data_dir = os.path.join(scene_dir, "1")

        bg_yaml_path = os.path.join(data_dir, f"{self.bg_index:06d}.yaml")

        road_split_label_path = os.path.join(data_dir, "predictions", f"{self.bg_index:06d}.label")

        road_split_pc_dir = os.path.join(data_dir, "road_pcd")

        # 存放路面点云
        os.makedirs(road_split_pc_dir, exist_ok=True)

        road_split_pc_path = os.path.join(road_split_pc_dir, f"{self.bg_index:06d}.bin")

        # 读取不同的数据类型
        if self.dataset_name == 'V2V4Real':
            bg_pc_path = os.path.join(data_dir, "pcd", f"{self.bg_index:06d}.pcd")
            self.pc = pcd_to_np(bg_pc_path)  # read bg pcd from .pcd files
        elif self.dataset_name == 'V2XReal':
            bg_pc_path = os.path.join(data_dir, "bin", f"{self.bg_index:06d}.bin")
            self.pc = read_Bin_PC(bg_pc_path, True)  # read bg pcd from .bin files

        # 分割路径
        self.road_pc, self.non_road_pc, self.road_label = road_split(self.pc[:, :3], road_split_pc_path,
                                                                        road_split_label_path)

        # 加载数据标签信息
        vehicle_label = load_yaml(bg_yaml_path)
        self.param = vehicle_label

        # 新增两个成员，用于坐标变换（v2x-real 需要将真值框变换到 LiDAR 局部坐标系下）
        if self.dataset_name == 'V2XReal':
            self.lidar_pose = vehicle_label['lidar_pose']
            self.true_ego_pose = vehicle_label['true_ego_pose']
        elif self.dataset_name == 'V2XReal':
            self.lidar_pose = pose_6d_to_4x4(vehicle_label['lidar_pose'])
            self.true_ego_pose = pose_6d_to_4x4(vehicle_label['true_ego_pose'])

    def load_vehicles_info(self):
        objs_yaw = []  # 偏航角（绕z轴旋转角）
        objs_yaw_degree = []
        objs_l = []
        objs_w = []
        objs_h = []
        objs_loc = []
        for i, vehicle in self.param["vehicles"].items():
            yaw_degree = vehicle['angle'][1]
            length = vehicle['extent'][0] * 2
            width = vehicle['extent'][1] * 2
            height = vehicle['extent'][2] * 2
            center = np.array(vehicle['location']) + np.array(vehicle['center'])

            objs_yaw.append(yaw_degree * np.pi / 180)
            objs_yaw_degree.append(yaw_degree)
            objs_l.append(length)
            objs_w.append(width)
            objs_h.append(height)
            objs_loc.append(center)

            # 保存完整的旋转角度 yaw,长，宽，高，中心点坐标
            self.vehicles_info[i] = {
                'yaw_degree': yaw_degree,
                'length': length,
                'width': width,
                'height': height,
                'center': center
            }
        rots = np.array(objs_yaw)

        l, h, w = np.array(objs_l).reshape(-1, 1), np.array(objs_h).reshape(-1, 1), np.array(objs_w).reshape(-1, 1)
        loc_lidar = np.array(objs_loc)

        rots = rots[..., np.newaxis]
        gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots], axis=1)
        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar).tolist()

        # 保存车框信息
        for i in self.vehicles_info.keys():
            self.vehicles_info[i]["corner"] = corners_lidar[0]
            corners_lidar = corners_lidar[1:]