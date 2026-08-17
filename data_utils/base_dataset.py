import os
import sys
import yaml
import numpy as np
import open3d as o3d

from utils.v2x_file import read_Bin_PC, load_yaml, pcd_to_np
from utils.road_split import road_split
from build.mtest.utils import box_utils
from utils.common_utils import pc_numpy_2_o3d, pose_6d_to_4x4, rotation_matrix_to_euler


class BaseDataset:
    """
    加载点云帧信息的基类
    """
    def __init__(self, bg_index: int, scene: str, dataset_config, is_ego=True):
        """
        1. load vehicle's data and labels
        2. split road
        """
        self.bg_index = bg_index
        self.is_ego = is_ego
        self.scene = scene
        self.dataset_config = dataset_config    # 数据集配置信息

        self.param = ""
        self.recent_deleted_car_id = -1  # 记录刚删除的车辆 id（方便除插入以外的变换操作）

    # def get_vehicles_nums(self):
    #     return len(list(self.param['vehicles'].keys()))

    def delete_vehicle_of_id(self, car_id):
        """
        删除 car_id 车辆
        :param car_id:
        :return:
        """
        self.param['vehicles'].pop(car_id, None)   # 删除标签
        self.vehicles_info.pop(car_id, None)    # 删除车辆信息
        self.recent_deleted_car_id = car_id     # 记录 id

    @staticmethod
    def completed_pc(mixed_pc_three):
        assert mixed_pc_three.shape[1] == 3

        hang = mixed_pc_three.shape[0]
        b = np.zeros((hang, 1))
        mixed_pc = np.concatenate([mixed_pc_three, b], axis=1)
        return mixed_pc

    @staticmethod
    def pcd_file_to_np(pcd_file):
        """
        Read  pcd and return numpy array.

        Parameters
        ----------
        pcd_file : str
            The pcd file that contains the point cloud.

        Returns
        -------
        pcd : o3d.PointCloud
            PointCloud object, used for visualization
        pcd_np : np.ndarray
            The lidar data in numpy format, shape:(n, 4)

        """
        pcd = o3d.io.read_point_cloud(pcd_file)

        xyz = np.asarray(pcd.points)
        # we save the intensity in the first channel
        intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)

        pcd_np = np.hstack((xyz, intensity))

        return np.asarray(pcd_np, dtype=np.float32)