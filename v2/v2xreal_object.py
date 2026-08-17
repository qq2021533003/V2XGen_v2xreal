import os
import sys
import yaml
import numpy as np
import open3d as o3d
from utils.v2x_file import read_Bin_PC, load_yaml, pcd_to_np
from utils.road_split import road_split
from build.mtest.utils import box_utils
from utils.common_utils import pc_numpy_2_o3d

class V2XRealInfo:
    """
    Load V2X-Real Vehicle and Road Info
    """
    def __init__(self, bg_index, is_ego=True, dataset_config=None):
        """
        1. load vehicle's data and labels
        2. split road
        """
        self.bg_index = bg_index
        self.is_ego = is_ego
        self.dataset_config = dataset_config

        path_info = self.load_data_path()

        # read pcd data
        # bg_pc = pcd_to_np(path_info['bg_path'])         # read background pcd from .pcd files
        # road_pc, non_road_pc, road_label = road_split(bg_pc[:, :3], path_info['road_path'],
        #                                               path_info['road_label_path'])

        # TODO: V2X-Real 是 .bin 格式的数据
        # read bin data
        bg_xyz = read_Bin_PC(path_info['bg_path'], True)  # read bg pcd from .bin files
        road_pc, non_road_pc, road_label = road_split(bg_xyz, path_info['road_path'],
                                                      path_info['road_label_path'])

        # TODO: v2x-real 需要转换一下位姿变为矩阵
        param = load_yaml(path_info['param_path'])

        # TODO: 这里不改 param 里的，而是增添新的变量（v2v4real数据集也适配一下）
        self.lidar_pose = pose_6d_to_4x4(param['lidar_pose'])
        self.true_ego_pose = pose_6d_to_4x4(param['true_ego_pose'])
        # param['lidar_pose'] = pose_6d_to_4x4(param['lidar_pose'])
        # param['true_ego_pose'] = pose_6d_to_4x4(param['true_ego_pose'])

        # self.pc = bg_pc
        self.pc = bg_xyz
        self.param = param
        self.road_pc = road_pc
        self.no_road_pc = non_road_pc
        self.road_label = road_label
        self.vehicles_info = {}
        self.recent_deleted_car_id = -1

        self.load_vehicles_info_v2x()
        # self.load_vehicles_info()

    def get_vehicles_nums(self):
        return len(list(self.param['vehicles'].keys()))

    # TODO: 暂时注释掉
    def load_vehicles_info(self):

        # init vehicle info lists
        objs_yaw = []
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

            objs_yaw.append(yaw_degree * np.pi / 180)   # 转换为弧度制
            objs_yaw_degree.append(yaw_degree)
            objs_l.append(length)
            objs_w.append(width)
            objs_h.append(height)
            objs_loc.append(center)

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

        for i in self.vehicles_info.keys():
            self.vehicles_info[i]["corner"] = corners_lidar[0]
            corners_lidar = corners_lidar[1:]


    def load_vehicles_info_v2x(self):
        # init vehicle info lists
        objs_yaw = []
        objs_yaw_degree = []
        objs_l = []
        objs_w = []
        objs_h = []
        objs_loc = []

        # 4x4 LiDAR pose matrix under CARLA system
        lidar_pose = np.array(self.param["lidar_pose"])
        lidar_pose_inv = np.linalg.inv(lidar_pose)

        # 关键步骤1：提取LiDAR在全局坐标系下的欧拉角（重点是yaw角）
        lidar_R = lidar_pose[:3, :3]  # 提取LiDAR旋转矩阵
        lidar_roll, lidar_pitch, lidar_yaw = rotation_matrix_to_euler(lidar_R)

        for i, vehicle in self.param["vehicles"].items():
            obj_type = vehicle['obj_type']
            vehicle_type_list = ['Car', 'Truck', 'Bus', 'Van']
            if obj_type not in vehicle_type_list:
                if obj_type != 'Pedestrian':
                    print(obj_type)
                continue


            # TODO: 看看需不需要 roll 和 pitch
            yaw_degree = vehicle['angle'][1]
            vehicle_yaw_global = np.radians(yaw_degree)

            length = vehicle['extent'][0] * 2
            width = vehicle['extent'][1] * 2
            height = vehicle['extent'][2] * 2

            # vehicle center under CALAR system
            center_global = np.array(vehicle['location']) + np.array(vehicle['center'])

            # TODO: 将全局坐标转换为相对 LiDAR 的局部坐标
            # conversion of central position
            center_global_homo = np.hstack([center_global, 1.0])
            center_local_homo = lidar_pose_inv @ center_global_homo.T
            center_local = center_local_homo[:3]

            # conversion of yaw (R_z)
            # TODO: 看看车框是否完全包含到车辆点云
            vehicle_yaw_local = vehicle_yaw_global - lidar_yaw
            vehicle_yaw_local = (vehicle_yaw_local + np.pi) % (2 * np.pi) - np.pi
            vehicle_yaw_local_degree = np.degrees(vehicle_yaw_local)

            objs_yaw.append(vehicle_yaw_local)  # 局部yaw角（弧度制，用于后续框生成）
            objs_yaw_degree.append(vehicle_yaw_local_degree)  # 局部yaw角（角度制，用于存储）
            objs_l.append(length)
            objs_w.append(width)
            objs_h.append(height)
            objs_loc.append(center_local)

            self.vehicles_info[i] = {
                'yaw_degree': vehicle_yaw_local_degree,  # 存储局部yaw角（角度制）
                'yaw_global_degree': yaw_degree,  # 可选：保留全局yaw角，用于调试
                'length': length,
                'width': width,
                'height': height,
                'center_global': center_global,  # 全局坐标（可选，用于调试）
                'center': center_local  # 局部坐标（核心，用于可视化）
            }

        rots = np.array(objs_yaw)
        l = np.array(objs_l).reshape(-1, 1)
        w = np.array(objs_w).reshape(-1, 1)
        h = np.array(objs_h).reshape(-1, 1)
        loc_lidar = np.array(objs_loc)  # 此时是相对LiDAR的局部坐标

        rots = rots.reshape(-1, 1)

        gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots], axis=1)

        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar).tolist()

        # TODO: ???修正原代码的致命错误：角点赋值一一对应（而非全部取corners_lidar[0]）
        for idx, i in enumerate(self.vehicles_info.keys()):
            self.vehicles_info[i]["corner"] = corners_lidar[idx]  # 按索引匹配

        del corners_lidar

    def delete_vehicle_of_id(self, car_id):
        self.param['vehicles'].pop(car_id, None)
        self.vehicles_info.pop(car_id, None)
        self.recent_deleted_car_id = car_id

    def get_vis_save_path(self):
        vis_save_dir = self.dataset_config.v2x_vis_saved_dir
        if not os.path.exists(vis_save_dir):
            os.makedirs(vis_save_dir)
        # vis_save_path = os.path.join(vis_save_dir, f"{self.bg_index:06d}.gltf")
        return vis_save_dir

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




