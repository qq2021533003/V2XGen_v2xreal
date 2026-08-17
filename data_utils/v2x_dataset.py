import os
import sys
import yaml
import numpy as np
import open3d as o3d

from config.common_config import SUPER_CLASS_MAP
from data_utils.base_dataset import BaseDataset
from utils.transformation_utils import x1_to_x2, x2_to_x1, x_to_world, matrix_to_pose
from utils.v2x_file import read_Bin_PC, load_yaml, pcd_to_np
from utils.road_split import road_split
from build.mtest.utils import box_utils
from opencood.utils.box_utils import boxes_to_corners_3d, project_box3d, create_bbx, corner_to_center
from utils.common_utils import get_obj_pcd_from_corner, pc_numpy_2_o3d

class V2XDataset(BaseDataset):
    """
    加载 V2X 点云帧信息 (适配 V2X-Real)
    """
    def __init__(self, bg_index: int, scene: str, dataset_config, is_ego=True):
        super().__init__(bg_index, scene, dataset_config, is_ego)

        self.pc = np.nan
        self.param = {}  # 读取/保存的标签
        self.road_pc = np.nan
        self.no_road_pc = np.nan  
        self.road_label = {}
        self.vehicles_info = {}
        self.objs_info = {}     # 保存所有对象，用于遮挡计算
        # self.dataset_name = "V2XReal"   # 标识数据集类型

        self.lidar_pose = []
        self.lidar_pose_6d = []
        self.true_ego_pose = []

        # 区分 V2V4Real 和 V2X-Real 的数据和标签格式
        self.init_dataset()

        self.load_vehicles_info()  # 从标签中加载车辆信息

    def get_vehicles_nums(self):
        return len(list(self.vehicles_info.keys()))

    def init_dataset(self):
        scene_dir = os.path.join(self.dataset_config.dataset_root, self.dataset_config.dataset, self.scene)

        if self.is_ego:
            data_dir = os.path.join(scene_dir, "0")
        else:
            data_dir = os.path.join(scene_dir, "1")

        bg_yaml_path = os.path.join(data_dir, "yaml", f"{self.bg_index:06d}.yaml")
        road_split_label_path = os.path.join(data_dir, "predictions", f"{self.bg_index:06d}.label")
        road_split_pc_dir = os.path.join(data_dir, "road_pcd")

        # 存放路面点云
        os.makedirs(road_split_pc_dir, exist_ok=True)

        road_split_pc_path = os.path.join(road_split_pc_dir, f"{self.bg_index:06d}.bin")

        # V2X-Real 的数据为 .bin 格式
        bg_pc_path = os.path.join(data_dir, "bin", f"{self.bg_index:06d}.bin")
        self.pc = read_Bin_PC(bg_pc_path, True)  # read bg pcd from .bin files

        # 分割路径
        self.road_pc, self.no_road_pc, self.road_label = road_split(self.pc[:, :3], road_split_pc_path,
                                                                        road_split_label_path)

        # 加载数据标签信息
        vehicle_label = load_yaml(bg_yaml_path)
        self.param = vehicle_label

        # 记录插入的数据
        self.param['inserted_ids'] = []

        # 将 6d 格式的位姿转换为 4*4 矩阵
        self.lidar_pose = x_to_world(vehicle_label['lidar_pose'])
        self.lidar_pose_6d = vehicle_label['lidar_pose']
        # self.true_ego_pose = pose_6d_to_4x4(vehicle_label['true_ego_pose'])

    def load_vehicles_info(self):
        GT_RANGE = [-100, -40, -15, 100, 40, 15]
        min_corner = np.array(GT_RANGE[:3])   # [minx, miny, minz]
        max_corner = np.array(GT_RANGE[3:])   # [maxx, maxy, maxz]
        min_num_corners = 2                   # 至少2个角点在范围内


        for i, vehicle in self.param["vehicles"].items():
            if not vehicle:
                continue

            # yaw_degree = vehicle['angle'][1]
            roll_degree, yaw_degree, pitch_degree = vehicle['angle']
            length = vehicle['extent'][0] * 2
            width = vehicle['extent'][1] * 2
            height = vehicle['extent'][2] * 2
            center = np.array(vehicle['location']) + np.array(vehicle['center'])

            # V2X-Real 点云存储在 LiDAR 局部坐标系，而标签标注坐标系为全局坐标系，因此需要进行坐标系对齐操作
            # 标签内的位姿信息
            object_pose = [center[0], center[1], center[2], roll_degree, yaw_degree, pitch_degree]

            # TODO: 只变换了 corner

            # 将当前对象的位姿信息转换到 LiDAR 局部坐标系下
            object2lidar = x1_to_x2(object_pose, self.lidar_pose_6d)

            # shape (3, 8) 生成物体在自身坐标系下的八个顶点
            bbx = create_bbx(vehicle['extent']).T
            # bounding box under ego coordinate shape (4, 8)
            bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

            # project the 8 corners to world coordinate 得到 LiDAR 坐标系下的八个角点 (8, 3)
            bbx_lidar = np.dot(object2lidar, bbx).T
            bbx_lidar = bbx_lidar[:, :3]

            # 变换航向和中心点到局部坐标系下
            center_lidar = object2lidar[:3, 3]
            yaw_degree_lidar = np.arctan2(object2lidar[1, 0], object2lidar[0, 0])
            yaw_degree_lidar = np.degrees(yaw_degree_lidar)

            #--------- 对待操作对象列表进行过滤，仅包含范围内有效 vehicle 对象 ---------

            # 过滤掉非 vehicle 对象
            obj_type = vehicle['obj_type']
            if obj_type not in SUPER_CLASS_MAP['vehicle']:
                continue

            # 保存所有对象，不进行距离和过滤
            self.objs_info[i] = {
                'yaw_degree': yaw_degree_lidar,
                'length': length,
                'width': width,
                'height': height,
                'center': center_lidar,
                'corner': bbx_lidar
            }

            # 过滤范围外的车辆
            in_range_mask = np.all((bbx_lidar >= min_corner) & (bbx_lidar <= max_corner), axis=1)
            if np.sum(in_range_mask) < min_num_corners:
                continue   # 丢弃该车辆
        
            # 过滤 gt box 内无点的车辆（幽灵标签），该对象无法作为操作对象
            vehicle_pcd = get_obj_pcd_from_corner(bbx_lidar, pc_numpy_2_o3d(self.pc))
            if not vehicle_pcd.has_points():
                continue

            self.vehicles_info[i] = {
                'yaw_degree': yaw_degree_lidar,
                'length': length,
                'width': width,
                'height': height,
                'center': center_lidar,
                'corner': bbx_lidar
            }

    def update_param_for_insert(self, extent, location, rz_degree, insert_car_id, use_old_id=False, ass_id=-1):
        """
        执行插入操作后，更新标签信息
        :param extent:
        :param location:
        :param rz_degree:
        :param insert_car_id: v2x-real 在全局命名，使用传入的 id 即可
        :param use_old_id:
        :param ass_id:
        :return:
        """
        # NOTE: 新 id 命名应该过滤掉 ego 和协同视角下的所有 id
        # if len(list(self.param['vehicles'].keys())) == 0:
        #     car_id = 1
        # elif use_old_id and self.recent_deleted_car_id != -1:
        #     car_id = self.recent_deleted_car_id
        # else:
        #     car_id = list(self.param['vehicles'].keys())[-1] + 1

        # TODO: 标签信息在全局坐标系下，执行插入操作后则在 lidar 局部坐标系下，需要进行坐标系变换
        object_in_lidar = [location[0], location[1], location[2], 0, rz_degree, 0]

        object2world = x2_to_x1(object_in_lidar, self.lidar_pose_6d)
        x, y, z, roll, yaw, pitch = matrix_to_pose(object2world)

        # V2X-Real 没有 ass_id 字段，而是多了 attribute 字段
        # attribute 统一设置为 ""，标识插入对象非静止对象
        car_dict = {
            'angle': [float(roll), float(yaw), float(pitch)],
            'attribute': '',
            'center': [0, 0, 0],
            'extent': extent.tolist(),
            'location': [float(x), float(y), float(z)],
            'obj_type': 'Car'
        }

        self.param['vehicles'][insert_car_id] = car_dict

        # 标识为新插入车辆，用于插入有效性检测
        # print(f"Inserted vehicle ID: {insert_car_id}")
        self.param['inserted_ids'].append(insert_car_id)

        self.load_vehicles_info()
    
    def delete_vehicle_of_id(self, car_id):
        self.param['vehicles'].pop(car_id, None)
        self.vehicles_info.pop(car_id, None)
        self.objs_info.pop(car_id, None)
        self.recent_deleted_car_id = car_id

    def save_data_and_label(self):
        """
        # TODO: 区分两个数据集，文件名（1， 2），可以预处理数据集统一改为 0,1
        文件夹结构:
        ├── folder_name
        │    └── scene_name
        │           ├── 0
        │           │   ├── xxx.bin
        │           │   └── xxx.yaml
        │           └── 1
        │                ├── xxx.bin
        │                └── xxx.yaml
        """
        saved_dir = os.path.join(self.dataset_config.gen_data_save_dir, self.scene)

        if self.is_ego:
            saved_path = os.path.join(saved_dir, "0")
        else:
            saved_path = os.path.join(saved_dir, "1")

        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        label_saved_path = os.path.join(saved_path, f"{self.bg_index:06d}.yaml")

        # 保存为 .bin 格式
        data_saved_path = os.path.join(saved_path, f"{self.bg_index:06d}.bin")
        saved_pc = self.completed_pc(self.pc[:, :3]).astype(np.float32)
        saved_pc.tofile(data_saved_path)

        with open(label_saved_path, 'w') as outfile:
            yaml.dump(self.param, outfile, default_flow_style=False)

    def get_corners(self, type="vehicle"):
        """
        返回车辆或所有对象的 box 角点
        """
        corners_list = []

        if type == "vehicle":
            for _, info in self.vehicles_info.items():
                corners_list.append(info['corner'])

            corners_lidar = np.array(corners_list)
        elif type == "obj":
            for _, info in self.objs_info.items():
                corners_list.append(info['corner'])

            corners_lidar = np.array(corners_list)
        # print(type(corners_lidar), corners_lidar.shape)

        return corners_lidar
    
    def get_objects_ids(self):
        """
        获取当前场景中所有对象的 id 列表，包括行人等，以防止新命名冲突
        """
        return list(self.param['vehicles'].keys())

    # TODO: 获得 lidar 扫描线分布配置
