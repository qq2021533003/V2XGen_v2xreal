import os
import numpy as np
import open3d as o3d
import copy
import utils.visual as vis
from utils.common_utils import pc_numpy_2_o3d


def load_road_split_labels(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
    return labels


def split_pc(labels):
    inx_road_arr = []
    inx_other_road_arr = []
    inx_other_ground_arr = []
    inx_no_road_arr = []
    '''
        40: "road"          
        44: "parking"       
        48: "sidewalk"      
        49: "other-ground"  
        72: "terrain"       
    '''
    for i in range(len(labels)):
        lb = labels[i][0]
        if lb == 40:
            inx_road_arr.append(i)
        elif lb in [44, 48]:
            inx_other_road_arr.append(i)
        elif lb in (49, 70, 71, 72, 79):
            inx_other_ground_arr.append(i)
        else:
            inx_no_road_arr.append(i)
    return inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr


def road_split(pc, road_pc_path, road_label_path):
    pc_path = road_pc_path
    label_path = road_label_path

    # TODO: 确认策略后，需要解注释节省开销
    if os.path.exists(pc_path):
        labels = load_road_split_labels(label_path)
        road_pc = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 3))
        inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr = split_pc(labels)

        # _pc_non_road = pc[inx_other_road_arr + inx_other_ground_arr + inx_no_road_arr]
        _pc_non_road = pc[inx_no_road_arr]
    else:
    # if True:
        labels = load_road_split_labels(label_path)

        inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr = split_pc(labels)
        if len(inx_road_arr) <= 10:
            return None, None, None, None

        _pc_road, _pc_other_road, _pc_other_ground, _pc_no_road = \
            pc[inx_road_arr], pc[inx_other_road_arr], pc[inx_other_ground_arr], pc[inx_no_road_arr]

        # _pc_non_road = pc[inx_other_road_arr + inx_other_ground_arr + inx_no_road_arr]
        _pc_non_road = pc[inx_no_road_arr]

        pcd_road = pc_numpy_2_o3d(_pc_road)

        cl, ind = pcd_road.remove_radius_outlier(nb_points=7, radius=1)
        pcd_inlier_road = pcd_road.select_by_index(ind)

        road_pcd_filtered = pcd_inlier_road

        # TODO：保存一份采样前的路面点云数据，采样后的用于路面补全，采样前的用于道路检测
        # _pc_inter_ori = np.asarray(road_pcd_filtered.points)
        # dis_ori = np.linalg.norm(_pc_inter_ori, axis=1, ord=2)
        # _pc_inter_valid_ori = _pc_inter_ori[dis_ori > 4]

        # road_pc_ori = _pc_inter_valid_ori.astype(np.float32)
        # road_pc_ori.astype(np.float32).tofile(pc_path, )

        # 测试离散过滤效果
        _pc_inter_nofilter = np.asarray(pcd_inlier_road.points)
        dis_nofilter = np.linalg.norm(_pc_inter_nofilter, axis=1, ord=2)
        _pc_inter_valid_filter = _pc_inter_nofilter[dis_nofilter > 4]
        road_pc_nofilter = _pc_inter_valid_filter.astype(np.float32)
        road_pc_nofilter.astype(np.float32).tofile(pc_path, )

        # road_pc_ori = _pc_inter_valid_ori.astype(np.float32)
        # road_pc_ori.astype(np.float32).tofile(pc_path, )
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(road_pcd_filtered, 10)

        pcd_inter = mesh.sample_points_uniformly(number_of_points=50000)

        _pc_inter = np.asarray(pcd_inter.points)
        dis = np.linalg.norm(_pc_inter, axis=1, ord=2)
        _pc_inter_valid = _pc_inter[dis > 4]

        road_pc = _pc_inter_valid.astype(np.float32)
        road_pc.astype(np.float32).tofile(pc_path, )
        # vis.show_pc(road_pc)
        # vis.show_pc(road_pc_ori)

    return road_pc, _pc_non_road, labels


if __name__ == '__main__':
    ...
