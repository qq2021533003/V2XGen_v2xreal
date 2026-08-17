import config
import random
import numpy as np
import core.obj_delete as delete
import core.obj_insert as insert
import utils.visual as vis
import utils.common_utils as common
from utils.v2x_object import V2XInfo
from logger import CLogger


def vehicle_insert(ego_info, cp_info):
    """
    Baseline insert (insert obj on road where baseline setting)

    :param ego_info: ego vehicle info object
    :param cp_info: cooperative vehicle info object
    :return: car id for cut (if needed)
    """
    success_flag = False
    count = 1

    # loop while success
    while not success_flag:
        CLogger.info(f"try baseline insert {count} times...")
        pos = insert.generate_base_insert_pos(ego_info.pc[:, :3])
        rz_degree = np.random.uniform(-180, 180)
        success_flag, ego_id, cp_id = insert.base_insert(ego_info, cp_info, pos, rz_degree)
        if success_flag:
            return ego_id, cp_id
        count += 1


def vehicle_delete(ego_info, cp_info, car_id=0):
    """
    Baseline delete, only the vehicle is deleted, and the lidar scan line is not processed.

    :param ego_info: ego vehicle info object
    :param cp_info: cooperative vehicle info object
    :param car_id: delete object of car_id
    :return: operation center for cut (if needed)
    """
    return delete.base_delete(ego_info, cp_info, car_id)


def vehicle_translate(ego_info, cp_info, car_id=0, translate=None):
    """
    Baseline translation, translate the vehicle point to the specified location.

    :param ego_info: ego vehicle info object
    :param cp_info: cooperative vehicle info object
    :param car_id: translate object of car_id
    :param translate: the same location as V2XGen translate
    :return: car id for cut (if needed)
    """
    if translate is None:
        translate = [0, 0]
    CLogger.info(f"Background index = {ego_info.bg_index}, translate vehicle car id = {car_id}")
    ego_car_id = car_id

    cp_car_id, flag = common.find_cp_vehicle_id(cp_info, ego_car_id)

    translate_vector = [translate[0], translate[1], 0]

    ego_corner = ego_info.vehicles_info[ego_car_id]["corner"]
    ego_obj_idx = common.get_pc_index_in_corner(ego_info.pc, ego_corner)
    ego_info.pc[ego_obj_idx] += translate_vector
    for i in range(len(ego_info.param["vehicles"][ego_car_id]["location"])):
        ego_info.param["vehicles"][ego_car_id]["location"][i] += translate_vector[i]    # update center param
    ego_info.load_vehicles_info()   # reload vehicle info

    # translate cooperative object
    if flag:
        cp_corner = cp_info.vehicles_info[cp_car_id]["corner"]
        cp_obj_idx = common.get_pc_index_in_corner(cp_info.pc, cp_corner)
        cp_info.pc[cp_obj_idx] += translate_vector
        for i in range(len(ego_info.param["vehicles"][ego_car_id]["location"])):
            cp_info.param["vehicles"][cp_car_id]["location"][i] += translate_vector[i]
        cp_info.load_vehicles_info()

        # visualize after baseline translate
        # vis.show_ego_and_cp_for_translation(ego_info, cp_info, ego_car_id, vis_corner)
        # vis.show_obj_for_translation(ego_info, ego_car_id, vis_corner)
        # vis_cp_corner = common.points_system_transform(vis_corner, ego_info.param['lidar_pose'],
        #                                                cp_info.param['lidar_pose'])
        # vis.show_obj_for_translation(cp_info, cp_car_id, vis_cp_corner)
        return ego_car_id, cp_car_id

    return ego_car_id, -1


def vehicle_scaling(ego_info, cp_info, car_id=0, scaling_ratio=1):
    """
    Baseline scaling，scale the vehicle point in the center of the vehicle box.

    :param ego_info: ego vehicle info object
    :param cp_info: cooperative vehicle info object
    :param car_id: scale object of car_id
    :return: car id for cut (if needed)
    :param scaling_ratio: the same ratio as V2XGen translate
    """
    CLogger.info(f"Background index = {ego_info.bg_index}, baseline scaling vehicle car id = {car_id}")
    ego_corner = ego_info.vehicles_info[car_id]["corner"]
    ego_obj_idx = common.get_pc_index_in_corner(ego_info.pc, ego_corner)
    pts = ego_info.pc[ego_obj_idx]

    cp_car_id, flag = common.find_cp_vehicle_id(cp_info, car_id)

    pts_center = ego_info.vehicles_info[car_id]["center"]
    vectors = pts[:, :3] - pts_center

    # scaling vector
    scaled_vectors = vectors * scaling_ratio
    ego_info.pc[ego_obj_idx] = scaled_vectors + pts_center
    ego_extent = ego_info.param["vehicles"][car_id]['extent']
    ego_info.param["vehicles"][car_id]['extent'] = \
        [ego_extent[0] * scaling_ratio, ego_extent[1] * scaling_ratio, ego_extent[2] * scaling_ratio]
    ego_info.load_vehicles_info()  # reload vehicle information

    # cooperative vehicle scaling
    if flag:
        cp_corner = cp_info.vehicles_info[cp_car_id]["corner"]
        cp_obj_idx = common.get_pc_index_in_corner(cp_info.pc, cp_corner)
        pts = cp_info.pc[cp_obj_idx]
        pts_center = cp_info.vehicles_info[cp_car_id]["center"]
        vectors = pts - pts_center
        scaled_vectors = vectors * scaling_ratio
        cp_info.pc[cp_obj_idx] = scaled_vectors + pts_center
        cp_extent = cp_info.param["vehicles"][cp_car_id]['extent']
        cp_info.param["vehicles"][cp_car_id]['extent'] = \
            [cp_extent[0] * scaling_ratio, cp_extent[1] * scaling_ratio, cp_extent[2] * scaling_ratio]
        cp_info.load_vehicles_info()

        # visualize after transformation
        # vis.show_ego_and_cp_with_id(ego_info, cp_info, car_id, cp_car_id)
        # vis.show_obj_with_car_id(ego_info, car_id)
        # vis.show_obj_with_car_id(cp_info, cp_car_id)
        return car_id, cp_car_id

    return car_id, -1


def vehicle_rotation(ego_info, cp_info, car_id, ego_rz_degree, cp_rz_degree):
    """
    Baseline rotation，rotate the vehicle point in the center of the vehicle box.

    :param ego_info: ego vehicle info object
    :param cp_info: cooperative vehicle info object
    :param car_id: scale object of car_id
    :param ego_rz_degree: ego vehicle rotate degree
    :param cp_rz_degree: cooperative vehicle rotate degree
    :return:
    """
    CLogger.info(f"Background index = {ego_info.bg_index}, baseline rotate vehicle car id = {car_id}")
    ego_corner = ego_info.vehicles_info[car_id]["corner"]
    ego_obj_idx = common.get_pc_index_in_corner(ego_info.pc, ego_corner)

    cp_car_id, flag = common.find_cp_vehicle_id(cp_info, car_id)

    # rotation matrix
    theta = np.radians(ego_rz_degree)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    pts = ego_info.pc[ego_obj_idx]
    pts_center = ego_info.vehicles_info[car_id]["center"]
    vectors = pts - pts_center

    # rotate points in corner
    rotated_vectors = np.dot(vectors, R.T)
    ego_info.pc[ego_obj_idx] = rotated_vectors + pts_center
    ego_info.param["vehicles"][car_id]["angle"][1] = np.degrees(theta)
    ego_info.load_vehicles_info()

    # cooperative rotation
    if flag:
        cp_theta = np.radians(cp_rz_degree)
        R = np.array([
            [np.cos(cp_theta), -np.sin(cp_theta), 0],
            [np.sin(cp_theta), np.cos(cp_theta), 0],
            [0, 0, 1]
        ])
        cp_corner = cp_info.vehicles_info[cp_car_id]["corner"]
        cp_obj_idx = common.get_pc_index_in_corner(cp_info.pc, cp_corner)
        pts = cp_info.pc[cp_obj_idx]
        pts_center = cp_info.vehicles_info[cp_car_id]["center"]
        vectors = pts - pts_center
        rotated_vectors = np.dot(vectors, R.T)
        cp_info.pc[cp_obj_idx] = rotated_vectors + pts_center
        cp_info.param["vehicles"][cp_car_id]["angle"][1] = np.degrees(cp_theta)
        cp_info.load_vehicles_info()

        # visualize after transformation
        # vis.show_ego_and_cp_with_id(ego_info, cp_info, car_id, cp_car_id)
        # vis.show_obj_with_car_id(ego_info, car_id)
        # vis.show_obj_with_car_id(cp_info, cp_car_id)
        return car_id, cp_car_id

    return car_id, -1


if __name__ == '__main__':
    # function test
    select_obj_list = []
    select_data_num = config.v2x_config.select_data_num

    index_list = list(range(1, select_data_num + 1))

    for bg_index in range(1, select_data_num + 1):
        index_list.remove(bg_index)
        ego_obj = V2XInfo(bg_index)
        cp_obj = V2XInfo(bg_index, is_ego=False)
        vehicle_num = len(ego_obj.param)

        # random get car index
        car_index = random.randint(0, vehicle_num - 1)

        vehicle_translate(ego_obj, cp_obj, car_index)
