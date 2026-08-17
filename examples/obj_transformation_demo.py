import numpy as np
import random
import math
import core.obj_delete as delete
import core.obj_insert as insert
import utils.random_param as rand
from core.occlusion_treatment import get_occ_rate_of_obj
from utils.common_utils import center_system_transform, rz_degree_system_transform
from logger import CLogger

# Combine of insert and delete operations to these interfaces.


def vehicle_insert(ego_info, cp_info):
    """
    Insert a car at a randomly generated location in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :return: transformation result, True or False
    """
    success_flag = False
    count = 1

    while not success_flag:
        # over 10 times
        if count >= 10:
            return False

        CLogger.info(f"try insert {count} times...")

        position, rz_degree = rand.get_insert_location(ego_info)

        # get id to cut
        success_flag, ego_id, cp_id = insert.vehicle_insert(ego_info, cp_info, position, True, transformation="rotation")
        count += 1

    return True


def vehicle_delete(ego_info, cp_info, car_id=0):
    """
    Delete a car of the car_id in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :param car_id: the car id for delete
    :return: transformation result, True or False
    """
    CLogger.info(f"background index = {ego_info.bg_index}, delete ego vehicle id = {car_id}")
    success_flag, v2x_ego_center, v2x_cp_center = \
        delete.vehicle_delete(ego_info, cp_info, car_id)
    if success_flag:
        return True

    return False


def vehicle_translation(ego_info, cp_info, car_id):
    """
    Translate a car of the car_id to a randomly generated location in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :param car_id: the car id for translation
    :return: transformation result, True or False
    """
    success_flag = False
    cnt = 1

    # delete the car in the original location
    while not success_flag:
        if cnt >= 10:
            return False
        success_flag = delete.vehicle_delete(ego_info, cp_info, car_id)
        cnt += 1

    # delete successful
    success_flag = False
    cnt = 1

    while not success_flag:
        # over 10 times
        if cnt >= 10:
            return False

        CLogger.info(f"try translation {cnt} times...")
        # insert car to new location
        position, rz_degree = rand.get_insert_location(ego_info)
        success_flag, ego_id, cp_id = insert.vehicle_insert(ego_info, cp_info, position, True, True, rz_degree, transformation="rotation")

        cnt += 1
    return True


def vehicle_scaling(ego_info, cp_info, car_id):
    """
    Scaling a car of the car_id to a randomly generated scale in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :param car_id: the car id for scaling
    :return: transformation result, True or False
    """
    scaling_flag = False
    cnt = 1
    ego_corner = ego_info.vehicles_info[car_id]['corner']
    rz_degree = -ego_info.vehicles_info[car_id]['yaw_degree']
    position = list(ego_info.vehicles_info[car_id]['center'][:2])

    # the range of scaling
    min_ratio, max_ratio = 0.9, 1.1

    # ego_center = ego_info.vehicles_info[car_id]['center']

    # delete the car in the original location
    while not scaling_flag:
        if cnt >= 10:
            return False
        scaling_flag = delete.vehicle_delete(ego_info, cp_info, car_id)
        cnt += 1

    # delete successful
    scaling_flag = False
    cnt = 1

    while not scaling_flag:
        # over 10 times
        if cnt >= 10:
            return False

        ratio = random.uniform(min_ratio, max_ratio)
        CLogger.info(f"try scaling {cnt} times..., scaling ratio = {ratio}")
        corner_center = np.mean(ego_corner, axis=0)
        vectors = ego_corner - corner_center
        scaled_vector = vectors * ratio
        scaling_box = scaled_vector + corner_center

        scaling_flag, ego_id, cp_id = insert.vehicle_insert(ego_info, cp_info, position, False, True, rz_degree, scaling_box, transformation="rotation")
        cnt += 1
    return True


def vehicle_rotation(ego_info, cp_info, car_id):
    """
    Rotation a car of the car_id to a randomly generated yaw degree in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :param car_id: the car id for translation
    :return: transformation result, True or False
    """
    success_flag = False
    cnt = 1
    org_degree = -ego_info.vehicles_info[car_id]['yaw_degree']
    position = list(ego_info.vehicles_info[car_id]['center'][:2])
    corner = ego_info.vehicles_info[car_id]['corner']

    # delete the car in the original location
    while not success_flag:
        if cnt >= 10:
            return False
        success_flag = delete.vehicle_delete(ego_info, cp_info, car_id)
        cnt += 1

    # delete successful
    success_flag = False
    cnt = 1

    while not success_flag:
        # over 10 times
        if cnt >= 10:
            return False

        rot_degree = rand.get_random_rotation()
        CLogger.info(f"try rotation {cnt} times..., rot degree = {rot_degree}")
        rz_degree = org_degree + rot_degree

        success_flag, ego_id, cp_id = insert.vehicle_insert(ego_info, cp_info, position, False, True, rz_degree, corner, transformation="rotation")

        cnt += 1

    return True




