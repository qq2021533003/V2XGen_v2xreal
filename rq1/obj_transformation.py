import numpy as np
import random
import core.obj_delete as delete
import core.obj_insert as insert
import utils.random_param as rand
from logger import CLogger
import baseline_transformation as baseline

# All operations are integrated into the insert and delete operations.
# The translation, rotation and scaling operations need to delete the vehicle first, and then
# insert the vehicle according to the new parameters generated.


def vehicle_insert(ego_info, cp_info, ego_info_baseline, cp_info_baseline):
    """
    Insert a car at a randomly generated location in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :param ego_info_baseline: ego vehicle info for Baseline
    :param cp_info_baseline: cooperative vehicle info for Baseline
    :return: car id for cut (if needed)
    """
    success_flag = False
    count = 1

    while not success_flag:
        # over 10 times， return
        if count >= 10:
            return False, -1, -1, -1, -1

        CLogger.info(f"try insert {count} times...")

        position, rz_degree = rand.get_insert_location(ego_info)

        # get id to cut
        success_flag, ego_id, cp_id = insert.vehicle_insert(ego_info, cp_info, position, True, transformation="rotation")

        # After the V2XGen operation is successful, starting the Baseline operation.
        if success_flag:
            base_ego_id, base_cp_id = baseline.vehicle_insert(ego_info_baseline, cp_info_baseline)
            return True, ego_id, cp_id, base_ego_id, base_cp_id
        count += 1


def vehicle_delete(ego_info, cp_info, ego_info_baseline, cp_info_baseline, car_id=0):
    """
    Delete a car of the car_id in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :param ego_info_baseline: ego vehicle info for Baseline
    :param cp_info_baseline: cooperative vehicle info for Baseline
    :param car_id: the car id for delete
    :return: operation center for cut (if needed)
    """
    success_flag, v2x_ego_center, v2x_cp_center = \
        delete.vehicle_delete(ego_info, cp_info, car_id)

    # After the V2XGen operation is successful, starting the Baseline operation.
    if success_flag:
        baseline_ego_center, baseline_cp_center = \
            baseline.vehicle_delete(ego_info_baseline, cp_info_baseline, car_id)
        return True, v2x_ego_center, v2x_cp_center, baseline_ego_center, baseline_cp_center
    return False, [0, 0], [0, 0], [0, 0], [0, 0]


def vehicle_translation(ego_info, cp_info, ego_info_baseline, cp_info_baseline, car_id):
    """
    Translate a car of the car_id to a randomly generated location in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :param ego_info_baseline: ego vehicle info for Baseline
    :param cp_info_baseline: cooperative vehicle info for Baseline
    :param car_id: the car id for translation
    :return: car id for cut (if needed)
    """
    CLogger.info(f"Background index = {ego_info.bg_index}, translate vehicle car id = {car_id}")
    success_flag = False
    cnt = 1

    # delete the car in the original location
    while not success_flag:
        if cnt >= 10:
            return False, -1, -1, -1, -1
        success_flag = delete.vehicle_delete(ego_info, cp_info, car_id)
        cnt += 1

    # delete successful
    success_flag = False
    cnt = 1

    while not success_flag:
        # over 10 times， return
        if cnt >= 10:
            return False, -1, -1, -1, -1

        CLogger.info(f"try translation {cnt} times...")

        # insert car to new location
        position, rz_degree = rand.get_insert_location(ego_info)
        success_flag, ego_id, cp_id = insert.vehicle_insert(ego_info, cp_info, position, True, True, rz_degree, transformation="rotation")

        if success_flag:
            base_ego_id, base_cp_id = baseline.vehicle_translate(ego_info_baseline, cp_info_baseline, car_id, position[:2])
            return True, ego_id, cp_id, base_ego_id, base_cp_id

        cnt += 1


def vehicle_scaling(ego_info, cp_info, ego_info_baseline, cp_info_baseline, car_id):
    """
    Scaling a car of the car_id to a randomly generated scale in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :param ego_info_baseline: ego vehicle info for Baseline
    :param cp_info_baseline: cooperative vehicle info for Baseline
    :param car_id: the car id for scaling
    :return: car id for cut (if needed)
    """
    CLogger.info(f"Background index = {ego_info.bg_index}, scaling vehicle car id = {car_id}")
    scaling_flag = False
    cnt = 1
    ego_corner = ego_info.vehicles_info[car_id]['corner']
    rz_degree = -ego_info.vehicles_info[car_id]['yaw_degree']
    position = list(ego_info.vehicles_info[car_id]['center'][:2])

    # the range of scaling
    min_ratio, max_ratio = 0.9, 1.1

    # delete the car in the original location
    while not scaling_flag:
        if cnt >= 10:
            return False, -1, -1, -1, -1
        scaling_flag = delete.vehicle_delete(ego_info, cp_info, car_id)
        cnt += 1

    # delete successful
    scaling_flag = False
    cnt = 1

    while not scaling_flag:
        # over 10 times， return
        if cnt >= 10:
            return False, -1, -1, -1, -1

        ratio = random.uniform(min_ratio, max_ratio)
        CLogger.info(f"try scaling {cnt} times..., scaling ratio = {ratio}")
        corner_center = np.mean(ego_corner, axis=0)
        vectors = ego_corner - corner_center
        scaled_vector = vectors * ratio
        scaling_box = scaled_vector + corner_center

        scaling_flag, ego_id, cp_id = insert.vehicle_insert(ego_info, cp_info, position, False, True, rz_degree, scaling_box, transformation="rotation")
        if scaling_flag:
            base_ego_id, base_cp_id = baseline.vehicle_scaling(ego_info_baseline, cp_info_baseline, car_id, ratio)
            return True, ego_id, cp_id, base_ego_id, base_cp_id
        cnt += 1


def vehicle_rotation(ego_info, cp_info, ego_info_baseline, cp_info_baseline, car_id):
    """
    Rotation a car of the car_id to a randomly generated yaw degree in the cooperative-detection scene.

    :param ego_info: ego vehicle info
    :param cp_info: cooperative vehicle info
    :param ego_info_baseline: ego vehicle info for Baseline
    :param cp_info_baseline: cooperative vehicle info for Baseline
    :param car_id: the car id for translation
    :return: car id for cut (if needed)
    """
    CLogger.info(f"Background index = {ego_info.bg_index}, rotate vehicle car id = {car_id}")
    success_flag = False
    cnt = 1
    org_degree = -ego_info.vehicles_info[car_id]['yaw_degree']
    position = list(ego_info.vehicles_info[car_id]['center'][:2])
    corner = ego_info.vehicles_info[car_id]['corner']

    # delete the car in the original location
    while not success_flag:
        if cnt >= 10:
            return False, -1, -1, -1, -1
        success_flag = delete.vehicle_delete(ego_info, cp_info, car_id)
        cnt += 1

    # delete successful
    success_flag = False
    cnt = 1

    while not success_flag:
        # over 10 times， return
        if cnt >= 10:
            return False, -1, -1, -1, -1

        rot_degree = rand.get_random_rotation()
        CLogger.info(f"try rotation {cnt} times..., rot degree = {rot_degree}")
        rz_degree = org_degree + rot_degree

        success_flag, ego_id, cp_id = insert.vehicle_insert(ego_info, cp_info, position, False, True, rz_degree, corner, transformation="rotation")

        if success_flag:
            if ego_id == -1:
                baseline_ego_id = random.choice(list(ego_info_baseline.vehicles_info.keys()))
            else:
                baseline_ego_id = ego_id
            if cp_id == -1:
                baseline_cp_id = random.choice(list(cp_info_baseline.vehicles_info.keys()))
            else:
                baseline_cp_id = cp_id
            ego_rz_degree = ego_info.vehicles_info[baseline_ego_id]['yaw_degree'] + rot_degree
            cp_rz_degree = cp_info.vehicles_info[baseline_cp_id]['yaw_degree'] + rot_degree
            base_ego_id, base_cp_id = baseline.vehicle_rotation(ego_info_baseline, cp_info_baseline, car_id, ego_rz_degree, cp_rz_degree)
            return True, ego_id, cp_id, base_ego_id, base_cp_id

        cnt += 1


