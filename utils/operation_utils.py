import math
import copy
import random
import numpy as np
import core.obj_delete as delete
import core.obj_insert as insert
import utils.random_param as rand
import utils.baseline_utils as baseline

from core.occlusion_treatment import get_occ_rate_of_obj
from utils.common_utils import center_system_transform, rz_degree_system_transform
from logger import CLogger

# 组合基础变换操作为五种变换算子
# 变换操作执行成功后，保留可视化文件

def vehicle_insert(ego_info, cp_info, is_vis=False, is_gen_baseline=False):
    """
    在协同检测场景随机生成的位置插入一辆车
    ---------------
    :param ego_info:
    :param cp_info:
    :is_gen_baseline: 是否需要生成 Baseline 数据
    :return:
    """
    success_flag = False
    count = 1

    if is_gen_baseline:
        ego_info_baseline = copy.deepcopy(ego_info)
        cp_info_baseline = copy.deepcopy(cp_info)
        ego_info_baseline.pc = ego_info_baseline.pc[:, :3]
        cp_info_baseline.pc = cp_info_baseline.pc[:, :3]

    while not success_flag:
        # over 10 times
        if count >= 10:
            return False

        CLogger.info(f"try insert {count} times...")

        position, _ = rand.get_insert_location(ego_info)

        if not position:
            continue

        success_flag, ego_id, cp_id = insert.vehicle_insert(ego_info=ego_info, 
                                                   cp_info=cp_info, 
                                                   position=position, 
                                                   detection_flag=True, 
                                                   is_vis=is_vis,
                                                   transformation="insert")
        count += 1
    
    if success_flag and is_gen_baseline:
        base_ego_id, base_cp_id = baseline.vehicle_insert(ego_info_baseline, cp_info_baseline)
        
        ego_info_baseline.save_data_and_label("baseline")
        cp_info_baseline.save_data_and_label("baseline")
        return True, ego_id, cp_id, base_ego_id, base_cp_id

    return True

def vehicle_delete(ego_info, cp_info, car_id=0, is_vis=False, is_gen_baseline=False):
    """
    删除 car_id 对应的车辆
    :param ego_info:
    :param cp_info:
    :param car_id: 要删除的车辆 id
    :return:
    """
    if is_gen_baseline:
        ego_info_baseline = copy.deepcopy(ego_info)
        cp_info_baseline = copy.deepcopy(cp_info)
        ego_info_baseline.pc = ego_info_baseline.pc[:, :3]
        cp_info_baseline.pc = cp_info_baseline.pc[:, :3]


    # CLogger.info(f"background index = {ego_info.bg_index}, delete ego vehicle id = {car_id}")

    # ego_center, cp_center 用于裁剪数据
    success_flag, v2x_ego_center, v2x_cp_center = \
        delete.vehicle_delete(ego_info=ego_info, 
                              cp_info=cp_info, 
                              car_id=car_id, 
                              is_vis=is_vis,
                              transformation="delete")
    
    # 成功后进行 Baseline 操作
    if success_flag:
        if is_gen_baseline:
            baseline_ego_center, baseline_cp_center = \
                baseline.vehicle_delete(ego_info_baseline, cp_info_baseline, car_id)
            ego_info_baseline.save_data_and_label("baseline")
            cp_info_baseline.save_data_and_label("baseline")
        return True
        

    return False


def vehicle_translation(ego_info, cp_info, car_id, is_vis=False, is_gen_baseline=False):
    """
    平移 car_id 的车辆
    :param ego_info:
    :param cp_info:
    :param car_id:
    :return:
    """
    old_corner = np.asarray(ego_info.vehicles_info[car_id]["corner"]).copy()
    if is_gen_baseline:
        ego_info_baseline = copy.deepcopy(ego_info)
        cp_info_baseline = copy.deepcopy(cp_info)
        ego_info_baseline.pc = ego_info_baseline.pc[:, :3]
        cp_info_baseline.pc = cp_info_baseline.pc[:, :3]

    success_flag = False
    cnt = 1

    # 删除原来位置的车辆 (不能删除协同车)
    while not success_flag:
        if cnt >= 10:
            return False
        success_flag = delete.vehicle_delete(ego_info=ego_info, 
                                             cp_info=cp_info, 
                                             car_id=car_id, 
                                             is_vis=is_vis,
                                             transformation="translation")
        cnt += 1

    # 删除成功
    success_flag = False
    cnt = 1

    # 多次插入尝试，直至成功
    while not success_flag:
        # over 10 times
        if cnt >= 10:
            return False

        # CLogger.info(f"try translation {cnt} times...")
        # 插入车辆到新位置
        position, rz_degree = rand.get_insert_location(ego_info)
        success_flag, ego_id, cp_id = insert.vehicle_insert(ego_info=ego_info, 
                                                    cp_info=cp_info,
                                                    position=position, 
                                                    detection_flag=True, 
                                                    gt_flag=True,   
                                                    gt_degree=rz_degree,
                                                    is_vis=is_vis,
                                                    transformation="translation",
                                                    vis_corner=old_corner
                                                )
        
        if success_flag and is_gen_baseline:
            base_ego_id, base_cp_id = baseline.vehicle_translate(ego_info_baseline, cp_info_baseline, car_id, position[:2])
            ego_info_baseline.save_data_and_label("baseline")
            cp_info_baseline.save_data_and_label("baseline")
            
            return True, ego_id, cp_id, base_ego_id, base_cp_id

        # 平移失败，再次尝试
        cnt += 1
    return True


def vehicle_scaling(ego_info, cp_info, car_id, is_vis=False, is_gen_baseline=False):
    """
    放缩
    :param ego_info:
    :param cp_info:
    :param car_id:
    :return:
    """
    if is_gen_baseline:
        ego_info_baseline = copy.deepcopy(ego_info)
        cp_info_baseline = copy.deepcopy(cp_info)
        ego_info_baseline.pc = ego_info_baseline.pc[:, :3]
        cp_info_baseline.pc = cp_info_baseline.pc[:, :3]

    scaling_flag = False
    cnt = 1
    ego_corner = ego_info.vehicles_info[car_id]['corner']
    rz_degree = -ego_info.vehicles_info[car_id]['yaw_degree']
    position = list(ego_info.vehicles_info[car_id]['center'][:2])
    # min_ratio, max_ratio = 0.8, 1.2
    min_ratio, max_ratio = 0.9, 1.1

    # ego_center = ego_info.vehicles_info[car_id]['center']

    # 删除原来位置的车辆 (不能删除协同车)
    while not scaling_flag:
        if cnt >= 10:
            return False
        scaling_flag = delete.vehicle_delete(ego_info=ego_info, 
                                             cp_info=cp_info, 
                                             car_id=car_id,
                                             is_vis=is_vis,
                                             transformation="scaling")
        cnt += 1

    # 删除成功
    scaling_flag = False
    cnt = 1

    while not scaling_flag:
        # over 10 times
        # over 10 times
        if cnt >= 10:
            return False

        ratio = random.uniform(min_ratio, max_ratio)
        # ratio = 0.7
        # CLogger.info(f"try scaling {cnt} times..., scaling ratio = {ratio}")
        # print("ratio = ", ratio)
        corner_center = np.mean(ego_corner, axis=0)
        vectors = ego_corner - corner_center
        scaled_vector = vectors * ratio
        scaling_box = scaled_vector + corner_center

        scaling_flag, _, _ = insert.vehicle_insert(ego_info=ego_info, 
                                                            cp_info=cp_info, 
                                                            position=position, 
                                                            detection_flag=False, 
                                                            gt_flag=True, 
                                                            gt_degree=rz_degree, 
                                                            gt_box=scaling_box, 
                                                            is_vis=is_vis,
                                                            transformation="scaling")
        
        if scaling_flag and is_gen_baseline:
            base_ego_id, base_cp_id = baseline.vehicle_scaling(ego_info_baseline, cp_info_baseline, car_id, ratio)
            ego_info_baseline.save_data_and_label("baseline")
            cp_info_baseline.save_data_and_label("baseline")
        cnt += 1
    return True


def vehicle_rotation(ego_info, cp_info, car_id, is_vis=False, is_gen_baseline=False):
    """
    旋转操作
    :param ego_info:
    :param cp_info:
    :param car_id:
    :return:
    """
    if is_gen_baseline:
        ego_info_baseline = copy.deepcopy(ego_info)
        cp_info_baseline = copy.deepcopy(cp_info)
        ego_info_baseline.pc = ego_info_baseline.pc[:, :3]
        cp_info_baseline.pc = cp_info_baseline.pc[:, :3]

    # CLogger.info(f"Background index = {ego_info.bg_index}, rotate vehicle car id = {car_id}")
    success_flag = False
    cnt = 1
    org_degree = -ego_info.vehicles_info[car_id]['yaw_degree']
    position = list(ego_info.vehicles_info[car_id]['center'][:2])
    corner = ego_info.vehicles_info[car_id]['corner']

    # 删除原来位置的车辆 (不能删除协同车)
    while not success_flag:
        if cnt >= 10:
            return False
        success_flag = delete.vehicle_delete(ego_info=ego_info, 
                                             cp_info=cp_info, 
                                             car_id=car_id,
                                             is_vis=is_vis,
                                             transformation="rotation")
        cnt += 1

    # 删除成功
    success_flag = False
    cnt = 1

    while not success_flag:
        # over 10 times
        if cnt >= 10:
            return False

        rot_degree = rand.get_random_rotation()
        # CLogger.info(f"try rotation {cnt} times..., rot degree = {rot_degree}")
        rz_degree = org_degree + rot_degree

        success_flag, ego_id, cp_id = insert.vehicle_insert(ego_info=ego_info, 
                                                   cp_info=cp_info, 
                                                   position=position, 
                                                   detection_flag=False, 
                                                   gt_flag=True, 
                                                   gt_degree=rz_degree, 
                                                   gt_box=corner, 
                                                   is_vis=is_vis,
                                                   transformation="rotation")
        
        if success_flag and is_gen_baseline:
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
            
            ego_info_baseline.save_data_and_label("baseline")
            cp_info_baseline.save_data_and_label("baseline")
            
            return True, ego_id, cp_id, base_ego_id, base_cp_id

        cnt += 1

    return True


def label_complete_for_ego(ego_info, cp_info):
    """
    扫描 ego 标签中的每一辆车,计算遮挡率和距离,保存在标签中
    """
    #  TODO：解决遮挡率计算的问题！！！
    # 1. 看看 ego 是否排除本身
    # 2. 看看是不是距离计算有问题

    # 在局部坐标系下进行计算，计算 "Car" 的遮挡率
    # for vehicle_id, vehicle_info in ego_info.vehicles_info.items():
    for vehicle_id, vehicle_info in ego_info.objs_info.items():
        ego_center = vehicle_info["center"]
        ego2cp_center = center_system_transform(ego_center, ego_info.lidar_pose, cp_info.lidar_pose)
        ego_degree = vehicle_info["yaw_degree"]

        ego_center_xy = ego_center[:2]
        cp_center_xy = ego2cp_center[:2]

        # 保存被遮挡率和距离
        occ_rate = get_occ_rate_of_obj(ego_info, ego_center, ego_degree, vehicle_id)

        # print(ego_info.param["vehicles"].keys(), vehicle_id)
        ego_info.param["vehicles"][vehicle_id]["ego_occ_rate"] = round(float(occ_rate), 2)

        # ego_distance, cp_distance for ego car
        # 1. delete distance key
        if "distance" in ego_info.param["vehicles"][vehicle_id]:
            ego_info.param["vehicles"][vehicle_id].pop("distance")

        # 2. add ego_distance and cp_distance
        ego_info.param["vehicles"][vehicle_id]["ego_distance"] = round(math.sqrt(ego_center_xy[0]**2 + ego_center_xy[1]**2), 2)
        ego_info.param["vehicles"][vehicle_id]["cp_distance"] = round(math.sqrt(cp_center_xy[0]**2 + cp_center_xy[1]**2), 2)

        if ego_info.param["vehicles"][vehicle_id]["ego_distance"] > 150:
            print("long distance: ", ego_center_xy)

        if occ_rate != 0:
            print("ego = ", ego_info.bg_index, occ_rate, ego_center_xy)

    # 将协同数据变换到当前局部坐标系下进行计算
    for vehicle_id, vehicle_info in cp_info.objs_info.items():
        # 排除已经计算的
        if vehicle_id in ego_info.objs_info.keys():
            cp_info.param["vehicles"][vehicle_id]["ego_occ_rate"] = 0
            continue

        cp_center = vehicle_info["center"]
        cp2ego_center = center_system_transform(cp_center, cp_info.lidar_pose, ego_info.lidar_pose)
        cp_degree = vehicle_info["yaw_degree"]
        cp2ego_degree = rz_degree_system_transform(cp_degree, cp_info.lidar_pose, ego_info.lidar_pose)

        occ_rate = get_occ_rate_of_obj(ego_info, cp2ego_center, cp2ego_degree, vehicle_id)
        cp_info.param["vehicles"][vehicle_id]["ego_occ_rate"] = round(float(occ_rate), 2)


def label_complete_for_cp(ego_info, cp_info):
    """
    扫描 cp 标签中的每一辆车,计算遮挡率和距离,保存在标签中
    """
    # 在协同坐标系下计算
    for vehicle_id, vehicle_info in cp_info.objs_info.items():
        cp_center = vehicle_info["center"]
        cp2ego_center = center_system_transform(cp_center, cp_info.lidar_pose, ego_info.lidar_pose)
        cp_degree = vehicle_info["yaw_degree"]

        cp_center_xy = cp_center[:2]
        ego_center_xy = cp2ego_center[:2]

        occ_rate = get_occ_rate_of_obj(cp_info, cp_center, cp_degree, vehicle_id)
        cp_info.param["vehicles"][vehicle_id]["cp_occ_rate"] = round(float(occ_rate), 2)

        # add ego_distance and cp_distance
        cp_info.param["vehicles"][vehicle_id]["ego_distance"] = round(math.sqrt(ego_center_xy[0]**2 + ego_center_xy[1]**2), 2)
        cp_info.param["vehicles"][vehicle_id]["cp_distance"] = round(math.sqrt(cp_center_xy[0]**2 + cp_center_xy[1]**2), 2)


    # 将 ego 数据变换到协同局部坐标系下计算
    for vehicle_id, vehicle_info in ego_info.objs_info.items():
        if vehicle_id in cp_info.objs_info.keys():
            ego_info.param["vehicles"][vehicle_id]["cp_occ_rate"] = 0
            continue

        ego_center = vehicle_info["center"]
        ego2cp_center = center_system_transform(ego_center, ego_info.lidar_pose, cp_info.lidar_pose)
        ego_degree = vehicle_info["yaw_degree"]
        ego2cp_degree = rz_degree_system_transform(ego_degree, ego_info.lidar_pose, cp_info.lidar_pose)

        occ_rate = get_occ_rate_of_obj(cp_info, ego2cp_center, ego2cp_degree, vehicle_id)
        ego_info.param["vehicles"][vehicle_id]["cp_occ_rate"] = round(float(occ_rate), 2)


