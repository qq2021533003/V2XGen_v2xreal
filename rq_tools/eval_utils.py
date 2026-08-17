import os
import shutil

import numpy as np
import torch
import random

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh,
                    left_range=-float('inf'), right_range=float('inf'),
                    gt_object_ids=None):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    right_range : float
        The evaluarion range right bound
    left_range : float
        The evaluation range left bound
    gt_object_ids : list
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]

    # print("before: ", len(gt_object_ids))

    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        score_order_descend = np.argsort(-det_score)
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))
       
        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

            # TODO: pop successfully pred gt box
            gt_object_ids.pop(gt_index)
    else:
        gt = gt_boxes.shape[0]
    # result_stat[iou_thresh]['fp'] += fp
    # result_stat[iou_thresh]['tp'] += tp
    # result_stat[iou_thresh]['gt'] += gt
    result_stat[iou_thresh]['tp'].append(tp)
    result_stat[iou_thresh]['fp'].append(fp)
    result_stat[iou_thresh]['gt'].append(gt)

    # print(len(gt_object_ids))

    return fp, tp, gt, gt_object_ids


def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = iou_5['fp']
    tp = iou_5['tp']
    assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path, range=""):
    dump_dict = {}
    file_name = 'eval.yaml' if range == "" else range + '_eval.yaml'
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    dump_dict.update({'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, file_name))

    # print('The range is %s, '
    #       'The Average Precision at IOU 0.5 is %.3f, '
    #       'The Average Precision at IOU 0.7 is %.3f' % (range, ap_50, ap_70))

    return round(ap_50, 3)


def get_occ_error(ego_gen_param, cp_gen_param, false_pred_ids):
    total_occ = 0
    occ_error = 0
    occ_threshold = 0

    for car_id, param_dict in ego_gen_param.items():
        if param_dict['ego_occlusion_rate'] > occ_threshold:
            # print("occ rate = ", param_dict['ego_occlusion_rate'])
            if car_id in false_pred_ids:
                occ_error += 1
            total_occ += 1

    for car_id, param_dict in cp_gen_param.items():
        # 跳过重复车辆
        if car_id in ego_gen_param.keys():
            continue
        if param_dict['ego_occlusion_rate'] > occ_threshold:
            if car_id in false_pred_ids:
                occ_error += 1
            total_occ += 1

    return occ_error, total_occ


def get_long_distance_error(ego_gen_param, cp_gen_param, false_pred_ids, distance_k=0):
    long_distance_error = 0
    total_long_distance = 0

    for car_id, param_dict in ego_gen_param.items():
        if param_dict['ego_distance'] > distance_k:
            if car_id in false_pred_ids:
                long_distance_error += 1
            total_long_distance += 1

    for car_id, param_dict in cp_gen_param.items():
        # 跳过重复车辆
        if car_id in ego_gen_param.keys():
            continue
        if param_dict['ego_distance'] > distance_k:
            if car_id in false_pred_ids:
                long_distance_error += 1
            total_long_distance += 1

    return long_distance_error, total_long_distance


def method_eval_result(method_stat, result_stat, model_dir, scale, is_save=False, dataset_dir=None, save_path=None, model=None):
    """
    1. Random select method
    2. CooTest select method
    3. V2x-Gen select method
    :param method_stat:
    :param result_stat:
    :param model_dir:
    :param scale:
    :param is_save:
    :param dataset_dir:
    :param save_path:
    :param model:
    :return:
    """
    total_result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                         0.7: {'tp': [], 'fp': [], 'gt': 0},
                         'occ_error': 0,
                         'dis_error': 0,
                         'total_occ': 0,
                         'total_dis': 0,
                         'timestamp': []}
    cootest_result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                           0.7: {'tp': [], 'fp': [], 'gt': 0},
                           'occ_error': 0,
                           'dis_error': 0,
                           'total_occ': 0,
                           'total_dis': 0,
                           'timestamp': []}
    random_result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                          0.7: {'tp': [], 'fp': [], 'gt': 0},
                          'occ_error': 0,
                          'dis_error': 0,
                          'total_occ': 0,
                          'total_dis': 0,
                          'timestamp': []}
    gen_result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                       0.7: {'tp': [], 'fp': [], 'gt': 0},
                       'occ_error': 0,
                       'dis_error': 0,
                       'total_occ': 0,
                       'total_dis': 0,
                       'timestamp': []}

    # scene timestamp intervals
    #  TODO: 重构数据保存逻辑，V2X-Real 的 folder_name 即是 scene
    # intervals = [(0, 147), (147, 261), (261, 405), (405, 603), (603, 783),
    #              (783, 1093), (1093, 1397), (1397, 1618), (1618, 1993)]
    split_scene = True

    num = len(result_stat[0.5]['tp'])   # sum data nums

    # random select
    if not split_scene:
        random_select_indices = sorted(random.sample(range(num), int(num * scale)))
    # else:
    #     random_select_indices = []

    #     # for start, end in intervals:
    #     scene_indices = []
    #     timestamp_list = result_stat['timestamp']

    #     for i, timestamp in enumerate(timestamp_list):
    #         if start <= int(timestamp[1]) < end:
    #             scene_indices.append(i)

    #     select_count = int(len(scene_indices) * scale)
    #     random_select_indices += random.sample(scene_indices, select_count)

    # CooTest select
    cootest_stat_list = method_stat['cootest']
    # Normalized data
    max_param = max(cootest_stat_list)
    min_param = min(cootest_stat_list)
    normalized_params_list = [-(x - min_param) / (max_param - min_param) for x in cootest_stat_list]
    select_number = int(len(normalized_params_list) * scale)

    # V2X-Gen select
    gen_stat_list = method_stat['v2x_gen']

    if split_scene:
        # ================= 按场景划分并筛选 =================
        scene_to_indices = {}
        timestamp_list = result_stat['timestamp']

        for i, ts in enumerate(timestamp_list):
            # ts 格式为 (scene_name, frame_id)
            scene_name = ts[0] 
            
            if scene_name not in scene_to_indices:
                scene_to_indices[scene_name] = []
            scene_to_indices[scene_name].append(i)
        
        # 初始化用于存放最终结果的索引列表
        random_select_indices = []
        cootest_select_indices = []
        gen_select_indices = []

        # 在每个场景内部独立进行筛选
        for scene, indices in scene_to_indices.items():
            # 计算当前场景需要抽取的数据量
            select_count = int(len(indices) * scale)
            
            # --- Random Select ---
            random_select_indices.extend(random.sample(indices, select_count))
            
            # --- CooTest Select ---
            # 根据当前场景内元素的 normalized_params_list 分数降序排列
            scene_coo_sorted = sorted(indices, key=lambda idx: normalized_params_list[idx], reverse=True)
            cootest_select_indices.extend(scene_coo_sorted[:select_count])
            
            # --- V2X-Gen Select ---
            # 根据当前场景内元素的 gen_stat_list 分数降序排列
            scene_gen_sorted = sorted(indices, key=lambda idx: gen_stat_list[idx], reverse=True)
            gen_select_indices.extend(scene_gen_sorted[:select_count])

        print(f"Selected total nums -> Random: {len(random_select_indices)}, CooTest: {len(cootest_select_indices)}, Gen: {len(gen_select_indices)}")


    else:
        cootest_select_indices = sorted(range(len(normalized_params_list)),
                                        key=lambda i: normalized_params_list[i],
                                        reverse=True)[:select_number]
        gen_select_indices = sorted(range(len(gen_stat_list)),
                                    key=lambda i: gen_stat_list[i],
                                    reverse=True)[:select_number]

    # print(len(random_select_indices), len(cootest_select_indices), len(gen_select_indices))

    get_part_list_stat(result_stat, cootest_select_indices, cootest_result_stat)
    get_part_list_stat(result_stat, random_select_indices, random_result_stat)
    get_part_list_stat(result_stat, gen_select_indices, gen_result_stat)
    get_part_list_stat(result_stat, range(num), total_result_stat)

    print("------------------------------------------------------")
    print(f"scale = {scale}")
    print("------------------------------------------------------")
    v2x_select_eval(total_result_stat, model_dir, 'total')
    print("------------------------------------------------------")
    v2x_select_eval(random_result_stat, model_dir, 'random')
    print("------------------------------------------------------")
    v2x_select_eval(cootest_result_stat, model_dir, 'cootest')
    print("------------------------------------------------------")
    v2x_select_eval(gen_result_stat, model_dir, 'gen')
    print("------------------------------------------------------")

    # save result
    if is_save:
        # pass
        save_selected_data_and_label(cootest_result_stat['timestamp'], dataset_dir, save_path, f'coo_test/{scale}/{model}')
        save_selected_data_and_label(random_result_stat['timestamp'], dataset_dir, save_path, f'random/{scale}/{model}')
        save_selected_data_and_label(gen_result_stat['timestamp'], dataset_dir, save_path, f'v2x_gen/{scale}/{model}')


def select_scene_scores(result_stat, score_list, scene_intervals, scale):
    selected_list = []

    for interval in scene_intervals:
        scene_indices = []
        timestamp_list = result_stat['timestamp']

        for i, timestamp in enumerate(timestamp_list):
            if interval[0] <= int(timestamp[1]) < interval[1]:
                scene_indices.append(i)

        if scene_indices:
            sorted_scene_indices = sorted(scene_indices, key=lambda i: score_list[i], reverse=True)

            select_count = int(len(sorted_scene_indices) * scale)

            selected_list.extend(sorted_scene_indices[:select_count])

    return selected_list


def get_part_list_stat(total_state, part_list, part_state):
    for i in part_list:
        part_state[0.5]['tp'] += total_state[0.5]['tp'][i]
        part_state[0.5]['fp'] += total_state[0.5]['fp'][i]
        part_state[0.5]['gt'] += total_state[0.5]['gt'][i]
        part_state['occ_error'] += total_state['occ_error'][i]
        part_state['dis_error'] += total_state['dis_error'][i]
        part_state['total_occ'] += total_state['total_occ'][i]
        part_state['total_dis'] += total_state['total_dis'][i]
        if 'timestamp' in part_state:
            part_state['timestamp'].append(total_state['timestamp'][i])


def save_selected_data_and_label(timestamps, dataset_dir, save_dir, method):
    save_root = os.path.join(save_dir, method)
    # if os.path.exists(save_root):
    #         shutil.rmtree(save_root)
            
    for i, timestamp in enumerate(timestamps):
        ego_pcd_path = f'{dataset_dir}/{timestamp[0]}/0/{timestamp[1]}.bin'
        cp_pcd_path = f'{dataset_dir}/{timestamp[0]}/1/{timestamp[1]}.bin'
        ego_label_path = f'{dataset_dir}/{timestamp[0]}/0/{timestamp[1]}.yaml'
        cp_label_path = f'{dataset_dir}/{timestamp[0]}/1/{timestamp[1]}.yaml'

        save_ego_folder = f'{save_dir}/{method}/{timestamp[0]}/0'
        save_cp_folder = f'{save_dir}/{method}/{timestamp[0]}/1'

        os.makedirs(save_ego_folder, exist_ok=True)
        os.makedirs(save_cp_folder, exist_ok=True)

        save_ego_pcd_path = f'{save_ego_folder}/{i:06d}.bin'
        save_cp_pcd_path = f'{save_cp_folder}/{i:06d}.bin'
        save_ego_label_path = f'{save_ego_folder}/{i:06d}.yaml'
        save_cp_label_path = f'{save_cp_folder}/{i:06d}.yaml'

        shutil.copy(ego_pcd_path, save_ego_pcd_path)
        shutil.copy(cp_pcd_path, save_cp_pcd_path)
        shutil.copy(ego_label_path, save_ego_label_path)
        shutil.copy(cp_label_path, save_cp_label_path)


def v2x_select_eval(result_stat, model_dir, method):
    occ_error = result_stat['occ_error']
    long_dis_error = result_stat['dis_error']
    occ_error_rate = occ_error / result_stat['total_occ']
    long_dis_error_rate = long_dis_error / result_stat['total_dis']

    print(f"method = {method}, model = {model_dir}")
    print(f"ap_50 = {eval_final_results(result_stat, model_dir)}\n"
          f"occ error = {occ_error}, occ error rate = {occ_error_rate}\n"
          f"long dis error = {long_dis_error}, long dis error rate = {long_dis_error_rate}")


def v2x_eval_result(result_stat, model_dir, method):
    total_result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                         0.7: {'tp': [], 'fp': [], 'gt': 0},
                         'occ_error': 0,
                         'dis_error': 0,
                         'total_occ': 0,
                         'total_dis': 0}
    num = len(result_stat[0.5]['tp'])
    get_part_list_stat(result_stat, range(num), total_result_stat)

    occ_error = total_result_stat['occ_error']
    long_dis_error = total_result_stat['dis_error']
    occ_error_rate = occ_error / total_result_stat['total_occ']
    long_dis_error_rate = long_dis_error / total_result_stat['total_dis']

    print("------------------------------------------------------")
    print(f"method = {method}, model = {model_dir}")
    print("------------------------------------------------------")
    print(f"ap_50 = {eval_final_results(total_result_stat, model_dir)}\n"
          f"occ error = {occ_error}, occ error rate = {occ_error_rate}\n"
          f"long dis error = {long_dis_error}, long dis error rate = {long_dis_error_rate}")
    print("------------------------------------------------------")


def V2X_Gen_method(ego_gen_param, cp_gen_param, false_pred_ids, a=0.5, b=0.5):
    fop = 0  # Fop: occlusion perceptual error
    flp = 0  # Flp: long distance perceptual error
    MAX_LIDAR_RANGE = 200

    # TODO: 将有协同关系的 cp 视角下的遮挡率进行调整
    for car_id, param_dict in cp_gen_param.items():
        if car_id in ego_gen_param.keys():
            cp_gen_param[car_id] = {
                "ego_occlusion_rate": 0,
                "cp_occlusion_rate": param_dict['cp_occlusion_rate'],
                "ego_distance": 0,
                "cp_distance": param_dict['cp_distance'],
                "timestamp": param_dict['timestamp'],
                "folder_name": param_dict['folder_name']
            }

    # traversal ego param
    for car_id, param_dict in ego_gen_param.items():
        if car_id in false_pred_ids:
            flp += min(param_dict['ego_distance'], MAX_LIDAR_RANGE) / MAX_LIDAR_RANGE * \
                   (1 - min(param_dict['cp_distance'], MAX_LIDAR_RANGE) / MAX_LIDAR_RANGE)

            if car_id in list(cp_gen_param.keys()):
                fop += param_dict['ego_occlusion_rate'] * (1 - cp_gen_param[car_id]['cp_occlusion_rate'])
            else:
                fop += param_dict['ego_occlusion_rate'] * (1 - param_dict['cp_occlusion_rate'])

    # traversal cooperative vehicles
    for car_id, param_dict in cp_gen_param.items():
        # only cooperative vehicle
        if car_id in false_pred_ids and car_id not in list(ego_gen_param.keys()):
            fop += param_dict['ego_occlusion_rate'] * (1 - param_dict['cp_occlusion_rate'])

            flp += min(param_dict['ego_distance'], MAX_LIDAR_RANGE) / MAX_LIDAR_RANGE * \
                (1 - min(param_dict['cp_distance'], MAX_LIDAR_RANGE) / MAX_LIDAR_RANGE)

    select_method_param = a * fop + b * flp

    return select_method_param, fop, flp


def CooTest_method_result(det_boxes, det_score, pred_boxes):
    """
    CooTest data select method
    :param det_boxes:
    :param det_score:
    :param pred_boxes:
    :return:
    """
    if det_boxes is not None and pred_boxes is not None:
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        pred_boxes = common_utils.torch_tensor_to_numpy(pred_boxes)

        pred_polygon_list = list(common_utils.convert_format(pred_boxes))
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        det_score_list = np.array(det_score).tolist()

        select_param_list = []
        det_boxes_volumes = []

        # calculate det boxes volume
        for box in det_boxes:
            box_lengths = []
            for i in range(3):
                lengths = box[:, i].max(axis=0) - box[:, i].min(axis=0)
                box_lengths.append(lengths)
            box_volume = np.prod(box_lengths)
            det_boxes_volumes.append(box_volume)

        for i, det_box in enumerate(det_boxes):
            overlap_volume = common_utils.compute_intersection_volume(det_box, pred_boxes)

            # guide method
            select_param = (overlap_volume * det_score_list[i]) / \
                           (len(det_polygon_list) * len(pred_polygon_list) * det_boxes_volumes[i])

            select_param_list.append(select_param)

    else:
        # print("boxes list is empty!")
        return 0.0

    # print(select_param_list)

    return sum(select_param_list)