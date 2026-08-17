import os
import shutil

import numpy as np
import torch
import random

from temp_storage.opencood.utils import common_utils
from temp_storage.opencood.hypes_yaml import yaml_utils


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

    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        det_polygon_list_origin = list(common_utils.convert_format(det_boxes))
        gt_polygon_list_origin = list(common_utils.convert_format(gt_boxes))
        det_polygon_list = []
        gt_polygon_list = []
        det_score_new = []
        # remove the bbx out of range
        for i in range(len(det_polygon_list_origin)):
            det_polygon = det_polygon_list_origin[i]
            distance = np.sqrt(det_polygon.centroid.x ** 2 +
                               det_polygon.centroid.y ** 2)
            if left_range < distance < right_range:
                det_polygon_list.append(det_polygon)
                det_score_new.append(det_score[i])

        for i in range(len(gt_polygon_list_origin)):
            gt_polygon = gt_polygon_list_origin[i]
            distance = np.sqrt(gt_polygon.centroid.x ** 2 +
                               gt_polygon.centroid.y ** 2)
            if left_range < distance < right_range:
                gt_polygon_list.append(gt_polygon)

        gt = len(gt_polygon_list)
        det_score_new = np.array(det_score_new)
        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score_new)

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
            if car_id in false_pred_ids:
                occ_error += 1
            total_occ += 1

    for car_id, param_dict in cp_gen_param.items():
        if param_dict['ego_occlusion_rate'] > occ_threshold:
            if car_id in false_pred_ids:
                occ_error += 1
            total_occ += 1

    return occ_error, total_occ


def get_long_distance_error(ego_gen_param, cp_gen_param, false_pred_ids, distance_k=50):
    long_distance_error = 0
    total_long_distance = 0

    for car_id, param_dict in ego_gen_param.items():
        if param_dict['ego_distance'] > distance_k:
            if car_id in false_pred_ids:
                long_distance_error += 1
            total_long_distance += 1

    for car_id, param_dict in cp_gen_param.items():
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
    intervals = [(0, 147), (147, 261), (261, 405), (405, 603), (603, 783),
                 (783, 1093), (1093, 1397), (1397, 1618), (1618, 1993)]
    split_scene = False

    num = len(result_stat[0.5]['tp'])   # sum data nums

    # random select
    if not split_scene:
        random_select_indices = sorted(random.sample(range(num), int(num * scale)))
    else:
        random_select_indices = []
        for start, end in intervals:
            scene_indices = []
            timestamp_list = result_stat['timestamp']

            for i, timestamp in enumerate(timestamp_list):
                if start <= int(timestamp[1]) < end:
                    scene_indices.append(i)

            select_count = int(len(scene_indices) * scale)
            random_select_indices += random.sample(scene_indices, select_count)

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
        cootest_select_indices = select_scene_scores(result_stat, normalized_params_list, intervals, scale)
        gen_select_indices = select_scene_scores(result_stat, gen_stat_list, intervals, scale)
    else:
        cootest_select_indices = sorted(range(len(normalized_params_list)),
                                        key=lambda i: normalized_params_list[i],
                                        reverse=True)[:select_number]
        gen_select_indices = sorted(range(len(gen_stat_list)),
                                    key=lambda i: gen_stat_list[i],
                                    reverse=True)[:select_number]

    print(len(random_select_indices), len(cootest_select_indices), len(gen_select_indices))

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
        pass
        save_selected_data_and_label(cootest_result_stat['timestamp'], dataset_dir, save_path, f'coo_test/{scale}/{model}/select')
        save_selected_data_and_label(random_result_stat['timestamp'], dataset_dir, save_path, f'random/{scale}/{model}/select')
        save_selected_data_and_label(gen_result_stat['timestamp'], dataset_dir, save_path, f'v2x_gen/{scale}/{model}/select')


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
    for i, timestamp in enumerate(timestamps):
        ego_pcd_path = f'{dataset_dir}/{timestamp[0]}/0/{timestamp[1]}.pcd'
        cp_pcd_path = f'{dataset_dir}/{timestamp[0]}/1/{timestamp[1]}.pcd'
        ego_label_path = f'{dataset_dir}/{timestamp[0]}/0/{timestamp[1]}.yaml'
        cp_label_path = f'{dataset_dir}/{timestamp[0]}/1/{timestamp[1]}.yaml'

        save_ego_folder = f'{save_dir}/{method}/0'
        save_cp_folder = f'{save_dir}/{method}/1'

        if not os.path.exists(save_ego_folder):
            os.makedirs(save_ego_folder)

        if not os.path.exists(save_cp_folder):
            os.makedirs(save_cp_folder)

        save_ego_pcd_path = f'{save_ego_folder}/{i:06d}.pcd'
        save_cp_pcd_path = f'{save_cp_folder}/{i:06d}.pcd'
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
