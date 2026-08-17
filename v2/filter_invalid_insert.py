import argparse
import statistics
import sys
import os

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.rq_eval.rq2_data_select import CooTest_method_result, V2X_Gen_method
from opencood.rq_eval.v2x_gen_utils import save_box_tensor, load_box_tensor, get_valid_param_dict, get_total_occ_and_dis

from config.common_config import VEHICLE_CLASS_ID
from utils.visual import visualize_individual_perspectives, calculate_max_iou_for_targets, filter_boxes_by_ids
import opencood.utils.common_utils as common_utils


#TODO
# 1. 对已生成的数据检测是否是合理插入
# 2. 删除非合理插入的数据
# 3. 循环执行筛选操作，直至生成足够数量的数据
# 4. 如果数据多于需要的，则随机从中选择定量数据，保存为新的数据集，并重新命名


def valid_insert_detection(dataset_dir, model_dir):
    model_path = os.path.join(model_dir, 'late_fusion/config.yaml')

    # 使用 late fusion 进行推理
    hypes = yaml_utils.load_yaml(model_path, None)

    hypes['validate_dir'] = dataset_dir
    hypes['dataset_mode'] = 'v2v'

    print('Dataset Building')  # 加载数据集

    opencood_dataset = build_dataset(hypes, visualize=True, train=False)

    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')  # 模型创建
    model = train_utils.create_model(hypes)  # 根据配置文件构建模型架构，并将其移动到 GPU
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # 用于全局统计新插入物体的变换质量
    global_transform_stats = {
        'total_inserted_objects': 0,  # 所有帧的新插入物体总数
        'valid_transforms': 0,  # 满足条件的变换总数
        'per_frame_stats': []  # 每帧的详细统计
    }

    # ========== 单车推理模式 (NO FUSION) ==========
    # 功能：分别在 ego 和 cv 视角下对新插入物体进行 IoU 评估
    # 逻辑：
    #   1. 对 ego 和 cp 两端分别进行独立的目标检测
    #   2. 根据 inserted_ids 筛选出新插入的物体
    #   3. 计算每个新插入物体在预测框中的最大 IoU
    #   4. 判断标准：当 ego 和 cp 任意一方的 IoU >= 0.5 时，视为一次合理的变换
    #   5. 统计全局和每帧的合理变换比例
    #
    #
    #   - 允许 ego 和 cp 两端的 inserted_ids 不一致（某些物体可能只在一端被标记）
    #   - 使用 ID 并集作为总的新插入物体集合
    #   - 对于某端缺失的物体，该端 IoU 设为 0.0
    # =============================================
    for i, batch_data in enumerate(data_loader):
        print('data idx =', i)

        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)

            det_box_tensor, det_score, gt_box_tensor, gt_label_tensor, gt_object_ids = \
                    inference_utils.inference_no_fusion(batch_data, model, opencood_dataset)
                
            det_box_tensor_cp, det_score_cp, gt_box_tensor_cp, gt_label_tensor_cp, gt_object_ids_cp = \
                inference_utils.inference_no_fusion_cp(batch_data, model, opencood_dataset)
            
            # 处理 CP 端无数据的情况
            if det_box_tensor_cp is None:
                det_box_tensor_cp = torch.zeros((0, 8, 3), device=device)
                det_score_cp = torch.zeros((0,), device=device)
                gt_box_tensor_cp = torch.zeros((0, 8, 3), device=device)
                gt_object_ids_cp = []

            ego_ins_ids = batch_data['ego']['inserted_ids']
            cp_ins_ids = batch_data['1']['inserted_ids']

            # 只关注类别为 vehicle (key = 1) 的对象
            det_box_tensor[det_score[:, -1] == VEHICLE_CLASS_ID]
            gt_box_tensor[gt_label_tensor == VEHICLE_CLASS_ID]

            det_box_tensor_cp[det_score_cp[:, -1] == VEHICLE_CLASS_ID]
            gt_box_tensor_cp[gt_label_tensor_cp == VEHICLE_CLASS_ID]

            # 筛选新插入物体的 GT框
            ego_target_gt_boxes = filter_boxes_by_ids(gt_box_tensor, gt_object_ids, ego_ins_ids)
            cp_target_gt_boxes = filter_boxes_by_ids(gt_box_tensor_cp, gt_object_ids_cp, cp_ins_ids)

            # 计算 IoU - 针对每个新插入的 GT 物体，找到预测框中的最大 IoU
            ego_ious = calculate_max_iou_for_targets(det_box_tensor, ego_target_gt_boxes)
            cp_ious = calculate_max_iou_for_targets(det_box_tensor_cp, cp_target_gt_boxes)

            print(f"Ego inserted IDs: {ego_ins_ids}, CP inserted IDs: {cp_ins_ids}")
            print(f"Ego IoUs: {ego_ious}")
            print(f"CP IoUs: {cp_ious}")

            # ========== V2X-Real 数据集直接采用全局 vehicle_id 进行合并 ==========
            merged_inserted_objects = []
            global_id_to_obj = {}

            for vehicle_id in ego_ins_ids:
                obj_info = {
                    'unique_id': vehicle_id,
                    'ego_exists': True,
                    'cp_exists': False
                }
                merged_inserted_objects.append(obj_info)
                global_id_to_obj[vehicle_id] = obj_info

            for vehicle_id in cp_ins_ids:
                if vehicle_id in global_id_to_obj:
                    # 已存在，进行合并
                    global_id_to_obj[vehicle_id]['cp_exists'] = True
                else:
                    obj_info = {
                    'unique_id': vehicle_id,
                    'ego_exists': False,
                    'cp_exists': True
                }
                merged_inserted_objects.append(obj_info)
                global_id_to_obj[vehicle_id] = obj_info
            
            total_inserted_objects = len(merged_inserted_objects)
            
            print(f"\nMerged objects count: {total_inserted_objects}")
            for obj in merged_inserted_objects:
                print(f"  Unique: {obj['unique_id']}, Ego: {obj['ego_id']}, CP: {obj['cp_id']}")

            if total_inserted_objects == 0:
                print(f"\n=== Frame {i} Summary ===")
                print(f"新插入物体总数：0 (跳过)")
                print("========================\n")
                continue

            valid_transform_count = 0
            transform_details = []  # 记录每个物体的详细变换信息

            # 遍历每个合并后的新插入物体
            for obj_info in merged_inserted_objects:
                ego_id = obj_info['ego_id']
                cp_id = obj_info['cp_id']

                # 获取该物体在 ego 端的 IoU
                if obj_info['ego_exists'] and ego_id is not None:
                    if ego_id in ego_ins_ids:
                        ego_idx = ego_ins_ids.index(ego_id)
                        iou_e = ego_ious[ego_idx] if ego_idx < len(ego_ious) else 0.0
                    else:
                        iou_e = 0.0
                else:
                    iou_e = 0.0  # ego 端没有该物体

                # 获取该物体在 cp 端的 IoU
                if obj_info['cp_exists'] and cp_id is not None:
                    if cp_id in cp_ins_ids:
                        cp_idx = cp_ins_ids.index(cp_id)
                        iou_c = cp_ious[cp_idx] if cp_idx < len(cp_ious) else 0.0
                    else:
                        iou_c = 0.0
                else:
                    iou_c = 0.0  # cp 端没有该物体

                # 判断条件：任意一端 IoU >= 0.5 即视为合理变换
                is_valid = (iou_e >= 0.5 or iou_c >= 0.5)
                if is_valid:
                    valid_transform_count += 1

                # 记录详细信息
                transform_details.append({
                    'unique_id': obj_info['unique_id'],
                    'ego_id': ego_id,
                    'cp_id': cp_id,
                    'ego_iou': iou_e,
                    'cp_iou': iou_c,
                    'max_iou': max(iou_e, iou_c),
                    'is_valid': is_valid,
                    'ego_exists': obj_info['ego_exists'],
                    'cp_exists': obj_info['cp_exists']
                })

                print(f"  Unique ID {obj_info['unique_id']}: "
                      f"Ego ID={ego_id} ({'✓' if obj_info['ego_exists'] else '✗'}), IoU={iou_e:.4f} | "
                      f"CP ID={cp_id} ({'✓' if obj_info['cp_exists'] else '✗'}), IoU={iou_c:.4f} | "
                      f"Max={max(iou_e, iou_c):.4f}, Valid={is_valid}")
            print(f"\n=== Frame {i} Summary ===")
            print(f"新插入物体总数：{total_inserted_objects}")
            print(f"满足条件 (Max(IoU) >= 0.5) 的数量：{valid_transform_count}")
            if total_inserted_objects > 0:
                print(f"合理变换比例：{valid_transform_count / total_inserted_objects:.2%}")
            print("========================\n")

            # 累积到全局统计
            global_transform_stats['total_inserted_objects'] += total_inserted_objects
            global_transform_stats['valid_transforms'] += valid_transform_count
            global_transform_stats['per_frame_stats'].append({
                'frame_idx': i,
                'total_inserted': total_inserted_objects,
                'valid_transforms': valid_transform_count,
                'details': transform_details
            })
            # # 可视化
            # visualize_individual_perspectives(batch_data, ego_results, cp_results, i)

        # 预测结果分开保存
        det_save_path = "/home/zyc/code/V2XGen/rq2/rq2_det_box"
        det_save_path_cp = "/home/zyc/code/V2XGen/rq2/rq2_det_box_cp"
        if det_box_tensor is not None:
            save_box_tensor(det_box_tensor, det_score, i, det_save_path)
        if det_box_tensor_cp is not None:
            save_box_tensor(det_box_tensor_cp, det_score_cp, i, det_save_path_cp)

    # 输出全局统计结果
    print("\n" + "=" * 60)
    print("=== NOFUSION 全局变换质量评估 ===")
    print("=" * 60)
    print(f"总帧数：{len(global_transform_stats['per_frame_stats'])}")
    print(f"新插入物体总数：{global_transform_stats['total_inserted_objects']}")
    print(f"满足条件的变换数 (Max(IoU_ego, IoU_cp) >= 0.5): {global_transform_stats['valid_transforms']}")

    if global_transform_stats['total_inserted_objects'] > 0:
        global_valid_ratio = global_transform_stats['valid_transforms'] / global_transform_stats[
            'total_inserted_objects']
        print(f"全局合理变换比例：{global_valid_ratio:.2%}")

        # 计算每帧的平均合理变换比例
        per_frame_ratios = []
        for frame_stat in global_transform_stats['per_frame_stats']:
            if frame_stat['total_inserted'] > 0:
                ratio = frame_stat['valid_transforms'] / frame_stat['total_inserted']
                per_frame_ratios.append(ratio)

        if len(per_frame_ratios) > 0:
            avg_per_frame_ratio = sum(per_frame_ratios) / len(per_frame_ratios)
            print(f"平均每帧合理变换比例：{avg_per_frame_ratio:.2%}")
            print(f"最高单帧合理变换比例：{max(per_frame_ratios):.2%}")
            print(f"最低单帧合理变换比例：{min(per_frame_ratios):.2%}")
    else:
        print("警告：没有检测到任何新插入的物体！")

    print("=" * 60 + "\n")


