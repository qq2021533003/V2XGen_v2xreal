import os
import torch
import shutil
import numpy as np
import random
import opencood.hypes_yaml.yaml_utils as yaml_utils
import opencood.utils.common_utils as common_utils
import utils.operation_utils as op

from torch.utils.data import DataLoader
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from utils.valid_eval import filter_boxes_by_ids, calculate_max_iou_for_targets
from config.dataset_config import DatasetConfig
from config.common_config import VEHICLE_CLASS_ID
from data_utils.v2x_dataset import V2XDataset


def evaluate_frame_validity(model, opencood_dataset, batch_data, device):
    """
    评估单帧的有效性
    """
    with torch.no_grad():
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
        if '1' in batch_data:
            cp_ins_ids = batch_data['1']['inserted_ids']
        else:
            cp_ins_ids = []

        # 只关注类别为 vehicle (key = 1) 的对象
        det_box_tensor[det_score[:, -1] == VEHICLE_CLASS_ID]
        gt_box_tensor = gt_box_tensor[gt_label_tensor == VEHICLE_CLASS_ID]

        if '1' in batch_data:
            det_box_tensor_cp[det_score_cp[:, -1] == VEHICLE_CLASS_ID]
            gt_box_tensor_cp = gt_box_tensor_cp[gt_label_tensor_cp == VEHICLE_CLASS_ID]

        # 筛选新插入物体的 GT框
        ego_target_gt_boxes = filter_boxes_by_ids(gt_box_tensor, gt_object_ids, ego_ins_ids)
        cp_target_gt_boxes = filter_boxes_by_ids(gt_box_tensor_cp, gt_object_ids_cp, cp_ins_ids)

        # 计算 IoU - 针对每个新插入的 GT 物体，找到预测框中的最大 IoU
        ego_ious = calculate_max_iou_for_targets(det_box_tensor, ego_target_gt_boxes)
        cp_ious = calculate_max_iou_for_targets(det_box_tensor_cp, cp_target_gt_boxes)

        # V2X-Real 数据集直接采用全局 vehicle_id 进行合并

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
        if total_inserted_objects == 0:
            # 当新插入物体为 0 时（如只执行delete操作），视为合理帧
            return True, 0, 0

        valid_transform_count = 0
        # 遍历每个合并后的新插入物体
        for obj_info in merged_inserted_objects:
            global_id = obj_info['unique_id']
            
            # 获取该物体在 ego 端的 IoU
            if obj_info['ego_exists']:
                if global_id in ego_ins_ids:
                    ego_idx = ego_ins_ids.index(global_id)
                    iou_e = ego_ious[ego_idx] if ego_idx < len(ego_ious) else 0.0
                else:
                    iou_e = 0.0
            else:
                iou_e = 0.0
            
            # 获取该物体在 cp 端的 IoU
            if obj_info['cp_exists']:
                if global_id in cp_ins_ids:
                    cp_idx = cp_ins_ids.index(global_id)
                    iou_c = cp_ious[cp_idx] if cp_idx < len(cp_ious) else 0.0
                else:
                    iou_c = 0.0
            else:
                iou_c = 0.0
            
            # 判断条件：任意一端 IoU >= 0.5 即视为合理变换
            is_valid = (iou_e >= 0.5 or iou_c >= 0.5)
            if is_valid:
                valid_transform_count += 1

        # 判断是否为有效帧（所有新插入物体都是Valid）
        is_valid_frame = (valid_transform_count == total_inserted_objects)

        return is_valid_frame, total_inserted_objects, valid_transform_count
    

def generate_and_validate_frame(bg_index, scene, transformation_mode, model_path, device, max_attempts=50):
    """
    生成并验证一个有效的帧

    :param bg_index: 
    :param scene: 
    :param transformation_mode:
    :return:
    """
    print(f"Generating and validating frame for bg_index {bg_index}, mode {transformation_mode}")
    
    temp_gen_folder = "temp_gen_dataset"

    # TODO: v2x-real_dataset 应该从配置中读取
    # 创建临时配置用于生成数据
    dataset_config = DatasetConfig(dataset_name="rq_dataset/rq_ori", rq_name=temp_gen_folder)

    temp_gen_save_path = dataset_config.gen_data_save_dir
    
    for attempt in range(max_attempts):
        print(f"  Attempt {attempt + 1}/{max_attempts}")
        
        try:
            # 清理临时目录 - 现在清理整个temp_gen_dir下的所有内容
            if os.path.exists(temp_gen_save_path):
                shutil.rmtree(temp_gen_save_path)
            os.makedirs(temp_gen_save_path, exist_ok=True)
            
            # 重新定义 temp_mode_dir
            temp_mode_dir = os.path.join(temp_gen_save_path, transformation_mode)
            
            # 设置临时保存目录 - 直接保存到temp_gen_dir下
            dataset_config.v2x_dataset_saved_dir = temp_gen_save_path
            
            # 创建新的V2XInfo对象
            ego_dataset = V2XDataset(bg_index, scene, dataset_config)
            coop_dataset = V2XDataset(bg_index, scene, dataset_config, False)
            
            # 执行变换
            op_count = 0
            total_car_num = ego_dataset.get_vehicles_nums()
            selected_car_id = []
            OP_TIMES = int(transformation_mode.split('_M')[1])
            
            while op_count < OP_TIMES:
                transformation_list = ["insert", "delete", "translation", "scaling", "rotation"]
                transformation = random.choice(transformation_list)
                
                car_id = 0
                if transformation != "insert":
                    if ego_dataset.get_vehicles_nums() == 0 or \
                       ego_dataset.get_vehicles_nums() == len(selected_car_id):
                        op_count += 1
                        continue
                    car_id = random.choice(list(ego_dataset.vehicles_info.keys()))
                    if car_id in selected_car_id:
                        continue
                    selected_car_id.append(car_id)
                    if len(selected_car_id) == total_car_num:
                        op_count += 1
                        continue

                success_flag = False

                if transformation == "insert":
                    success_flag = op.vehicle_insert(ego_dataset, coop_dataset, is_vis=False)
                elif transformation == "delete":
                    success_flag = op.vehicle_delete(ego_dataset, coop_dataset, car_id, is_vis=False)
                elif transformation == "translation":
                    success_flag = op.vehicle_translation(ego_dataset, coop_dataset, car_id, is_vis=False)
                elif transformation == "scaling":
                    success_flag = op.vehicle_scaling(ego_dataset, coop_dataset, car_id, is_vis=False)
                else:
                    success_flag = op.vehicle_rotation(ego_dataset, coop_dataset, car_id, is_vis=False)

                if success_flag:
                    op_count += 1
            
            # 标签补全
            op.label_complete_for_ego(ego_dataset, coop_dataset)
            op.label_complete_for_cp(ego_dataset, coop_dataset)

            # 保存到临时目录 - 传入正确的folder_name (transformation_mode)
            ego_dataset.save_data_and_label()
            ego_dataset.save_data_and_label()
            
            # 创建临时数据集配置来加载这个单帧
            model_path = os.path.join(model_path, 'late_fusion')
            config_path = os.path.join(model_path, 'config.yaml')

            temp_hypes = yaml_utils.load_yaml(config_path, None)
            # 关键修改：validate_dir应该指向包含trans_M3的父目录
            temp_hypes['validate_dir'] = temp_gen_save_path
            
            # 构建临时数据集（只包含这一帧）
            from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
            temp_dataset = LateFusionDataset(temp_hypes, visualize=True, train=False, isSim=True)
            
            if len(temp_dataset) == 0:
                print(f"    Temp dataset empty, retrying...")
                continue
                
            # 创建DataLoader
            temp_loader = DataLoader(temp_dataset,
                                   batch_size=1,
                                   num_workers=0,  # 避免多进程问题
                                   collate_fn=temp_dataset.collate_batch_test,
                                   shuffle=False,
                                   pin_memory=False,
                                   drop_last=False)
            
            # 获取batch数据
            temp_batch = next(iter(temp_loader))
            
            # 验证帧的有效性
            is_valid, total_objs, valid_objs = evaluate_frame_validity(model_path, temp_dataset, temp_batch, device)
            
            if is_valid and total_objs >= 0:
                print(f"    Successfully generated valid frame! Total objects: {total_objs}, Valid: {valid_objs}")
                # 返回临时目录中的数据路径和信息
                return temp_mode_dir, bg_index, total_objs, valid_objs
            else:
                print(f"    Generated frame is not valid. Total objects: {total_objs}, Valid: {valid_objs}")
                
        except Exception as e:
            import traceback
            print(f"    [ERROR] Attempt {attempt + 1} failed.")
            print(f"    Exception Type: {type(e).__name__}")
            print(f"    Exception Message: {str(e)}")
            print("    Full Traceback:")
            traceback.print_exc()
            continue
    
    print(f"Failed to generate valid frame for bg_index {bg_index} after {max_attempts} attempts")
    return None, None, 0, 0


