import os
import torch
import shutil
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
import opencood.utils.common_utils as common_utils

from torch.utils.data import DataLoader
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.rq_eval.rq2_data_select import CooTest_method_result, V2X_Gen_method
from opencood.rq_eval.v2x_gen_utils import save_box_tensor
from utils.visual import visualize_individual_perspectives


#TODO
# 1. 对已生成的数据检测是否是合理插入
# 2. 删除非合理插入的数据
# 3. 循环执行筛选操作，直至生成足够数量的数据
# 4. 如果数据多于需要的，则随机从中选择定量数据，保存为新的数据集，并重新命名


def valid_frame_test(dataset_dir, model_dir, valid_dataset_dir, total_frames=1000):
    """ 
    评估新插入物体的变换质量
    
    Parameters:
    - dataset_dir: 生成数据的目录巨鲸  
    - model_dir: 训练好的 late fusion 模型路径
    - valid_dataset_dir: 有效帧数据的保存目录
    - total_frames: 评估的总帧数（可选，默认为

    待解决问题
        TODO 1. 协同视角下没有检测到预测框
            - 看看是否与命名有关，超过限度了吗（200），是否需要调整为找到最小未使用车辆 id 的命名方式
        TODO 2. 筛选后的 gt 框内无新插入对象
    """
    model_path = os.path.join(model_dir, 'late_fusion')
    config_path = os.path.join(model_path, "config.yaml")
    # model_path = "opencood/hypes_yaml/point_pillar_late_fusion.yaml"

    # 使用 late fusion 进行推理
    # print(f"Validating inserted objects using model from: {model_path}")

    hypes = yaml_utils.load_yaml(config_path, None)
    hypes['validate_dir'] = dataset_dir
    hypes['dataset_mode'] = 'v2v'   # v2x-real 采用的模式，仅限 v2v 推理

    # print('Dataset Building, dataset path = ', dataset_dir)  # 加载数据集

    opencood_dataset = build_dataset(hypes, visualize=True, train=False)

    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')  # 模型创建

    # TODO: 无融合需要训练好的模型吗？
    model = train_utils.create_model(hypes)  # 根据配置文件构建模型架构，并将其移动到 GPU
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = model_path
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # 用于全局统计新插入物体的变换质量
    global_transform_stats = {
        'total_inserted_objects': 0,  # 所有帧的新插入物体总数
        'valid_transforms': 0,  # 满足条件的变换总数
        'per_frame_stats': []  # 每帧的详细统计
    }

    # ========== 新增：有效帧保存相关变量 ==========
    if total_frames > 0:
        # 初始化各变换模式的有效帧计数器
        valid_frame_counters = {'trans_M1': 0, 'trans_M2': 0, 'trans_M3': 0}
        current_mode = None 
        target_count = total_frames
        
        # 创建目标保存目录
        valid_save_root = valid_dataset_dir
        for mode in valid_frame_counters.keys():
            mode_dir = os.path.join(valid_save_root, mode)
            os.makedirs(mode_dir, exist_ok=True)

            # NOTE: 暂时不清空
            # # 清空目录（可选，根据需求决定是否保留之前的结果）
            # for item in os.listdir(mode_dir):
            #     item_path = os.path.join(mode_dir, item)
            #     if os.path.isfile(item_path):
            #         os.remove(item_path)
            #     elif os.path.isdir(item_path):
            #         shutil.rmtree(item_path)


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
    total = 0
    total_iou = 0

    for i, batch_data in enumerate(data_loader):
        total += 1
        # if i < 109:
        #     continue
        print('data idx =', i)
        if i == 204:
            continue

        if "1" not in batch_data:
            print(f"Warning: Frame {i} - Missing CP data, skipping frame.")
            continue    

        with torch.no_grad():
            # torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)

            # 获取 ego 和 cp 视角下单独的检测和 gt 信息
            det_box_tensor, det_score, gt_box_tensor, gt_label_tensor, gt_object_ids = \
                inference_utils.inference_no_fusion(batch_data, model, opencood_dataset)
            
            det_box_tensor_cp, det_score_cp, gt_box_tensor_cp, gt_label_tensor_cp, gt_object_ids_cp = \
                inference_utils.inference_no_fusion_cp(batch_data, model, opencood_dataset)
            
            # 处理 CP 端无数据的情况
            if det_box_tensor_cp is None:
                print(f"Warning: Frame {i} - CP has no valid data, setting empty results.")
                det_box_tensor_cp = torch.zeros((0, 8, 3), device=device)
                det_score_cp = torch.zeros((0,), device=device)
                gt_box_tensor_cp = torch.zeros((0, 8, 3), device=device)
                gt_object_ids_cp = []

            ego_ins_ids = batch_data['ego']['inserted_ids']
            cp_ins_ids = batch_data['1']['inserted_ids']
            
            # 只关注类别为 vehicle (key = 1) 的对象
            CLASS_ID = 1

            det_box_tensor[det_score[:, -1] == CLASS_ID]
            gt_box_tensor[gt_label_tensor == CLASS_ID]

            det_box_tensor_cp[det_score_cp[:, -1] == CLASS_ID]
            gt_box_tensor_cp[gt_label_tensor_cp == CLASS_ID]
            
            # keep_mask = (gt_label_tensor == CLASS_ID).cpu().numpy()
            # gt_object_ids = [obj_id for obj_id, keep in zip(gt_object_ids, keep_mask) if keep]

            # 可视化
            ego_results = det_box_tensor, det_score, gt_box_tensor, gt_object_ids
            cp_results = det_box_tensor_cp, det_score_cp, gt_box_tensor_cp, gt_object_ids_cp

            ego_target_gt_boxes = filter_boxes_by_ids(gt_box_tensor, gt_object_ids, ego_ins_ids)
            cp_target_gt_boxes = filter_boxes_by_ids(gt_box_tensor_cp, gt_object_ids_cp, cp_ins_ids)

            # 计算 IoU - 针对每个新插入的 GT 物体，找到预测框中的最大 IoU
            ego_ious = calculate_max_iou_for_targets(det_box_tensor, ego_target_gt_boxes)
            cp_ious = calculate_max_iou_for_targets(det_box_tensor_cp, cp_target_gt_boxes)

            total_iou += sum(ego_ious)

            print(f"Ego inserted IDs: {ego_ins_ids}, CP inserted IDs: {cp_ins_ids}")
            print(f"Ego IoUs: {ego_ious}")
            print(f"CP IoUs: {cp_ious}")

            # ========== 利用 ass_id 建立 ego 和 cp 两端 ID 的映射关系 ==========
            # 直接从 batch_data 中获取 vehicles 信息（包含 ass_id）
            ego_vehicles = batch_data['ego'].get('vehicles', {})
            cp_vehicles = batch_data['1'].get('vehicles', {})

            print(f"\nDebug - Ego vehicles keys: {list(ego_vehicles.keys())}")
            print(f"Debug - CP vehicles keys: {list(cp_vehicles.keys())}")

            # 构建 cp_id 到 ass_id 的映射
            cp_id_to_ass_id = {}
            for cp_id, cp_vehicle in cp_vehicles.items():
                ass_id = cp_vehicle.get('ass_id', -1)
                cp_id_to_ass_id[cp_id] = ass_id

            print(f"Debug - CP ID to Ass_ID mapping: {cp_id_to_ass_id}")

            # ========== 使用 ass_id 进行 ID 合并 ==========
            # 核心逻辑：
            # 1. 如果 cp 端某车的 ass_id != -1，说明它对应 ego 端的 ass_id 那辆车
            # 2. 将这两端的车合并为同一个物理对象
            # 3. 对于 ass_id = -1 的 cp 端车辆，视为 cp 端独有的物体

            merged_inserted_objects = []  # 存储合并后的物体列表
            used_ego_ids = set()
            used_cp_ids = set()

            # 首先处理 ego 端的所有 inserted_ids
            for ego_id in ego_ins_ids:
                merged_inserted_objects.append({
                    'unique_id': ego_id,
                    'ego_id': ego_id,
                    'cp_id': None,
                    'ego_exists': True,
                    'cp_exists': False
                })
                used_ego_ids.add(ego_id)

            # 然后处理 cp 端的物体，通过 ass_id 判断是否与 ego 端的物体关联
            for cp_id in cp_ins_ids:
                if cp_id in used_cp_ids:
                    continue

                # 获取该 cp 车辆对应的 ego 车辆 ID（通过 ass_id）
                ass_id = cp_id_to_ass_id.get(cp_id, -1)

                # V2X-Real 全局 id，直接过滤即可
                # if ass_id != -1 and ass_id in used_ego_ids:
                #     # 该 cp 车辆对应 ego 端的 ass_id 车辆，进行合并
                #     for obj in merged_inserted_objects:
                #         if obj['ego_id'] == ass_id:
                #             obj['cp_exists'] = True
                #             obj['cp_id'] = cp_id
                #             used_cp_ids.add(cp_id)
                #             print(f"  -> Merged: Ego ID {ass_id} with CP ID {cp_id} (ass_id={ass_id})")
                #             break
                if cp_id in used_ego_ids:
                    # 进行合并
                    for obj in merged_inserted_objects:
                        if obj['ego_id'] == cp_id:
                            obj['cp_exists'] = True
                            obj['cp_id'] = cp_id
                            used_cp_ids.add(cp_id)
                            print(f"  -> Merged: Ego ID {ass_id} with CP ID {cp_id} (ass_id={ass_id})")
                            break
                else:
                    # cp 端独有的物体（ass_id=-1 或 ass_id 不在 ego 端）
                    merged_inserted_objects.append({
                        'unique_id': f"cp_{cp_id}",
                        'ego_id': None,
                        'cp_id': cp_id,
                        'ego_exists': False,
                        'cp_exists': True
                    })
                    used_cp_ids.add(cp_id)
                    print(f"  -> CP only: CP ID {cp_id} (ass_id={ass_id})")

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
                print(f"当前平均 IoU：{total_iou / total:.4f}")
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
        # det_save_path = "/home/zyc/code/V2XGen/rq2/rq2_det_box"
        # det_save_path_cp = "/home/zyc/code/V2XGen/rq2/rq2_det_box_cp"
        # if det_box_tensor is not None:
        #     save_box_tensor(det_box_tensor, det_score, i, det_save_path)
        # if det_box_tensor_cp is not None:
        #     save_box_tensor(det_box_tensor_cp, det_score_cp, i, det_save_path_cp)

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


def filter_boxes_by_ids(all_boxes, all_ids, target_ids):
    """
    根据 ID 列表筛选 Box。
    all_boxes: (N, ...)
    all_ids: List[int]
    target_ids: List[int]
    """
    # 如果 all_boxes 为 None，返回空数组
    if all_boxes is None:
        return np.array([])

    if len(target_ids) == 0:
        # 返回空数组，保持维度一致 (0, ...)
        if isinstance(all_boxes, np.ndarray):
            return all_boxes[:0]
        elif isinstance(all_boxes, torch.Tensor):
            return all_boxes[0:0]  # 兼容 torch
        else:
            return np.array([])

    # 构建映射
    id_to_idx = {val: idx for idx, val in enumerate(all_ids)}

    indices = []
    for tid in target_ids:
        if tid in id_to_idx:
            indices.append(id_to_idx[tid])
        else:
            # 可选：警告 ID 不存在
            pass

    if len(indices) == 0:
        if isinstance(all_boxes, np.ndarray):
            return all_boxes[:0]
        elif isinstance(all_boxes, torch.Tensor):
            return all_boxes[0:0]
        else:
            return np.array([])

    # 使用索引切片
    if isinstance(all_boxes, np.ndarray):
        return all_boxes[indices]
    elif isinstance(all_boxes, torch.Tensor):
        return all_boxes[indices]
    else:
        # 如果是 list of lists
        return [all_boxes[i] for i in indices]
    

def calculate_max_iou_for_targets(pred_box_tensor, target_gt_box_tensor):
    """
    在insert/translation/scaling/rotation操作后，可视化点云、预测框和GT框的重合程度，
    并统计基于IoU阈值的正确预测数量（TP）。

    :param v2x_info: V2XInfo 对象，包含点云 (pc) 和车辆信息 (vehicles_info)
    :param pred_box_tensor: 模型预测的边界框张量 (N, 8, 3) torch.Tensor 或 np.ndarray
    :param pred_scores: 预测框置信度 (N,)
    :param gt_box_tensor: 操作后的GT边界框张量 (M, 8, 3) torch.Tensor 或 np.ndarray
    """
    print(f"Calculating IoU for {len(pred_box_tensor)} predicted boxes and {len(target_gt_box_tensor)} target GT boxes.")

    # 1. 数据是否是torch.Tensor，是的话转换为Numpy
    if pred_box_tensor is None:
        pred_boxes = np.array([])
    elif isinstance(pred_box_tensor, torch.Tensor):
        pred_boxes = common_utils.torch_tensor_to_numpy(pred_box_tensor)
    else:
        pred_boxes = pred_box_tensor

    if target_gt_box_tensor is None:
        target_gt_box = np.array([])
    elif isinstance(target_gt_box_tensor, torch.Tensor):
        target_gt_box = common_utils.torch_tensor_to_numpy(target_gt_box_tensor)
    else:
        target_gt_box = target_gt_box_tensor

    if len(pred_boxes) == 0 or len(target_gt_box) == 0:
        return np.array([], dtype=np.float32)

    # 2. 计算IoU并筛选有效预测框（TP统计）
    det_polygons = common_utils.convert_format(pred_boxes)  # 转换为Shapely Polygon (BEV)
    target_gt_polygons = common_utils.convert_format(target_gt_box)

    max_ious = []
    for gt_poly in target_gt_polygons:
        # 计算当前 GT 框与所有预测框的 IoU
        # compute_iou(box, boxes_list) -> 返回 array
        ious = common_utils.compute_iou(gt_poly, det_polygons)

        if len(ious) > 0:
            max_iou = np.max(ious)
        else:
            max_iou = 0.0

        max_ious.append(max_iou)

    result_array = np.array(max_ious, dtype=np.float32)
    if result_array.ndim == 0:
        result_array = result_array.reshape(-1)
    return result_array