import argparse
import os
import torch
import shutil
import random
import utils.operation_utils as op

from torch.utils.data import DataLoader 
from opencood.hypes_yaml import yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from logger import CLogger
from config.dataset_config import DatasetConfig
from data_utils.v2x_dataset import V2XDataset
from rq_tools.valid_frame_generation import evaluate_frame_validity, generate_and_validate_frame


def rq2_parser():
    parser = argparse.ArgumentParser(description="Generate valid frames for RQ2")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Test dataset dir.')
    parser.add_argument('--max_attempts', type=int, default=10,
                        help='Maximum attempts to generate a valid frame')
    opt = parser.parse_args()
    return opt


def rq2_gen(dataset_dir, is_gen=True):
    """
    通过 M1/M2/M3 变换生成数据，需要 v2xgen 标签
    """
    # dataset_root = os.path.dirname(dataset_dir)
    # dataset_save_root = os.path.join(dataset_root, "rq2_gen")
    # TODO: 排除协同信息缺失的数据！
    # TODO: 解决 lidar simulation failed 问题

    # 生成数据并补充实验用标签信息
    if is_gen:
        for trans_time in range(1, 4):
            # dataset_config = DatasetConfig(dataset_name="rq_dataset/rq_ori", rq_name=f"rq_dataset/rq2/rq2_gen/trans_M{trans_time}")

            dataset_config = DatasetConfig(dataset_name="rq_dataset/rq_ori_train", rq_name=f"rq_dataset/rq2/rq2_gen/train_20")

            for scene, bg_list in dataset_config.scenes_data_dict.items():
                for bg_index in bg_list:
                    # if scene in [
                    #     "2023-03-17-16-12-12_3_0",
                    #     "2023-04-03-18-19-32_13_0",
                    #     "2023-04-03-18-28-32_22_0",
                    #     "2023-04-04-14-27-53_44_0",
                    #     "2023-04-04-14-30-53_47_0",
                    #     "2023-04-04-14-34-53_51_1",
                    #     "2023-04-04-15-47-17_19_0",
                    #     "2023-04-05-16-25-26_22_0",
                    #     "2023-04-05-16-31-26_28_1",
                    #     "2023-04-07-15-02-15_1_0",
                    #     "2023-04-07-15-02-15_1_1",
                    #     "2023-04-07-15-04-15_3_1",
                    #     # "2023-04-07-15-05-15_4_0",
                    #     "2023-04-07-15-05-15_4_1"
                    # ]:
                    #     continue
                    # if scene == "2023-04-07-15-05-15_4_0" and (bg_index <= 93 or bg_index >= 214):
                    #     continue
                        
                    ego_dataset = V2XDataset(bg_index, scene, dataset_config)
                    coop_dataset = V2XDataset(bg_index, scene, dataset_config, False)

                    trans_count = 0
                    total_car_num = ego_dataset.get_vehicles_nums()

                    selected_car_id = []  # 不选择重复车辆

                    while trans_count < trans_time:
                        transformation = random.choice(["insert", "delete", "translation", "scaling", "rotation"])

                        CLogger.info(f"Background: {bg_index}, scene:` {scene}, transformation: {transformation}, transform times: {trans_time}")

                        if transformation != "insert":
                            # 从 ego 场景随机选择一辆车
                            if ego_dataset.get_vehicles_nums() == 0 or \
                                    ego_dataset.get_vehicles_nums() == len(selected_car_id):
                                trans_count += 1
                                continue
                            car_id = random.choice(list(ego_dataset.vehicles_info.keys()))
                            # 每次操作(无论成功否)选择不同的车
                            if car_id in selected_car_id:
                                continue
                            selected_car_id.append(car_id)
                            # print(car_id, selected_car_id)
                            if len(selected_car_id) == total_car_num:
                                # 场景中只有协同车
                                trans_count += 1
                                continue

                        success_flag = False

                        # v2x and baseline transformation
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

                        # 本轮变换失败，则重复操作
                        if not success_flag:
                            continue

                        trans_count += 1
                    
                    # 记录遮挡率和距离
                    op.label_complete_for_ego(ego_dataset, coop_dataset)
                    op.label_complete_for_cp(ego_dataset, coop_dataset)

                    ego_dataset.save_data_and_label()
                    coop_dataset.save_data_and_label()
    # 读取数据并补充标签信息
    else:
        dataset_config = DatasetConfig(dataset_name="rq_test", rq_name="rq3/rq3_ori")

        for scene, bg_list in dataset_config.scenes_data_dict.items():

            for bg_index in bg_list:

                CLogger.info(f"Background: {bg_index}, scene:` {scene}")
                ego_dataset = V2XDataset(bg_index, scene, dataset_config)
                coop_dataset = V2XDataset(bg_index, scene, dataset_config, False)

                op.label_complete_for_ego(ego_dataset, coop_dataset)
                op.label_complete_for_cp(ego_dataset, coop_dataset)

                ego_dataset.save_data_and_label()
                coop_dataset.save_data_and_label()


def main(model_dir, gen_dir):
    """
    有效性插入接口:
    1. 检测已生成数据的有效性
    2. 将不满足有效性的数据重新生成
    最终保存目录下的所有生成数据均可通过有效性检测

    --------------------------------------------
    :param model_dir: 加载预训练模型的路径
    :param gen_dir: 先生成数据的路径
    :return:
    
    """
    # 先生成一批 M1/M2/M3 数据
    # rq2_gen(dataset_dir=opt.dataset_dir, is_gen=True)

    # 使用无融合模型（后融合模型）
    dataset_root = os.path.dirname(gen_dir)
    model_path = os.path.join(model_dir, "late_fusion")
    hypes_path = os.path.join(model_path, "config.yaml")

    processed_frames = {'trans_M1': 0, 'trans_M2': 0, 'trans_M3': 0}

    # 最终保存有效性数据的路径
    rq2_save_root = os.path.join(dataset_root, "rq2_gen_valid")     

    # 清空有效数据保存目录
    for mode in ['trans_M1', 'trans_M2', 'trans_M3']:
        mode_dir = os.path.join(rq2_save_root, mode)
        os.makedirs(mode_dir, exist_ok=True)
        for item in os.listdir(mode_dir):
            item_path = os.path.join(mode_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    # 初始化预训练模型
    hypes = yaml_utils.load_yaml(hypes_path, None)
    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    saved_path = model_path
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # ---------------- 检测已生成数据的有效性 ----------------
    for mod in ['trans_M1', 'trans_M2', 'trans_M3']:
        gen_path = os.path.join(gen_dir, mod)

        if not os.path.exists(gen_path):
            continue

        hypes['validate_dir'] = gen_path

        opencood_dataset = build_dataset(hypes, visualize=True, train=False)
        
        data_loader = DataLoader(opencood_dataset,
                                batch_size=1,
                                num_workers=4,
                                collate_fn=opencood_dataset.collate_batch_test,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False)

        print(f"Processing {len(opencood_dataset)} frames...")
        
        for i, batch_data in enumerate(data_loader):
            print(f'Processing frame {i+1}/{len(opencood_dataset)}')
            
            with torch.no_grad():           
                # 获取变换模式信息
                ego_v2x_gen_dict = batch_data['ego']['v2x_gen']
                if not ego_v2x_gen_dict:
                    print(f"Frame {i}: No v2x_gen info, skipping")
                    continue
    
                first_key = list(ego_v2x_gen_dict.keys())[0]
                scene = ego_v2x_gen_dict[first_key]['folder_name']
                timestamp = ego_v2x_gen_dict[first_key]['timestamp']

                # 评估帧的有效性
                is_valid, total_objs, valid_objs = evaluate_frame_validity(model, opencood_dataset, batch_data, device)
                
                print(f"DEBUG: Frame {i} evaluation result - is_valid={is_valid}, total_objs={total_objs}, valid_objs={valid_objs}")

                if is_valid and total_objs >= 0:  # 修改条件：total_objs 可以为 0
                    print(f"Frame {i} ({mod}) is already valid! Total objects: {total_objs}, Valid: {valid_objs}")
                    
                    # 直接复制原始数据到目标目录
                    try:
                        # 获取原始数据路径
                        original_data_dir = gen_path
                        bg_index_str = f"{int(timestamp):06d}" if isinstance(timestamp, (int, str)) else f"{i:06d}"
                        
                        # 源路径
                        src_ego_pcd = os.path.join(original_data_dir, scene, "0", f"{bg_index_str}.bin")
                        src_ego_yaml = os.path.join(original_data_dir, scene, "0", f"{bg_index_str}.yaml")
                        src_cp_pcd = os.path.join(original_data_dir, scene, "1", f"{bg_index_str}.bin")
                        src_cp_yaml = os.path.join(original_data_dir, scene, "1", f"{bg_index_str}.yaml")
                        
                        # 目标路径
                        valid_save_root = os.path.join(rq2_save_root, mod)
                        target_mode_dir = os.path.join(valid_save_root, scene)
                        target_ego_dir = os.path.join(target_mode_dir, "0")
                        target_cp_dir = os.path.join(target_mode_dir, "1")
                        os.makedirs(target_ego_dir, exist_ok=True)
                        os.makedirs(target_cp_dir, exist_ok=True)
                        
                        # 复制文件
                        if os.path.exists(src_ego_pcd):
                            shutil.copy2(src_ego_pcd, os.path.join(target_ego_dir, f"{bg_index_str}.bin"))
                        if os.path.exists(src_ego_yaml):
                            shutil.copy2(src_ego_yaml, os.path.join(target_ego_dir, f"{bg_index_str}.yaml"))
                        if os.path.exists(src_cp_pcd):
                            shutil.copy2(src_cp_pcd, os.path.join(target_cp_dir, f"{bg_index_str}.bin"))
                        if os.path.exists(src_cp_yaml):
                            shutil.copy2(src_cp_yaml, os.path.join(target_cp_dir, f"{bg_index_str}.yaml"))
                        
                        processed_frames[mod] += 1
                        print(f"Copied valid frame {bg_index_str} to {mod} (count: {processed_frames[mod]})")
                        
                    except Exception as e:
                        print(f"Error copying frame {i}: {e}")
                        # 如果复制失败，尝试重新生成并验证
                        print(f"Regenerating and validating frame {i} due to copy error...")
                        temp_dir, bg_idx, total_o, valid_o = generate_and_validate_frame(
                            int(timestamp), mod, model, opencood_dataset, device, opt.max_attempts)
                        if temp_dir is not None:
                            # 复制验证有效的数据
                            bg_index_str = f"{int(timestamp):06d}"
                            target_mode_dir = os.path.join(valid_save_root, mod)
                            target_ego_dir = os.path.join(target_mode_dir, "0")
                            target_cp_dir = os.path.join(target_mode_dir, "1")
                            os.makedirs(target_ego_dir, exist_ok=True)
                            os.makedirs(target_cp_dir, exist_ok=True)
                            
                            # 从临时目录复制
                            temp_ego_pcd = os.path.join(temp_dir, "0", f"{bg_index_str}.bin")
                            temp_ego_yaml = os.path.join(temp_dir, "0", f"{bg_index_str}.yaml")
                            temp_cp_pcd = os.path.join(temp_dir, "1", f"{bg_index_str}.bin")
                            temp_cp_yaml = os.path.join(temp_dir, "1", f"{bg_index_str}.yaml")
                            
                            if os.path.exists(temp_ego_pcd):
                                shutil.copy2(temp_ego_pcd, os.path.join(target_ego_dir, f"{bg_index_str}.bin"))
                            if os.path.exists(temp_ego_yaml):
                                shutil.copy2(temp_ego_yaml, os.path.join(target_ego_dir, f"{bg_index_str}.yaml"))
                            if os.path.exists(temp_cp_pcd):
                                shutil.copy2(temp_cp_pcd, os.path.join(target_cp_dir, f"{bg_index_str}.bin"))
                            if os.path.exists(temp_cp_yaml):
                                shutil.copy2(temp_cp_yaml, os.path.join(target_cp_dir, f"{bg_index_str}.yaml"))
                            
                            processed_frames[mod] += 1
                            print(f"Saved regenerated and validated frame {bg_index_str} to {mod}")
                
                else:
                    print(f"Frame {i} ({mod}) is not valid. Total objects: {total_objs}, Valid: {valid_objs}")
                    print(f"Regenerating and validating frame {i} until valid...")
                    
                    # 重新生成并验证有效帧
                    try:
                        temp_dir, bg_idx, total_o, valid_o = generate_and_validate_frame(
                            int(timestamp), mod, model, opencood_dataset, device, opt.max_attempts)
                        
                        if temp_dir is not None:
                            # 保存验证有效的帧到目标目录
                            bg_index_str = f"{int(timestamp):06d}"
                            target_mode_dir = os.path.join(valid_save_root, mod)
                            target_ego_dir = os.path.join(target_mode_dir, "0")
                            target_cp_dir = os.path.join(target_mode_dir, "1")
                            os.makedirs(target_ego_dir, exist_ok=True)
                            os.makedirs(target_cp_dir, exist_ok=True)
                            
                            # 从临时目录复制
                            temp_ego_pcd = os.path.join(temp_dir, "0", f"{bg_index_str}.pcd")
                            temp_ego_yaml = os.path.join(temp_dir, "0", f"{bg_index_str}.yaml")
                            temp_cp_pcd = os.path.join(temp_dir, "1", f"{bg_index_str}.pcd")
                            temp_cp_yaml = os.path.join(temp_dir, "1", f"{bg_index_str}.yaml")
                            
                            if os.path.exists(temp_ego_pcd):
                                shutil.copy2(temp_ego_pcd, os.path.join(target_ego_dir, f"{bg_index_str}.pcd"))
                            if os.path.exists(temp_ego_yaml):
                                shutil.copy2(temp_ego_yaml, os.path.join(target_ego_dir, f"{bg_index_str}.yaml"))
                            if os.path.exists(temp_cp_pcd):
                                shutil.copy2(temp_cp_pcd, os.path.join(target_cp_dir, f"{bg_index_str}.pcd"))
                            if os.path.exists(temp_cp_yaml):
                                shutil.copy2(temp_cp_yaml, os.path.join(target_cp_dir, f"{bg_index_str}.yaml"))
                            
                            processed_frames[mod] += 1
                            print(f"Saved regenerated and validated frame {bg_index_str} to {mod}")
                        else:
                            print(f"Failed to regenerate valid frame for timestamp {timestamp}")
                            
                    except Exception as e:
                        print(f"Error regenerating frame {i}: {e}")
        
        # # 清理临时目录
        temp_gen_dir = "/home/zyc/code/V2XGen/rq_eval/rq2_gen_temp"
        if os.path.exists(temp_gen_dir):
            shutil.rmtree(temp_gen_dir)
        
        print("\n=== Process Complete ===")
        print("Original data remains in:", gen_dir)
        # print("Valid frames saved to:", valid_save_root)
        print("Processed frames per mode:")
        for mode, count in processed_frames.items():
            print(f"  {mode}: {count} frames")


if __name__ == '__main__':
    opt = rq2_parser()

    main(model_dir=opt.model_dir, gen_dir=opt.dataset_dir)
    # rq2_gen(dataset_dir=opt.dataset_dir, is_gen=False)