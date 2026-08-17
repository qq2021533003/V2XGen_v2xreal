import random
import argparse
import shutil
import yaml
import os

import utils.operation_utils as op

from logger import CLogger
from config.dataset_config import DatasetConfig
from data_utils.v2x_dataset import V2XDataset
# from utils.valid_eval import valid_insert_detection


def gen_test_parser():
    parser = argparse.ArgumentParser(description="v2x-real dataset test")
    parser.add_argument('-t', '--transform',
                        help="Select a transform operation, insert/delete/translation/scaling/rotation")
    parser.add_argument('-n', '--number',
                        help="The amount of data selected from each scenario")
    parser.add_argument('-r', '--rq_name',
                        help="The name of the request")
    args = parser.parse_args()
    return args


def rq1_frd_gen(select_number, gen_time, dataset_name="v2x-real_dataset_64"):
    # ========== 生成 RQ1 FRD 数据 ==========
    # 功能：基于 V2XGen 和 Baseline 两种方法生成数据
    # 逻辑：
    #   1. 分三次循环生成数据，选择随机变换方式，设置单个数据操作次数为 M1/M2/M3
    #   2. 每个场景下随机选取 50 个数据作为原始数据集
    #   3. 循环进行有效性检测，使得每个原始数据都成功变换
    #
    # =============================================
    dataset_config = DatasetConfig(dataset_name=dataset_name, rq_name=f"rq1_frd/M{gen_time}")

    for scene, frame_ids in dataset_config.scenes_data_dict.items():
        
        selected_ids = random.sample(frame_ids, select_number)

        # 原始数据保存路径
        ori_data_save_dir = os.path.join(dataset_config.gen_data_save_dir, "ori", scene)
        ego_ori_save_path = os.path.join(ori_data_save_dir, "0")
        cp_ori_save_path = os.path.join(ori_data_save_dir, "1")

        for bg_index in selected_ids:
            ego_dataset = V2XDataset(bg_index, scene, dataset_config)
            coop_dataset = V2XDataset(bg_index, scene, dataset_config, False)

            trans_count = 0
            total_car_num = ego_dataset.get_vehicles_nums()

            selected_car_id = []  # 不选择重复车辆

            while trans_count < gen_time:

                transformation = random.choice(["insert", "delete", "translation", "scaling", "rotation"])

                CLogger.info(f"RQ1 FRD Gen start, background: {bg_index}, scene: {scene}, transformation: {transformation}")

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

                # loading_info_verification_test(ego_dataset, coop_dataset, car_id)
                #
                # sys.exit()

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

                ego_dataset.save_data_and_label()
                coop_dataset.save_data_and_label()

if __name__ == "__main__":
    cmd_args = gen_test_parser()

    rq1_frd_gen(
        select_number=int(cmd_args.number),
        gen_time=1,
        dataset_name="v2x-real_dataset_64",
    )