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


def vis_test_parser():
    parser = argparse.ArgumentParser(description="v2x-real dataset test")
    parser.add_argument('-t', '--transform',
                        help="Select a transform operation, insert/delete/translation/scaling/rotation")
    parser.add_argument('-n', '--number',
                        help="Number of times the transformation is performed.")
    args = parser.parse_args()
    return args


def rq1_vis_gen(transformation="insert", gen_time=1):
    """
    V2X-Real 可视化：逐个场景、逐个数据帧 transformation 变换后可视化，并锁定选择的可视化数据
    """
    dataset_config = DatasetConfig(dataset_name="v2x-real_dataset_64", rq_name="rq1_vis")
    # dataset_config = DatasetConfig(dataset_name="v2x-real_dataset", rq_name="rq1_vis")  # 128 lines

    for scene, frame_ids in dataset_config.scenes_data_dict.items():
        # 定位场景
        # if scene in [
        #     "2023-03-17-16-12-12_3_0", 
        #     "2023-04-03-18-19-32_13_0",
        #     "2023-04-03-18-28-32_22_0",
        #     "2023-04-04-14-27-53_44_0"
        #     ]:
        #     continue
        bg_list = frame_ids

        for bg_index in bg_list:
            # 定位具体帧
            # scene = "2023-04-04-14-30-53_47_0"
            # bg_index = 12

            ego_dataset = V2XDataset(bg_index, scene, dataset_config)
            coop_dataset = V2XDataset(bg_index, scene, dataset_config, False)

            trans_count = 0
            total_car_num = ego_dataset.get_vehicles_nums()

            selected_car_id = []  # 不选择重复车辆

            while trans_count < gen_time:

                if transformation == "random":
                        transformation = random.choice(["insert", "delete", "translation", "scaling", "rotation"])

                CLogger.info(f"Background: {bg_index}, scene: {scene}, transformation: {transformation}, transform times: {trans_count}")

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

                    # Test: scnen 2, car id = 5
                    # car_id = 9
                    # if car_id > 10:
                    #     continue

                    CLogger.info(f"selected vehicle id = {car_id}")

                success_flag = False

                # v2x and baseline transformation
                if transformation == "insert":
                    success_flag = op.vehicle_insert(ego_dataset, coop_dataset, is_vis=True)
                elif transformation == "delete":
                    success_flag = op.vehicle_delete(ego_dataset, coop_dataset, car_id, is_vis=True)
                elif transformation == "translation":
                    success_flag = op.vehicle_translation(ego_dataset, coop_dataset, car_id, is_vis=True)
                elif transformation == "scaling":
                    success_flag = op.vehicle_scaling(ego_dataset, coop_dataset, car_id, is_vis=True)
                else:
                    success_flag = op.vehicle_rotation(ego_dataset, coop_dataset, car_id, is_vis=True)

                # 本轮变换失败，则重复操作
                if not success_flag:
                    continue

                trans_count += 1

            # 可视化目前不需要保存结果
            # ego_dataset.save_data_and_label()
            # coop_dataset.save_data_and_label()


if __name__ == '__main__':
    cmd_args = vis_test_parser()

    # 重新生成数据
    with open("config/dataset_config.yml", "r") as config_file:
            dataset_dict = yaml.load(config_file, Loader=yaml.FullLoader)

    rq1_vis_gen(cmd_args.transform, int(cmd_args.number))
