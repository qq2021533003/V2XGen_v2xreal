import random
import argparse
import shutil
import yaml
import os

import utils.operation_utils as op

from logger import CLogger
from config.dataset_config import DatasetConfig
from data_utils.v2x_dataset import V2XDataset
from utils.valid_eval import valid_insert_detection

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


# def valid_v2x_real_gen(dataset_dit, model_dir, rq_name="demo"):
#     """
#     TODO: 对执行变换操作后的数据进行有效性检验
#     - 最终有效数据要覆盖整个数据集
#     """
#     valid_insert_detection(dataset_dit, model_dir， model_dir)


def v2x_real_gen(transformation="insert", select_number=10, rq_name="demo"):
    """
    V2X-Real 数据集变换样例：从每个场景中选取 select_number 个数据帧执行 transformation 变换操作
    """
    dataset_config = DatasetConfig(dataset_name="v2x-real_dataset_64", rq_name=rq_name)

    for scene, number in dataset_config.scenes_data_dict.items():
        bg_list = list(range(0, number - 1))
        selected_ids = random.sample(bg_list, select_number)

        for bg_index in selected_ids:
            ego_dataset = V2XDataset(bg_index, scene, dataset_config)
            coop_dataset = V2XDataset(bg_index, scene, dataset_config, False)

            trans_count = 0
            total_car_num = ego_dataset.get_vehicles_nums()

            selected_car_id = []  # 不选择重复车辆

            if transformation == "random":
                transformation = random.choice(["insert", "delete", "translation", "scaling", "rotation"])

            CLogger.info(f"Background: {bg_index}, scene: {scene}, transformation: {transformation}")

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



if __name__ == '__main__':
    cmd_args = gen_test_parser()

    # 重新生成数据
    with open("config/dataset_root.yml", "r") as config_file:
            dataset_dict = yaml.load(config_file, Loader=yaml.FullLoader)

    rq_name = "demo"

    save_path = os.path.join(dataset_dict["dataset_path"], rq_name)

    # model_dir = "/mnt/e/Workspace/LabProject/V2XGen/trained_model"

    # valid_dataset_dir = os.path.join(dataset_dict["dataset_path"], f"{rq_name}_valid_nofilter")

    # 重新生成
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    # # # generate data for frd
    v2x_real_gen(cmd_args.transform, int(cmd_args.number), rq_name)

    # valid_v2x_real_gen(save_path, "/mnt/e/Workspace/LabProject/V2XGen/trained_model", rq_name)
    # valid_insert_detection(save_path, model_dir, valid_dataset_dir)

    