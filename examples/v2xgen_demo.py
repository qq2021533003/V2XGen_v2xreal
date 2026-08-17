import json
import random
import argparse
import obj_transformation_demo as trans
from logger import CLogger
from config.config import Config
from utils.v2x_object import V2XInfo


def demo_parser():
    parser = argparse.ArgumentParser(description="rq1 command")
    parser.add_argument('-t', '--transform',
                        help="select a transform operation, insert/delete/translation/scaling/rotation")
    parser.add_argument('-s', '--scene', help="the scene of dataset", default=1)
    args = parser.parse_args()
    return args


def v2xgen_demo(transformation='insert', scene=1):
    dataset_config = Config(dataset="v2x_dataset", scene=scene)
    begin_index = dataset_config.begin_index
    data_num = dataset_config.scene_data_num

    select_data_num = dataset_config.select_data_num

    for bg_index in range(1, select_data_num + 1):
        if bg_index < begin_index:
            continue
        elif bg_index > data_num + begin_index:
            break

        CLogger.info(f"Background {bg_index}")

        # load vehicle info
        ego_info = V2XInfo(bg_index, dataset_config=dataset_config)
        cp_info = V2XInfo(bg_index, is_ego=False, dataset_config=dataset_config)

        # random select car
        car_id = random.choice(list(ego_info.vehicles_info.keys()))

        if transformation == 'insert':
            trans.vehicle_insert(ego_info, cp_info)
        elif transformation == 'delete':
            trans.vehicle_delete(ego_info, cp_info, car_id)
        elif transformation == 'translation':
            trans.vehicle_translation(ego_info, cp_info, car_id)
        elif transformation == 'scaling':
            trans.vehicle_scaling(ego_info, cp_info, car_id)
        elif transformation == 'rotation':
            trans.vehicle_rotation(ego_info, cp_info, car_id)


if __name__ == '__main__':
    cmd_args = demo_parser()
    v2xgen_demo(cmd_args.transform, int(cmd_args.scene))
