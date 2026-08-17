import math
import random
import numpy as np


def get_random_rotation(min_range=5, max_range=30):
    """
    random rotation degree, [-30,-5] or [5,30]
    """
    range1 = (-max_range, -min_range)
    range2 = (min_range, max_range)

    selected_range = range1 if np.random.rand() < 0.5 else range2

    random_rot = np.random.uniform(selected_range[0], selected_range[1])

    return random_rot


def get_insert_location(v2x_info):
    """
    Randomly generate insert coordinates based on the road split extent.
    """
    # GT_RANGE = [-100, -40, -15, 100, 40, 15]

    road_pc = v2x_info.road_pc

    if road_pc is None or len(road_pc) == 0:
        print("Warning: road_pc is empty, cannot insert.")
        return None, None
    
    distances = np.hypot(road_pc[:, 0], road_pc[:, 1])
    valid_mask = distances <= 100

    valid_mask &= (road_pc[:, 1] >= -40) & (road_pc[:, 1] <= 40)

    valid_road_pc = road_pc[valid_mask]

    if len(valid_road_pc) == 0:
         print("Warning: No road points valid for insertion.")
         return None, None

    # 基于路面点生成插入位置
    chosen_point = random.choice(valid_road_pc)
    pos_x = chosen_point[0] + random.uniform(-0.5, 0.5)
    pos_y = chosen_point[1] + random.uniform(-0.5, 0.5)

    degree = get_random_rotation()

    return [pos_x, pos_y], degree
    

    # road_x = road_pc[:, 0]
    # road_y = road_pc[:, 1]
    # x_range = [np.min(road_x), np.max(road_x)]
    # y_range = [np.min(road_y), np.max(road_y)]

    # # 过滤掉检测范围外的点
    # x_range[0] = max(x_range[0], GT_RANGE[0])
    # x_range[1] = min(x_range[1], GT_RANGE[3])
    # y_range[0] = max(y_range[0], GT_RANGE[1])
    # y_range[1] = min(y_range[1], GT_RANGE[4])

    # pos_x = 0
    # pos_y = 0

    # is_valid = False
    # while not is_valid:
    #     pos_x = random.uniform(x_range[0], x_range[1])
    #     pos_y = random.uniform(y_range[0], y_range[1])
    #     # print(pos_x, pos_y)

    #     # 在 100 的检测范围内
    #     if math.sqrt(pos_x**2 + pos_y**2) <= 100:
    #         is_valid = True
    # degree = get_random_rotation()

    # return [pos_x, pos_y], degree
