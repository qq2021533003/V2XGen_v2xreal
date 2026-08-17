import os
import pickle
import shutil
import numpy as np
import random
import utils.v2X_file as v2v
from utils.common_utils import get_corners_positions, center_system_transform, crop_point_cloud
from utils.v2X_file import read_Bin_PC
from config import v2x_config as config
from glob import glob


def read_ori_pcd(input_path):
    lidar = []
    with open(input_path, 'r') as f:
        line = f.readline().strip()
        while line:
            linestr = line.split(" ")
            if len(linestr) == 4:
                linestr_convert = list(map(float, linestr))
                lidar.append(linestr_convert)
            line = f.readline().strip()
    return np.array(lidar)


def convert2bin(input_pcd_dir, output_bin_dir):
    # file_list = os.listdir(input_pcd_dir)
    file_list = glob(input_pcd_dir)
    print(len(file_list))
    print(file_list)
    if not os.path.exists(output_bin_dir):
        os.makedirs(output_bin_dir)
    for i, file in enumerate(file_list):
        # (filename, extension) = os.path.splitext(file)
        # velodyne_file = os.path.join(input_pcd_dir, filename) + '.pcd'
        p_xyzi = read_ori_pcd(file)
        p_xyzi = p_xyzi.reshape((-1, 4)).astype(np.float32)
        min_val = np.amin(p_xyzi[:, 3])
        max_val = np.amax(p_xyzi[:, 3])
        p_xyzi[:, 3] = (p_xyzi[:, 3] - min_val)/(max_val-min_val)
        p_xyzi[:, 3] = np.round(p_xyzi[:, 3], decimals=2)
        p_xyzi[:, 3] = np.minimum(p_xyzi[:, 3], 0.99)
        velodyne_file_new = os.path.join(output_bin_dir, f"{i:06d}") + '.bin'
        p_xyzi.tofile(velodyne_file_new)


def get_files(directory):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in sorted(files):
            files_list.append(os.path.join(root, file))
    return files_list


def save_files(files, destination_path):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    for idx, file in enumerate(files):
        dest_file = os.path.join(destination_path, f"{idx+1:06d}{os.path.splitext(file)[1]}")
        shutil.copy(file, dest_file)


def get_matching_files(source_folder, prefix_list):
    files_list = []
    for root, dirs, files in os.walk(source_folder):
        for file in sorted(files):
            if any(file.startswith(prefix) for prefix in prefix_list):
                files_list.append(os.path.join(root, file))
    # print(files_list)
    return files_list


if __name__ == "__main__":
    # base_path = "_datasets/semantic_kitti/infer_result/1/sequences"
    # destination_dir = "/home/software/V2V4Real_semantic/v2x_dataset/v2v_test/1/"
    # destination_folders = {
    #     "labels": os.path.join(destination_dir, "labels"),
    #     "velodyne": os.path.join(destination_dir, "velodyne")
    # }
    # folders = ["labels", "velodyne"]
    # folders = ["predictions"]
    # destination_folder = os.path.join(destination_dir, "predictions")
    # sequences = ["11", "12", "13", "14", "15", "16", "17", "18", "19"]

    # for folder in folders:
    #     all_files = []
    #     for seq in sequences:
    #         path = os.path.join(base_path, seq, folder)
    #         all_files.extend(get_files(path))
    #     save_files(all_files, destination_folder)

    # ori_pcd_dir = os.path.join("/home/software/V2V4Real/", "test/*/0/*.pcd")
    # # ori_pcd_dir = os.path.join("/home/software/V2V4Real/", "test/*/1/*.pcd")
    #
    # ori_bin_dir = "/home/software/V2V4Real_semantic/v2x_dataset/test_bin/0"
    # # ori_bin_dir = "/home/software/V2V4Real_semantic/v2x_dataset/test_bin/1"
    # convert2bin(ori_pcd_dir, ori_bin_dir)

    # base_path = "/home/software/V2V4Real_semantic/v2x_dataset/v2v_test/1"
    # destination_dir = "/home/software/V2V4Real_semantic/v2x_dataset/rq1/1"
    # folders = ["labels", "velodyne", "predictions"]
    # files_select_num = config.select_data_num
    #
    # # get random list of all prefixes
    # all_files = os.listdir(os.path.join(base_path, "labels"))
    # all_prefixes = [os.path.splitext(file)[0] for file in all_files]
    #
    # # randomly select prefixes
    # random.seed(42)
    # prefix_list = random.sample(all_prefixes, files_select_num)
    #
    # # copy matching files to the new directory
    # for folder in folders:
    #     source_folder = os.path.join(base_path, folder)
    #     destination_folder = os.path.join(destination_dir, folder)
    #     matching_files = get_matching_files(source_folder, prefix_list)
    #     save_files(matching_files, destination_folder)

    base_dir = "/home/software/V2V4Real_semantic/v2x_dataset/rq1"
    ego_pc_path = os.path.join(base_dir, "result_baseline/0")
    # ego_pc_path = os.path.join(base_dir, "0/velodyne")
    ego_label_path = os.path.join(base_dir, "0/labels")
    coop_pc_path = os.path.join(base_dir, "result_baseline/1")
    # coop_pc_path = os.path.join(base_dir, "1/velodyne")
    coop_label_path = os.path.join(base_dir, "1/labels")

    select_obj_list = []

    filename = "index_list.pkl"
    with open(filename, 'rb') as f:
        select_obj_list = pickle.load(f)

    for idx in range(1, 200 + 1):
        car_index = select_obj_list[idx - 1]

        ego_xyz = v2v.load_pc(idx, ego_pc_path)
        ego_param = v2v.load_label(idx, ego_label_path)
        coop_xyz = v2v.load_pc(idx, coop_pc_path)
        coop_param = v2v.load_label(idx, coop_label_path)

        ego_corners_lidar, ego_rz_list, ego_pos_list, ego_h_list = get_corners_positions(ego_param)
        ego_location = ego_pos_list[car_index]
        coop_location = center_system_transform(ego_location, ego_param['lidar_pose'], coop_param['lidar_pose'])
        ego_xyz = crop_point_cloud(ego_xyz, ego_location, 200)
        coop_xyz = crop_point_cloud(coop_xyz, coop_location, 200)

        v2v.save_data_and_label(ego_xyz, ego_param, coop_xyz, coop_param, idx, folder_name="result_baseline/cut")







