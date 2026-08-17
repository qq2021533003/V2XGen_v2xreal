import os
import yaml
import shutil
import argparse
import numpy as np
from glob import glob


def args_parser():
    parser = argparse.ArgumentParser(description="rq1 command")
    parser.add_argument('-d', '--dataset_dir', type=str, required=True,
                        help='Test dataset dir')
    args = parser.parse_args()
    return args


def read_ori_pcd(input_path):
    """
    1. read x, y, z, intensity of every point in pcd files
    2. fit to semanticKitti
    """
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


# def convert2bin(input_pcd_dir, output_bin_dir):
#     file_list = os.listdir(input_pcd_dir)
#     if not os.path.exists(output_bin_dir):
#         os.makedirs(output_bin_dir)
#     for file in file_list:
#         (filename, extension) = os.path.splitext(file)
#         velodyne_file = os.path.join(input_pcd_dir, filename) + '.pcd'
#         p_xyzi = read_ori_pcd(velodyne_file)
#         p_xyzi = p_xyzi.reshape((-1, 4)).astype(np.float32)
#         min_val = np.amin(p_xyzi[:, 3])
#         max_val = np.amax(p_xyzi[:, 3])
#         p_xyzi[:, 3] = (p_xyzi[:, 3] - min_val)/(max_val-min_val)
#         p_xyzi[:, 3] = np.round(p_xyzi[:, 3], decimals=2)
#         p_xyzi[:, 3] = np.minimum(p_xyzi[:, 3], 0.99)
#         velodyne_file_new = os.path.join(output_bin_dir, filename) + '.bin'
#         p_xyzi.tofile(velodyne_file_new)


def convert2bin(input_pcd_dir, output_bin_dir):
    # pcd files to bin
    # file_list = os.listdir(input_pcd_dir)
    # file_list = glob(input_pcd_dir)
    file_list = sorted(glob(input_pcd_dir))
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
        velodyne_file_new = os.path.join(output_bin_dir, f"{i + 1:06d}") + '.bin'
        p_xyzi.tofile(velodyne_file_new)


def copy_files(input_dir, output_dir, file_format="pcd"):
    file_list = sorted(glob(input_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, source_file in enumerate(file_list):
        # print(source_file)
        des_file = os.path.join(output_dir, f"{i + 1:06d}.") + file_format
        shutil.copy(source_file, des_file)


if __name__ == '__main__':
    opt = args_parser()
    test_dir = opt.dataset_dir
    dataset_root = os.path.dirname(test_dir)

    v2x_dataset_dir = os.path.join(dataset_root, "v2x_dataset")

    if not os.path.exists(v2x_dataset_dir):
        os.makedirs(v2x_dataset_dir)

    for file_format in ["pcd", "yaml"]:
        ego_files = os.path.join(test_dir, f"*/0/*.{file_format}")
        cp_files = os.path.join(test_dir, f"*/1/*.{file_format}")

        des_ego_dir = os.path.join(v2x_dataset_dir, f"0/{file_format}")
        des_cp_dir = os.path.join(v2x_dataset_dir, f"1/{file_format}")

        # copy pcds and labels
        copy_files(ego_files, des_ego_dir, file_format)
        copy_files(cp_files, des_cp_dir, file_format)

        # pcd to bin (velodyne)
        if file_format == "pcd":
            ego_bin_dir = os.path.join(v2x_dataset_dir, f"0/velodyne")
            cp_bin_dir = os.path.join(v2x_dataset_dir, f"1/velodyne")
            convert2bin(ego_files, ego_bin_dir)
            convert2bin(cp_files, cp_bin_dir)

    # get semantic predictions
    # 1. copy bin files conforms to the semanticKitti format
    ego_bin_dir = os.path.join(v2x_dataset_dir, f"0/velodyne/*.bin")
    cp_bin_dir = os.path.join(v2x_dataset_dir, f"1/velodyne/*.bin")

    semantic_dir = os.path.join(dataset_root, "semantic")

    des_bin_ego_dir = os.path.join(semantic_dir, "semanticKitti/sequences/11/velodyne")
    des_bin_cp_dir = os.path.join(semantic_dir, "semanticKitti/sequences/12/velodyne")

    copy_files(ego_bin_dir, des_bin_ego_dir, "bin")
    copy_files(cp_bin_dir, des_bin_cp_dir, "bin")

    # 2. semantic segmentation
    cmd1 = "cd ./third/SalsaNext/train/tasks/semantic"
    cmd2 = f"python infer.py -d '{semantic_dir}/semanticKitti' -m pretrained -l '{semantic_dir}/result' -s validation"

    os.system(f"{cmd1} && {cmd2}")

    # 3. copy the results of the semantic segmentation predictions
    ego_prediction_results = os.path.join(semantic_dir, f'{semantic_dir}/result/sequences/11/predictions/*.label')
    cp_prediction_results = os.path.join(semantic_dir, f'{semantic_dir}/result/sequences/12/predictions/*.label')

    des_prediction_ego_dir = os.path.join(v2x_dataset_dir, f"0/predictions")
    des_prediction_cp_dir = os.path.join(v2x_dataset_dir, f"1/predictions")

    copy_files(ego_prediction_results, des_prediction_ego_dir, "label")
    copy_files(cp_prediction_results, des_prediction_cp_dir, "label")

    # 4. save the dataset path in the config
    dataset_config = {"dataset_path": dataset_root}

    with open("./config/dataset_config.yml", "w") as config_file:
        yaml.dump(dataset_config, config_file, default_flow_style=False)
