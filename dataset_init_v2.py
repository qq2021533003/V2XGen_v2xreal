import os
import glob
import yaml
import random
import shutil
import argparse
import numpy as np

from logger import CLogger

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


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


def copy_data_files(input_dir, output_dir, file_format="bin", is_clear=False):
    file_list = sorted(glob.glob(input_dir))
    if is_clear:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    for i, source_file in enumerate(file_list):
        # print(source_file)
        des_file = os.path.join(output_dir, f"{i:06d}.") + file_format
        shutil.copy(source_file, des_file)


def copy_files(src_pattern, des_base_dir, file_format, sub_dir, sample_ratio=1.0):
    """
    复制文件并保留父文件夹结构
    src_pattern: 源文件通配符路径
    des_base_dir: 目标基础目录
    file_format: 文件格式
    sub_dir: 目标子目录（0 或 1）
    """
    # 获取所有匹配的文件
    src_files = glob.glob(src_pattern, recursive=True)

    # 只保留普通文件
    src_files = [f for f in src_files if os.path.isfile(f)]

    # 随机抽样
    if sample_ratio < 1.0:
        n_sample = max(1, int(len(src_files) * sample_ratio))
        src_files = random.sample(src_files, n_sample)

    for file_path in src_files:
        if not os.path.isfile(file_path):
            continue

        # 获取父目录路径（../1 或 ../2 的上一级）
        parent_dir = os.path.dirname(os.path.dirname(file_path))
        # 提取父文件夹名称
        parent_folder_name = os.path.basename(parent_dir)

        # 目标路径：原父文件夹名/0/format 或 原父文件夹名/1/format
        des_dir = os.path.join(des_base_dir, parent_folder_name, str(sub_dir), file_format)
        os.makedirs(des_dir, exist_ok=True)

        # 复制文件
        des_file_path = os.path.join(des_dir, os.path.basename(file_path))
        shutil.copy2(file_path, des_file_path)


if __name__ == '__main__':
    # 初始化 v2x-real 数据
    opt = args_parser()

    # 传入 test 路径
    # test_dir = "/mnt/g/v2x_dataset/V2X-Real/V2X-Real-Lidar-64"
    
    # dataset_root = os.path.dirname(test_dir)

    # 新建数据集
    # v2x_dataset_dir = os.path.join(dataset_root, "v2x-real_dataset_64")

    # train 路径
    train_dir = "/mnt/g/v2x_dataset/V2X-Real/train_64"
    dataset_root = "/mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset"
    v2x_dataset_dir = "/mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset/rq_ori_train"

    random.seed(42)

    if not os.path.exists(v2x_dataset_dir):
        os.makedirs(v2x_dataset_dir)

    # 1. 初始化数据集：转存，该存 1/2 为 0/1，仍然分场景进行划分
    # 复制数据到新目录下
    for file_format in ["bin", "yaml"]:
        ego_files = os.path.join(train_dir, f"*/1/*.{file_format}")
        cp_files = os.path.join(train_dir, f"*/2/*.{file_format}")
 
        # 0.1 是初始化训练集，否则忽略 sample_ratio
        copy_files(ego_files, v2x_dataset_dir, file_format, sub_dir=0, sample_ratio=1)
        copy_files(cp_files, v2x_dataset_dir, file_format, sub_dir=1, sample_ratio=1)


    # 2. 对逐个场景下的数据进行语义分割
    for file_name in os.listdir(v2x_dataset_dir):
        # get semantic predictions
        # 1. copy bin files conforms to the semanticKitti format 将数据复制到路面分割指定目录
        ego_bin_dir = os.path.join(v2x_dataset_dir, file_name, f"0/bin/*.bin")
        cp_bin_dir = os.path.join(v2x_dataset_dir, file_name, f"1/bin/*.bin")

        semantic_dir = os.path.join(dataset_root, "semantic")

        des_bin_ego_dir = os.path.join(semantic_dir, "semanticKitti/sequences/11/velodyne")
        des_bin_cp_dir = os.path.join(semantic_dir, "semanticKitti/sequences/12/velodyne")

        copy_data_files(ego_bin_dir, des_bin_ego_dir, "bin", True)
        copy_data_files(cp_bin_dir, des_bin_cp_dir, "bin", True)

        # 2. semantic segmentation 语义分割
        cmd1 = "cd ./third/SalsaNext/train/tasks/semantic"
        cmd2 = f"python infer.py -d '{semantic_dir}/semanticKitti' -m pretrained -l '{semantic_dir}/result' -s validation"

        os.system(f"{cmd1} && {cmd2}")

        # 3. copy the results of the semantic segmentation predictions 复制推理结果到数据集
        ego_prediction_results = os.path.join(semantic_dir, f'{semantic_dir}/result/sequences/11/predictions/*.label')
        cp_prediction_results = os.path.join(semantic_dir, f'{semantic_dir}/result/sequences/12/predictions/*.label')

        des_prediction_ego_dir = os.path.join(v2x_dataset_dir, file_name, f"0/predictions")
        des_prediction_cp_dir = os.path.join(v2x_dataset_dir, file_name, f"1/predictions")

        copy_data_files(ego_prediction_results, des_prediction_ego_dir, "label")
        copy_data_files(cp_prediction_results, des_prediction_cp_dir, "label")

        CLogger.info(f"Scene {file_name} has completed semantic segmentation!")

    # 4. save the dataset path in the config
    dataset_config = {"dataset_path": dataset_root}

    with open("./config/dataset_config.yml", "w") as config_file:
        yaml.dump(dataset_config, config_file, default_flow_style=False)

