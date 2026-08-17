import yaml
import os
import numpy as np

from collections import OrderedDict
from utils.v2x_file import load_yaml
from sklearn.cluster import KMeans
from utils.v2x_file import read_Bin_PC
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation

def sort_lidar_angles(input_yml, output_yml, keep_every=1):
    """
    读取 lidar 配置，按角度值升序重新排列线束，并可选地降采样到更少的线数

    参数
    ----
    input_yml : str
        输入 YAML 文件路径。
    output_yml : str
        输出 YAML 文件路径。
    keep_every : int
        保留步长，1 表示保留所有线束（128 线），
        2 表示隔一删一（64 线），以此类推。
    """
    # 读取原始配置
    config = load_yaml(input_yml)

    # 提取角度字典 {线号: 角度}
    angle_dict = config.get('evaluate_angle', {})
    if not angle_dict:
        raise ValueError("配置文件中缺少 'evaluate_angle' 字段")

    # 按角度值排序，保留线号与角度的元组列表
    sorted_pairs = sorted(angle_dict.items(), key=lambda x: x[1])

    # 根据步长降采样
    selected_pairs = sorted_pairs[::keep_every]

    # 重新分配线号 1..N，构建新的有序字典
    new_angle_dict = OrderedDict()
    for new_id, (_, angle) in enumerate(selected_pairs, start=1):
        new_angle_dict[new_id] = angle

    # 更新配置中的 evaluate_angle
    config['evaluate_angle'] = new_angle_dict

    # 写入新的 YAML 文件
    with open(output_yml, 'w', encoding='utf-8') as f:
        yaml.dump(config, f,
                  Dumper=yaml.Dumper,          
                  default_flow_style=False,     
                  sort_keys=False,              
                  allow_unicode=True,           
                  indent=2)                     

    print(f"处理完成，输出文件：{output_yml}，共 {len(new_angle_dict)} 条线束")


def extract_vertical_angles(points, num_beams=64):
    """
    根据 K-means 算法提取雷达线分布

    ----
    points: Nx3 或 Nx4 numpy array, 前三列为 x,y,z
    返回排序后的垂直角度列表 (度)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    vert_angles = np.arctan(z / np.sqrt(x**2 + y**2)) * 180 / np.pi

    # 聚类
    kmeans = KMeans(n_clusters=num_beams, random_state=0).fit(vert_angles.reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()
    centers_sorted = np.sort(centers)
    return centers_sorted


def extract_vertical_angles_robust(frame_files, num_beams=64, min_dist=3.0, max_dist=60.0):
    """
    基于直方图寻峰的多帧激光雷达垂直视场角提取。
    """
    all_peaks = []

    for file_path in frame_files:
        points = read_Bin_PC(file_path) # 替换为你的读取函数
        if points.shape[0] < 1000:
            continue

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # 1. 距离过滤：去除车身近处的噪点（引擎盖反射）和远处的稀疏噪点
        dist = np.sqrt(x**2 + y**2)
        valid_mask = (dist > min_dist) & (dist < max_dist)
        x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
        dist = dist[valid_mask]

        # 2. 计算垂直角
        vert_angles = np.arctan2(z, dist) * 180 / np.pi

        # 3. 构建高分辨率直方图 (精度达到 0.01 度)
        # 大部分雷达的分布在 -30 到 +25 度之间
        hist, bin_edges = np.histogram(vert_angles, bins=6000, range=(-25.0, 15.0))
        
        # 4. 寻峰算法 (Peak Finding)
        # distance: 限制两个峰之间的最小角度间隔 (例如 64线雷达线距通常 > 0.1度)
        # prominence: 突出度，用于过滤微小的噪声峰
        peaks_indices, _ = find_peaks(hist, distance=5, prominence=50) 
        
        # 将索引转换回实际角度
        frame_angles = bin_edges[peaks_indices] + (bin_edges[1] - bin_edges[0]) / 2

        # 校验：如果找到的峰值数量接近 64，则认为该帧质量良好
        if abs(len(frame_angles) - num_beams) <= 5: 
            # 如果略多或略少，可以通过幅度排序或差值匹配做二次筛选
            # 简单起见，这里假设参数调优后能稳定找到 64 个峰
            if len(frame_angles) == num_beams:
                all_peaks.append(np.sort(frame_angles))
            else:
                print(len(frame_angles))
        else:
            print(len(frame_angles))

    if not all_peaks:
        print(len(all_peaks))
        raise ValueError("未能从提供的帧中提取出完整的 64 线特征，请检查点云质量或调整寻峰参数。")

    # 5. 多帧融合与 MAD 离群值剔除 (保留你原有的优秀逻辑)
    all_peaks = np.array(all_peaks)
    median_peaks = np.median(all_peaks, axis=0)
    
    deviations = np.abs(all_peaks - median_peaks)
    mad = np.median(deviations, axis=0)
    
    # 避免 MAD 为 0 导致除以零错误
    mad = np.where(mad == 0, 1e-6, mad) 
    threshold = 3 * 1.4826 * mad
    
    frame_scores = np.max(deviations / threshold, axis=1)
    good_frames = frame_scores < 1.5
    
    final_angles = np.median(all_peaks[good_frames], axis=0)
    
    # 返回从上到下 (大到小) 排序的角度，符合仿真器惯例
    return final_angles[::-1]


def extract_vertical_angles_from_frames(frame_files, num_beams=64, 
                                        mount_angle_range=(-10, 10), 
                                        ground_percentile=10):
    """
    循环处理多帧点云，鲁棒提取64线垂直角度配置。

    参数:
        frame_files : list of str  点云文件路径列表
        num_beams : int            激光线数
        mount_angle_range : tuple  安装倾角搜索范围 (度)
        ground_percentile : int    地面点判定用的z值分位数
    返回:
        final_angles_deg : np.ndarray 排序后的64个垂直角度 (度)
    """
    all_centers = []   # 存储每帧提取的中心
    mount_angles = []  # 存储每帧估计的安装倾角

    for file_path in frame_files:
        # 读取点云 (这里以npy为例)
        points = read_Bin_PC(file_path)
        if points.shape[1] < 3:
            continue

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        vert_angles = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi

        # ----- 1. 估计安装倾角 (基于地面点) -----
        # 取z最低的百分比点，认为它们是地面
        threshold = np.percentile(z, ground_percentile)
        ground_mask = z < threshold
        if np.sum(ground_mask) < 100:
            continue  # 地面点太少，跳过
        ground_z = z[ground_mask]
        ground_dist = np.sqrt(x[ground_mask]**2 + y[ground_mask]**2)
        ground_angles = np.arctan2(ground_z, ground_dist) * 180 / np.pi
        # 安装倾角近似为地面点垂直角的中位数（假设平坦地面）
        mount_angle = np.median(ground_angles)
        mount_angles.append(mount_angle)

        # ----- 2. 校正垂直角到雷达本体坐标系 -----
        corrected_angles = vert_angles - mount_angle

        # ----- 3. K-Means 聚类得到64个中心 -----
        kmeans = KMeans(n_clusters=num_beams, random_state=0, n_init=10)
        kmeans.fit(corrected_angles.reshape(-1, 1))
        centers = kmeans.cluster_centers_.flatten()
        centers_sorted = np.sort(centers)
        all_centers.append(centers_sorted)

    # ----- 4. 多帧融合：使用中位数聚合 -----
    all_centers = np.array(all_centers)  # shape (n_frames, 64)
    median_centers = np.median(all_centers, axis=0)

    # 可选：剔除离群帧后再次计算
    # 计算每帧与中位数的MAD，去除偏差过大的帧
    deviations = np.abs(all_centers - median_centers)
    mad = np.median(deviations, axis=0)
    # 使用每个通道的MAD阈值过滤（按1.4826缩放至正态分布）
    threshold = 3 * 1.4826 * mad
    frame_scores = np.max(deviations / threshold, axis=1)
    good_frames = frame_scores < 1.5  # 保留90%相似度以上的帧
    if np.sum(good_frames) > 5:       # 至少5帧
        median_centers = np.median(all_centers[good_frames], axis=0)

    # 通常我们返回基于车身水平面的角度（即已经减去安装倾角的值）
    return median_centers


def generate_lidar_yaml_with_angles(angles_deg, template_yaml_path, output_yaml_path):
    """
    用提取的角度生成与先前相同格式的YAML配置文件。
    """
    with open(template_yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # 构建 evaluate_angle
    angle_dict = OrderedDict()
    for i, ang in enumerate(sorted(angles_deg), start=1):
        angle_dict[i] = float(ang)

    config['evaluate_angle'] = angle_dict
    # 可能需要更新 FOV 范围参数
    config['vertical_up'] = float(np.min(angles_deg))
    config['vertical_down'] = float(np.max(angles_deg))

    with open(output_yaml_path, 'w') as f:
        yaml.dump(config, f, Dumper=yaml.Dumper, default_flow_style=False,
                  sort_keys=False, allow_unicode=True, indent=2)


if __name__ == '__main__':
    # # 生成排序后的 128 线配置
    # sort_lidar_angles('config/lidar_config_128.yml', 'config/rs_ruby_plus_sorted.yml', keep_every=1)

    # # 生成隔一删一的 64 线配置
    # sort_lidar_angles('config/lidar_config_128.yml', 'config/rs_ruby_plus_64lines.yml', keep_every=2)

    with open("config/dataset_config.yml", "r") as config_file:
            dataset_dict = yaml.load(config_file, Loader=yaml.FullLoader)

    dataset_root = dataset_dict["dataset_path"]

    scene = "2023-03-17-16-12-12_3_0"

    bg_index = 1

    dataset_path = os.path.join(dataset_root, "v2x-real_dataset_64")

    data_paths = f"{dataset_path}/*/*/bin/*.bin"

    import glob
    frame_files = glob.glob(data_paths)
    print(f"找到 {len(frame_files)} 帧数据")

    # 循环计算得到最终角度
    # final_angles = extract_vertical_angles_from_frames(frame_files, num_beams=64)
    final_angles = extract_vertical_angles_robust(frame_files=frame_files, 
                                                  num_beams=64, 
                                                  min_dist=0, 
                                                  max_dist=200)
    print("提取的64线垂直角度（度）:")
    print(np.round(final_angles, 2))

    # 生成YAML
    generate_lidar_yaml_with_angles(
        final_angles,
        template_yaml_path='config/lidar_config_128.yml',  # 或之前输出的模板
        output_yaml_path='config/extracted_peak_64line_config.yml')