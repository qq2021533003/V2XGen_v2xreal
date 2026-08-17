import torch
import config
import numpy as np
import open3d as o3d
import utils.common_utils as common

from opencood.utils import common_utils


def show_mesh_with_pcd(
        mesh, pcd):
    """
    visualize mesh and pcd
    :param mesh:
    :param pcd:
    :return:
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    vis.add_geometry(pcd)

    box3d = mesh.get_minimal_oriented_bounding_box()
    mesh.compute_vertex_normals()

    vis.add_geometry(mesh)
    vis.add_geometry(box3d)

    vis.run()
    vis.destroy_window()


def show_mesh_with_box(mesh_obj):
    """
    visualize mesh and bounding box
    :param mesh_obj:
    :return:
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    box3d = mesh_obj.get_minimal_oriented_bounding_box()
    box_points = box3d.get_box_points()

    # box_mesh.compute_vertex_normals()
    points = np.asarray(box_points)
    # print(points)
    # print("height = ", np.ptp(points[:, 2]))

    mesh_obj.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh_obj])
    vis.add_geometry(mesh_obj)

    vis.add_geometry(box3d)

    # vis.add_geometry(mixed_pcd)
    vis.run()
    vis.destroy_window()


def show_pc_with_box(pc, box):
    pcd = common.pc_numpy_2_o3d(pc)
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)
    vis.add_geometry(box)
    rgb_color = [245 / 255, 144 / 255, 1 / 255]

    pcd.paint_uniform_color(rgb_color)

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def show_bg_with_boxes(v2x_info):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    for val in v2x_info.vehicles_info.values():
        line_set = common.corner_to_line_set_box(val["corner"])
        vis.add_geometry(line_set)

    pcd = common.pc_numpy_2_o3d(v2x_info.pc)

    rgb_color = [245/255, 144/255, 1/255]

    pcd.paint_uniform_color(rgb_color)

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def show_obj_with_car_id(v2x_info, car_id):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    corner = v2x_info.vehicles_info[car_id]['corner']
    line_set = common.corner_to_line_set_box(corner)
    vis.add_geometry(line_set)

    # pcd.paint_uniform_color([0, 0, 0])
    pcd = common.pc_numpy_2_o3d(v2x_info.pc)
    if v2x_info.is_ego:
        pcd_color = [245 / 255, 144 / 255, 1 / 255]
    else:
        pcd_color = [1, 1, 1]
    pcd.paint_uniform_color(pcd_color)

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def show_obj_with_corner(v2x_info, corner):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size

    render.background_color = np.array(config.lidar_config.render_background_color)

    # line_set = common.corner_to_line_set_box(corner)
    lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7],
                          [4, 5], [5, 6], [6, 7], [7, 4]])

    cylinders = []

    for line in lines_box:
        point1 = corner[line[0]]
        point2 = corner[line[1]]

        cylinder = common.create_cylinder_between_points(point1, point2, radius=0.03)

        cylinder.paint_uniform_color([1, 0, 0])
        cylinders.append(cylinder)

    mesh = o3d.geometry.TriangleMesh()
    for cyl in cylinders:
        mesh += cyl
    # vis.add_geometry(line_set)
    vis.add_geometry(mesh)

    # pcd.paint_uniform_color([0, 0, 0])
    pcd = common.pc_numpy_2_o3d(v2x_info.pc)
    if v2x_info.is_ego:
        pcd_color = [0, 0, 1]
        pcd_color = [245 / 255, 144 / 255, 1 / 255]
    else:
        pcd_color = [0, 100 / 255, 0]
    pcd.paint_uniform_color(pcd_color)

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def show_pc_with_info(v2x_info):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    for val in v2x_info.vehicles_info.values():
        line_set = common.corner_to_line_set_box(val["corner"])
        vis.add_geometry(line_set)

    ego_numpy = v2x_info.pc
    ego_pcd = common.pc_numpy_2_o3d(ego_numpy)

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    vis.add_geometry(ego_pcd)

    vis.run()
    vis.destroy_window()


def show_pc(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    ego_pcd = common.pc_numpy_2_o3d(pcd)
    # ego_pcd = pcd

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    vis.add_geometry(ego_pcd)

    vis.run()
    vis.destroy_window()


def show_ego_and_cp_pc(ego_info, cp_info):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    for val in ego_info.vehicles_info.values():
        line_set = common.corner_to_line_set_box(val["corner"])
        vis.add_geometry(line_set)

    for val in cp_info.vehicles_info.values():
        line_set = common.corner_to_line_set_box(val["corner"])
        vis.add_geometry(line_set)

    # pcd.paint_uniform_color([0, 0, 0])
    T_cp2ego = np.linalg.inv(ego_info.lidar_pose) @ cp_info.lidar_pose
    ego_pcd = common.pc_numpy_2_o3d(ego_info.pc)
    cp_pcd = common.pc_numpy_2_o3d(cp_info.pc).transform(T_cp2ego)

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    vis.add_geometry(ego_pcd)
    cp_color = [1, 1, 1]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)

    vis.run()
    vis.destroy_window()


def show_ego_and_cp_with_id(ego_info, cp_info, ego_id, cp_id):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    # pcd.paint_uniform_color([0, 0, 0])
    T_cp2ego = np.linalg.inv(ego_info.lidar_pose) @ cp_info.lidar_pose
    ego_pcd = common.pc_numpy_2_o3d(ego_info.pc)
    cp_pcd = common.pc_numpy_2_o3d(cp_info.pc).transform(T_cp2ego)

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    cp_color = [1, 1, 1]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)

    corner = ego_info.vehicles_info[ego_id]['corner']
    line_set = common.corner_to_line_set_box(corner)
    vis.add_geometry(line_set)

    # 确定中心点位置
    # point = ego_info.vehicles_info[ego_id]['center']

    # print(point.shape)
    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    # sphere.translate(point.tolist())  # 移到目标点
    # sphere.paint_uniform_color([1, 0, 0])  # 红色

    # vis.add_geometry(sphere)

    vis.add_geometry(ego_pcd)
    vis.add_geometry(cp_pcd)

    vis.run()
    vis.destroy_window()


def show_ego_and_cp_with_corner(ego_info, cp_info, corner):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    # render.background_color = np.array(config.lidar_config.render_background_color)
    lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7],
                          [4, 5], [5, 6], [6, 7], [7, 4]])

    cylinders = []

    for line in lines_box:
        point1 = corner[line[0]]
        point2 = corner[line[1]]

        cylinder = common.create_cylinder_between_points(point1, point2, radius=0.05)

        cylinder.paint_uniform_color([1, 0, 0])
        cylinders.append(cylinder)

    mesh = o3d.geometry.TriangleMesh()
    for cyl in cylinders:
        mesh += cyl
    # vis.add_geometry(line_set)
    vis.add_geometry(mesh)

    # pcd.paint_uniform_color([0, 0, 0])
    T_cp2ego = np.linalg.inv(ego_info.lidar_pose) @ cp_info.lidar_pose
    ego_pcd = common.pc_numpy_2_o3d(ego_info.pc)
    cp_pcd = common.pc_numpy_2_o3d(cp_info.pc).transform(T_cp2ego)

    # ego_color = [0, 0, 1]
    cp_color = [0, 0, 1]
    ego_color = [0, 75 / 255, 0]
    ego_pcd.paint_uniform_color(ego_color)
    # cp_color = [0, 100 / 255, 0]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)

    line_set = common.corner_to_line_set_box(corner)
    vis.add_geometry(line_set)

    vis.add_geometry(ego_pcd)
    vis.add_geometry(cp_pcd)

    vis.run()
    vis.destroy_window()


def show_ego_and_cp_for_translation(ego_info, cp_info, car_id, corner):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    T_cp2ego = np.linalg.inv(ego_info.lidar_pose) @ cp_info.lidar_pose
    ego_pcd = common.pc_numpy_2_o3d(ego_info.pc)
    cp_pcd = common.pc_numpy_2_o3d(cp_info.pc).transform(T_cp2ego)

    ego_color = [245 / 255, 144 / 255, 1 / 255]
    ego_pcd.paint_uniform_color(ego_color)
    cp_color = [1, 1, 1]
    cp_pcd.paint_uniform_color(cp_color)
    vis.add_geometry(cp_pcd)

    ego_corner = ego_info.vehicles_info[car_id]['corner']
    ego_line_set = common.corner_to_line_set_box(ego_corner)
    vis.add_geometry(ego_line_set)

    line_set = common.corner_to_line_set_box(corner, [1, 1, 1])
    vis.add_geometry(line_set)

    vis.add_geometry(ego_pcd)
    vis.add_geometry(cp_pcd)

    vis.run()
    vis.destroy_window()


def show_obj_for_translation(v2x_info, car_id, corner):
    vis = o3d.visualization.Visualizer()
    vis.create_window(config.lidar_config.window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()

    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    pcd = common.pc_numpy_2_o3d(v2x_info.pc)

    if v2x_info.is_ego:
        pcd_color = [245 / 255, 144 / 255, 1 / 255]
    else:
        pcd_color = [1, 1, 1]
    pcd.paint_uniform_color(pcd_color)

    cur_corner = v2x_info.vehicles_info[car_id]['corner']
    cur_line_set = common.corner_to_line_set_box(cur_corner)
    vis.add_geometry(cur_line_set)

    line_set = common.corner_to_line_set_box(corner, [1, 1, 1])
    vis.add_geometry(line_set)

    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()


def visualize_and_eval_post_operation(v2x_info, pred_box_tensor, pred_scores, gt_box_tensor,
                                      iou_threshold=0.5, is_ego=True, window_name="Post-Operation Evaluation"):
    """
    在insert/translation/scaling/rotation操作后，可视化点云、预测框和GT框的重合程度，
    并统计基于IoU阈值的正确预测数量（TP）。

    :param v2x_info: V2XInfo对象，包含点云 (pc) 和车辆信息 (vehicles_info)
    :param pred_box_tensor: 模型预测的边界框张量 (N, 8, 3) torch.Tensor 或 np.ndarray
    :param pred_scores: 预测框置信度 (N,)
    :param gt_box_tensor: 操作后的GT边界框张量 (M, 8, 3) torch.Tensor 或 np.ndarray
    :param iou_threshold: IoU阈值，用于过滤和统计
    :param is_ego: 是否为ego车辆（影响点云颜色）
    :param window_name: 窗口名称
    """
    # 1. 数据是否是torch.Tensor，是的话转换为Numpy
    if isinstance(pred_box_tensor, torch.Tensor):
        pred_boxes = common_utils.torch_tensor_to_numpy(pred_box_tensor)
    else:
        pred_boxes = pred_box_tensor

    if isinstance(pred_scores, torch.Tensor):
        pred_scores = common_utils.torch_tensor_to_numpy(pred_scores)
    else:
        pred_scores = pred_scores

    if isinstance(gt_box_tensor, torch.Tensor):
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_box_tensor)
    else:
        gt_boxes = gt_box_tensor

    if len(pred_boxes) == 0:
        print("No predictions detected.")

    # 2. 计算IoU并筛选有效预测框（TP统计）
    det_polygons = common_utils.convert_format(pred_boxes)  # 转换为Shapely Polygon (BEV)
    gt_polygons = common_utils.convert_format(gt_boxes)

    valid_pred_indices = []  # IoU >= 阈值的预测框索引
    tp_count = 0  # 正确预测数量（TP）
    matched_gt_indices = set()  # 避免一个GT匹配多个预测
    for i, det_poly in enumerate(det_polygons):
        if len(gt_polygons) == 0:
            break
        ious = common_utils.compute_iou(det_poly, gt_polygons)
        max_iou = np.max(ious)  # 寻找最佳匹配
        if max_iou >= iou_threshold:  # 最大 IoU 超过设定的阈值（例如 0.5）时，才认为这个预测可能是一个 TP
            gt_idx = np.argmax(ious)
            if gt_idx not in matched_gt_indices:  # 1:1匹配 检查这个 gt_idx 是否已经被之前的预测框“占坑”了
                valid_pred_indices.append(i)
                matched_gt_indices.add(gt_idx)  # 如果没有被占用，记录该预测框索引 i，并将该真值标记为已匹配
                tp_count += 1

    # 打印统计结果
    print(f"Post-Operation Evaluation (IoU Threshold: {iou_threshold}):")
    print(f"Total GT Boxes: {len(gt_boxes)}")
    print(f"Total Pred Boxes: {len(pred_boxes)}")
    print(f"Correct Predictions (TP): {tp_count} (IoU >= {iou_threshold})")

    # 3. 创建Open3D可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name, width=config.lidar_config.window_width,
                      height=config.lidar_config.window_height)

    render = vis.get_render_option()
    render.point_size = config.lidar_config.render_point_size
    render.background_color = np.array(config.lidar_config.render_background_color)

    if isinstance(v2x_info, dict):
        pc_raw = v2x_info.get('pc', v2x_info.get('origin_lidar'))
    else:
        pc_raw = v2x_info.pc

    # 统一转为 CPU Numpy
    if torch.is_tensor(pc_raw):
        pc_np = pc_raw.detach().cpu().numpy()
    else:
        pc_np = np.array(pc_raw)

    if pc_np.ndim == 3:
        pc_np = pc_np.squeeze(0)
    # pcd.points = o3d.utility.Vector3dVector(pc_np[:, :3])

    pc_for_o3d = np.ascontiguousarray(pc_np[:, :3], dtype=np.float64)
    # 渲染背景点云
    pcd = o3d.geometry.PointCloud()
    try:
        pcd.points = o3d.utility.Vector3dVector(pc_for_o3d)
    except RuntimeError as e:
        print(f"Open3D casting error: {e}")
        pcd.points = o3d.utility.Vector3dVector(pc_for_o3d.astype(np.float64))
    pcd_color = [245 / 255, 144 / 255, 1 / 255] if is_ego else [1, 1, 1]  # ego橙色，其他白色
    pcd.paint_uniform_color(pcd_color)
    vis.add_geometry(pcd)

    # 定义边界框线序
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
             [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    # 渲染GT框（红色）
    for box in gt_boxes:
        ls = o3d.geometry.LineSet()
        box_np = np.ascontiguousarray(box, dtype=np.float64)
        ls.points = o3d.utility.Vector3dVector(box)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.paint_uniform_color([1, 0, 0])  # 红色
        vis.add_geometry(ls)

    # 渲染有效预测框（绿色，只渲染IoU >= 阈值的）
    for idx in valid_pred_indices:
        ls = o3d.geometry.LineSet()
        pred_box_np = np.ascontiguousarray(pred_boxes[idx], dtype=np.float64)
        ls.points = o3d.utility.Vector3dVector(pred_boxes[idx])
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.paint_uniform_color([0, 1, 0])  # 绿色
        vis.add_geometry(ls)

    # 弹出窗口并运行
    vis.run()
    vis.destroy_window()


def visualize_individual_perspectives(batch_data, ego_preds, cp_preds, frame_idx):
    """
    先后展示主车视角和协作车视角的感知结果。

    :param batch_data: 完整的 batch 数据字典
    :param ego_preds: tuple (pred_box, pred_score, gt_box) 为主车结果
    :param cp_preds: tuple (pred_box, pred_score, gt_box) 为协作车结果
    :param frame_idx: 当前帧序号
    """

    # 处理单个视角的渲染
    def render_single_view(v2x_dict, preds, window_name, is_ego=True):
        if preds[0] is None or len(preds[0]) == 0:
            print(f"Warning: No predictions to show for {window_name}")
            return

        # 1. 提取并转换点云 (关键修复：移动到 CPU)
        # pc_raw = v2x_dict.get('pc', v2x_dict.get('lidar_np'))
        pc_raw = None
        # 按照优先级尝试获取原始点云
        search_keys = ['pc', 'lidar_np', 'origin_lidar', 'processed_lidar']

        for key in search_keys:
            if key in v2x_dict:
                pc_raw = v2x_dict[key]
                # print(f"DEBUG: Key found = {key}")
                # print(f"DEBUG: pc_raw type = {type(pc_raw)}")
                # print(f"DEBUG: pc_raw device = {pc_raw.device if hasattr(pc_raw, 'device') else 'No Device'}")
                break

        if pc_raw is None:
            print(f"Error: Could not find point cloud in {v2x_dict.keys()}")
            return

        if isinstance(pc_raw, list):
            # 逐个将列表里的元素搬到 CPU
            processed_list = []
            for item in pc_raw:
                if hasattr(item, 'cpu'):
                    processed_list.append(item.detach().cpu().numpy())
                else:
                    processed_list.append(item)

            # 拼接成一个大的 Numpy 矩阵
            if len(processed_list) > 0:
                pc_np = np.concatenate(processed_list, axis=0) if isinstance(processed_list[0],
                                                                             np.ndarray) else np.array(processed_list)
            else:
                pc_np = np.empty((0, 3))

        # 情况 B：pc_raw 直接就是 Tensor
        elif hasattr(pc_raw, 'cpu'):
            pc_np = pc_raw.detach().cpu().numpy()

        # 情况 C：已经是 Numpy 或其他
        else:
            pc_np = np.array(pc_raw)

        # --- 后续处理保持不变 ---
        # 再次确保维度正确 (N, 3)
        if pc_np.ndim == 3:
            pc_np = pc_np.squeeze(0)

        # 如果列表转换后依然形状不对，强制转为 2D
        if pc_np.ndim == 1 and pc_np.size > 0:
            pc_np = pc_np.reshape(-1, 3)

        # 检查是否依然是标量（即报错的位置）
        if pc_np.ndim < 2:
            print(f"Error: Point cloud array dimension is too low: {pc_np.ndim}. Value: {pc_np}")
            return

        # 2. 提取预测框和真值框 (确保是 Numpy)
        pred_boxes = preds[0].detach().cpu().numpy() if torch.is_tensor(preds[0]) else preds[0]
        gt_boxes = preds[2].detach().cpu().numpy() if torch.is_tensor(preds[2]) else preds[2]

        # 3. 创建 Open3D 窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name, width=config.lidar_config.window_width,
                          height=config.lidar_config.window_height)

        # 背景和点云设置
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
        opt.point_size = 1.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np[:, :3])
        # 主车橙色，协作车白色
        pcd.paint_uniform_color([1, 0.7, 0.2] if is_ego else [0.8, 0.8, 0.8])

        vis.add_geometry(pcd)

        # 4. 绘制框的线序
        lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
                 [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

        # 渲染 GT (红色)
        if gt_boxes is not None and len(gt_boxes) > 0:
            for box in gt_boxes:
                # --- 新增转换逻辑 ---
                # 确保 box 是有效的数组/张量且形状正确 (N, 3)
                if not hasattr(box, '__len__') or isinstance(box, (int, float)):
                    continue
                if hasattr(box, 'cpu'):
                    box = box.detach().cpu().numpy()

                # 确保 box 至少是二维的 (8, 3)
                box_np = np.array(box)
                if box_np.ndim != 2:
                    continue
                # ------------------
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(box.astype(np.float64))
                ls.lines = o3d.utility.Vector2iVector(lines)
                ls.paint_uniform_color([1, 0, 0])
                vis.add_geometry(ls)

        # 渲染 Pred (绿色)
        if pred_boxes is not None:
            for box in pred_boxes:
                # --- 新增转换逻辑 ---
                if not hasattr(box, '__len__') or isinstance(box, (int, float)):
                    continue
                if hasattr(box, 'cpu'):
                    box = box.detach().cpu().numpy()

                box_np = np.array(box)
                if box_np.ndim != 2:
                    continue
                # ------------------
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(box.astype(np.float64))
                ls.lines = o3d.utility.Vector2iVector(lines)
                ls.paint_uniform_color([0, 1, 0])
                vis.add_geometry(ls)

        vis.run()
        vis.destroy_window()

    # --- 开始执行展示 ---

    # 视角一：主车
    if 'ego' in batch_data:
        render_single_view(batch_data['ego'], ego_preds, f"Frame {frame_idx} - EGO View", is_ego=True)

    # 视角二：协作车 (ID: 1)
    if '1' in batch_data:
        render_single_view(batch_data['1'], cp_preds, f"Frame {frame_idx} - CP ID:1 View", is_ego=False)





