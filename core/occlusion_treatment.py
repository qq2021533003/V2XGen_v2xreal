import numpy as np
import os
import config
import math
import open3d as o3d
import utils.common_utils as common
import utils.visual as vis

from build.mtest.utils.Utils_o3d import pc_numpy_2_o3d, load_normalized_mesh_obj
from build.mtest.core.pose_estimulation.pose_generator import tranform_mesh_by_pose
from utils.common_utils import pc_numpy_2_o3d


def occlusion_rate_calculate(obj_pcd, v2x_info, high=0):
    # 改成计算 obj_pcd 被场景内"所有"其他对象遮挡的概率
    occ_rates = {}
    bg_pcd = pc_numpy_2_o3d(v2x_info.pc)

    for idx, obj_info in v2x_info.objs_info.items():
    # for idx, obj_info in v2x_info.vehicles_info.items():
        vehicle_pcd = common.get_obj_pcd_from_corner(obj_info["corner"], bg_pcd)
        try:
            tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(vehicle_pcd)
            mesh_obj = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(vehicle_pcd, 10, tetra_mesh,                                                                           pt_map)
        except:
            occ_rates[idx] = 0
            continue

        shadow_mask, shadow_mesh = get_delete_points_idx(mesh_obj, obj_pcd, high)

        # vis.show_mesh_with_pcd(shadow_mesh, obj_pcd)

        occ_rates[idx] = np.sum(shadow_mask) / shadow_mask.size

    return occ_rates


def insert_occlusion_detect(mesh_obj, v2x_info, high):
    occlusion_result = {}

    bg_pcd = pc_numpy_2_o3d(v2x_info.pc)
    shadow_mask, _ = get_delete_points_idx(mesh_obj, bg_pcd, high)
    if np.sum(shadow_mask) == 0:
        return {}
    shadow_pc = o3d.utility.Vector3dVector(np.array(bg_pcd.points)[shadow_mask])
    shadow_pcd = pc_numpy_2_o3d(shadow_pc)

    for idx, vehicle_info in v2x_info.vehicles_info.items():
        obj_pcd = common.get_obj_pcd_from_corner(vehicle_info["corner"], bg_pcd)

        distances = np.asarray(obj_pcd.compute_point_cloud_distance(shadow_pcd))

        occlusion_rate = np.sum(distances == 0) / distances.size
        if occlusion_rate > 0.9:
            occlusion_result[idx] = "full"

    return occlusion_result


def filter_part_occlusion_bg(v2x_info, car_id):
    bg_pcd = pc_numpy_2_o3d(v2x_info.pc)
    vehicle_pcd = common.get_obj_pcd_from_corner(v2x_info.vehicles_info[car_id]['corner'], bg_pcd)
    try:
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(vehicle_pcd)
        mesh_obj = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(vehicle_pcd, 10, tetra_mesh, pt_map)
    except:
        return

    delete_mask, _ = get_delete_points_idx(mesh_obj, bg_pcd)
    bg_pc = np.asarray(common.pcd_to_np(bg_pcd))[~delete_mask]
    bg_pcd = common.pc_numpy_2_o3d(bg_pc)
    v2x_info.pc = common.pcd_to_np(vehicle_pcd + bg_pcd)


def delete_occlusion_detect(deleted_v2x_info, other_v2x_info, mesh_obj):
    part_occlusion_result = []
    full_occlusion_result = []
    shadow_mesh = get_lidar_shadow_mesh(mesh_obj)
    bg_pcd = pc_numpy_2_o3d(deleted_v2x_info.pc)

    for idx, vehicle_info in deleted_v2x_info.vehicles_info.items():
        vehicle_pcd = common.get_obj_pcd_from_corner(vehicle_info['corner'], bg_pcd)
        shadow_mask = check_points_inclusion_mesh(vehicle_pcd, shadow_mesh)
        if np.sum(shadow_mask) == 0:
            continue
        else:
            part_occlusion_result.append(idx)

    for idx, vehicle_info in other_v2x_info.vehicles_info.items():
        vehicle_pcd = common.get_obj_pcd_from_corner(vehicle_info['corner'], bg_pcd)

        transformed_vehicle_pc = common.points_system_transform(vehicle_pcd.points, other_v2x_info.lidar_pose,
                                                                deleted_v2x_info.lidar_pose)
        shadow_mask = check_points_inclusion_mesh(pc_numpy_2_o3d(transformed_vehicle_pc),
                                                        shadow_mesh)
        if np.sum(shadow_mask) == 0:
            continue
        else:
            full_occlusion_result.append(idx)

    return part_occlusion_result, full_occlusion_result


centerCamPoint = np.asarray([0, 0, 0])


def get_lidar_shadow_mesh(mesh):
    global centerCamPoint
    hull, _ = mesh.compute_convex_hull()
    hullVertices = np.asarray(hull.vertices)

    castHullPoints = np.array([])

    for point1 in hullVertices:
        ba = centerCamPoint - point1
        baLen = math.sqrt((ba[0] * ba[0]) + (ba[1] * ba[1]) + (ba[2] * ba[2]))
        ba2 = ba / baLen
        pt2 = centerCamPoint + ((-200) * ba2)
        if np.size(castHullPoints):
            castHullPoints = np.vstack((castHullPoints, [pt2]))
        else:
            castHullPoints = np.array([pt2])

    pcdCastHull = o3d.geometry.PointCloud()
    pcdCastHull.points = o3d.utility.Vector3dVector(castHullPoints)

    hull2, _ = pcdCastHull.compute_convex_hull()
    hull2Vertices = np.asarray(hull2.vertices)

    combinedVertices = np.vstack((hullVertices, hull2Vertices))

    pcdShadow = o3d.geometry.PointCloud()
    pcdShadow.points = o3d.utility.Vector3dVector(combinedVertices)

    shadowMesh, _ = pcdShadow.compute_convex_hull()
    return shadowMesh


def get_delete_points_idx(obj, bg, high=0):
    if isinstance(obj, np.ndarray):
        obj_pcd = pc_numpy_2_o3d(obj)
        mesh = obj_pcd.compute_convex_hull()
    elif isinstance(obj, o3d.geometry.PointCloud):
        mesh = obj.compute_convex_hull()
    elif isinstance(obj, o3d.geometry.TriangleMesh):
        mesh = obj
    else:
        raise ValueError()
    global centerCamPoint

    centerCamPoint[2] = high

    obj_shadow_mesh = get_lidar_shadow_mesh(mesh)

    occ_delete_mask = check_points_inclusion_mesh(bg, obj_shadow_mesh)

    return occ_delete_mask, obj_shadow_mesh


def check_points_inclusion_mesh(pcd, mesh):
    if isinstance(pcd, o3d.geometry.PointCloud):
        points = np.asarray(pcd.points)
    else:
        points = pcd

    legacyMesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()

    _ = scene.add_triangles(legacyMesh)

    query_point = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)

    occupancy = scene.compute_occupancy(query_point).numpy()

    mask = (occupancy == 1)

    return mask


def get_base_insert_shadowed_obj(obj, box):
    center = centerCamPoint
    box_points = np.asarray(box.get_box_points())
    box_points_xy = box_points[:, :2]    # (x,y)

    angles = np.arctan2(box_points_xy[:, 1], box_points_xy[:, 0])
    min_index = np.argmin(angles)
    max_index = np.argmax(angles)

    center_xy = center[:2]
    points = np.asarray(obj.points)
    points_xy = points[:, :2]

    inside_pts_index = check_points_inclusion_triangle(points_xy, center_xy, box_points_xy[min_index], box_points_xy[max_index])
    inside_pts = points[inside_pts_index]
    obj.points = o3d.utility.Vector3dVector(inside_pts)


def check_points_inclusion_triangle(points, pt1, pt2, pt3):
    def get_triangle_area(p1, p2, p3):
        return 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))

    # area of original triangle
    triangle_area = get_triangle_area(pt1, pt2, pt3)

    inside_mask = []
    for i, point in enumerate(points):
        area_pt12 = get_triangle_area(point, pt1, pt2)
        area_pt13 = get_triangle_area(point, pt1, pt3)
        area_pt23 = get_triangle_area(point, pt2, pt3)
        if np.isclose(triangle_area, area_pt12 + area_pt13 + area_pt23):
            inside_mask.append(i)

    return inside_mask


def get_occ_rate_of_obj(v2x_info, occ_car_center, occ_car_degree, car_id):
    obj_filename = config.common_config.obj_filename
    assets_dir = config.common_config.obj_dir_path
    obj_car_dirs = os.listdir(config.common_config.obj_dir_path)
    obj_num = len(obj_car_dirs)
    objs_index = np.random.randint(1, obj_num)
    obj_mesh_path = os.path.join(assets_dir, obj_car_dirs[objs_index], obj_filename)
    mesh_obj_initial = load_normalized_mesh_obj(obj_mesh_path)

    mesh_obj = tranform_mesh_by_pose(mesh_obj_initial, occ_car_center, occ_car_degree)
    pcd_obj = mesh_obj.sample_points_uniformly(number_of_points=5000)

    occ_rate_dict = occlusion_rate_calculate(pcd_obj, v2x_info)
    res_occ_rate = 0

    bg_pcd = pcd_obj + pc_numpy_2_o3d(v2x_info.pc)


    for i, occ_rate in occ_rate_dict.items():
        # 排除计算被遮挡率的车辆本身
        if v2x_info.is_ego and car_id == i:
            continue
        # if car_id == i:
        #     continue

        if occ_rate > res_occ_rate:
            res_occ_rate = occ_rate
            # print("car id = ", car_id, "; obj id = ", i, "; rate = ", occ_rate)

    return res_occ_rate









