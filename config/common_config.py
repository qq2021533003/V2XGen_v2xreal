import os
occlusion_th = 0.95
occ_point_max = 20

project_dir = "./"

assets_dir = "{}/_assets".format(project_dir)
obj_dir_path = "{}/shapenet".format(assets_dir)
obj_cp_dir = "{}/copy_paste".format(assets_dir)

obj_filename = "models/model_normalized.gltf"
#
multi_scale = 5.5

# TODO: 是否做大型车辆 truck 的变换
# 车辆类别映射
SUPER_CLASS_MAP = {
    "vehicle": ["LongVehicle", "Car", "PoliceCar"],
    "pedestrian": ["Child", "RoadWorker", "Pedestrian", "Scooter",
                   "ScooterRider", "Motorcycle", "MotorcyleRider",
                   "BicycleRider"],
    "truck": ["Truck", "Van", "TrashCan", "ConcreteTruck", "Bus"],
}

VEHICLE_CLASS_ID = 1 