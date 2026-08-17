import os
import yaml

class DatasetConfig:
    """
    V2X 数据路径相关配置

    根据实验步骤，设置基本数据路径

    TODO:
    1. 保存数据目录下的所有场景
    2. 保存
    """
    def __init__(self, dataset_name="test", rq_name="demo"):
        self.dataset = dataset_name
        self.rq_name = rq_name

        self.dataset_root = ""
        self.scenes_data_dict = {}
        self.reinitialize()

        # 根据 rq 类型命名数据保存路径
        self.gen_data_save_dir = os.path.join(self.dataset_root, self.rq_name)

    def reinitialize(self):
        """
        加载基本数据路径
        :return:
        """
        with open("config/dataset_config.yml", "r") as config_file:
            dataset_dict = yaml.load(config_file, Loader=yaml.FullLoader)

        self.dataset_root = dataset_dict["dataset_path"]

        dataset_path = os.path.join(self.dataset_root, self.dataset)

        # 场景文件列表
        scene_folders = sorted([
            os.path.join(dataset_path, x)
            for x in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, x, "0", "bin"))
        ])

        for scene in scene_folders:
            pcd_file_path = os.path.join(scene, "0", "bin")
            file_ids = [
                os.path.splitext(f)[0]  # 分离文件名和扩展名，取前面的文件名作为 id
                for f in os.listdir(pcd_file_path)
                if os.path.isfile(os.path.join(pcd_file_path, f))
            ]

            # file_numbers = len([f for f in os.listdir(pcd_file_path)
            #                 if os.path.isfile(os.path.join(pcd_file_path, f))])
                            
            self.scenes_data_dict[os.path.basename(scene)] = sorted([int(index) for index in file_ids])




















