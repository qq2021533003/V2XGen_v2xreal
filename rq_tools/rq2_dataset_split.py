import os
import shutil
import glob
import random
import argparse

def rq2_parser():
    parser = argparse.ArgumentParser(description="Generate valid frames for RQ2")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Test dataset dir.')
    opt = parser.parse_args()
    return opt


def rq2_split_dataset(dataset_dir):
    """
    1. Half of the sequences are randomly selected and saved as a training set for retrain
    and a test set for testing.
    2. copy selected data to new folder velodyne/labels/predictions.
    """
    dataset_root = os.path.dirname(dataset_dir)
    # rq_dataset = os.path.join(dataset_root, 'rq_dataset')
    rq_dataset =  os.path.join(dataset_root, 'rq2')

    # generate data and retrain
    rq2_select_dir = os.path.join(rq_dataset, 'train_20')

    # for test retrained models
    rq_test_dir = os.path.join(rq_dataset, 'train_80')
    
    # print(f"Dataset Root: {dataset_root}")
    print(f"Dataset: {dataset_dir}")
    print(f"Target Dirs : {rq2_select_dir} & {rq_test_dir}")

    scenes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    for scene in scenes:
        scene_path = os.path.join(dataset_dir, scene)
        
        sub_dirs = [d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d))]
        
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(scene_path, sub_dir)
            
            bin_dir = os.path.join(sub_dir_path, 'bin')
            yaml_dir = os.path.join(sub_dir_path, 'yaml')
            pred_dir = os.path.join(sub_dir_path, 'predictions')
            
            if not os.path.exists(bin_dir):
                continue
                
            bin_files = sorted([f for f in os.listdir(bin_dir) if f.endswith('.bin')])
            file_indices = [os.path.splitext(f)[0] for f in bin_files]
            
            if not file_indices:
                continue
                
            # 随机分为两半
            random.seed(42)  # 固定随机种子

            random.shuffle(file_indices)
            # half_idx = len(file_indices) // 2
            
            # select_indices = file_indices[:half_idx]
            # test_indices = file_indices[half_idx:]

            # 随机选取20%
            select_count = max(1, int(len(file_indices) * 0.2))
            select_indices = file_indices[:select_count]
            test_indices = file_indices[select_count:]
            
            def copy_files(indices, target_base_dir):
                target_sub_dir = os.path.join(target_base_dir, scene, sub_dir)
                target_bin = os.path.join(target_sub_dir, 'bin')
                target_yaml = os.path.join(target_sub_dir, 'yaml')
                target_pred = os.path.join(target_sub_dir, 'predictions')
                
                os.makedirs(target_bin, exist_ok=True)
                os.makedirs(target_yaml, exist_ok=True)
                os.makedirs(target_pred, exist_ok=True)
                
                for idx in indices:
                    # copy .bin
                    src_bin = os.path.join(bin_dir, f"{idx}.bin")
                    if os.path.exists(src_bin):
                        shutil.copy2(src_bin, os.path.join(target_bin, f"{idx}.bin"))
                        
                    # copy .yaml
                    src_yaml = os.path.join(yaml_dir, f"{idx}.yaml")
                    if os.path.exists(src_yaml):
                        shutil.copy2(src_yaml, os.path.join(target_yaml, f"{idx}.yaml"))
                        
                    # copy .label
                    if os.path.exists(pred_dir):
                        search_pattern = os.path.join(pred_dir, f"{idx}.*")
                        for pred_file in glob.glob(search_pattern):
                            file_name = os.path.basename(pred_file)
                            shutil.copy2(pred_file, os.path.join(target_pred, file_name))

            # 执行复制
            print(f"Processing {scene}/{sub_dir} ...")
            copy_files(select_indices, rq2_select_dir)
            # copy_files(test_indices, rq_test_dir)

    print("Finished")


if __name__ == '__main__':
    cmd_args = rq2_parser()

    rq2_split_dataset(cmd_args.dataset_dir)