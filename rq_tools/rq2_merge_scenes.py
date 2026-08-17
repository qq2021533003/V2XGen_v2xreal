import sys
import shutil
from pathlib import Path

def merge_and_reindex_scenes(base_path: str, output_path: str):
    """
    遍历 trans_M1, trans_M2, trans_M3，将相同场景的数据汇总到 output_path，
    并从 000000 开始为 0(Ego) 和 1(CP) 的 .bin 和 .yaml 重新排序命名。
    """
    base_dir = Path(base_path)
    out_dir = Path(output_path)
    
    # 定义需要按顺序合并的文件夹源
    source_folders = ["trans_M1", "trans_M2", "trans_M3"]
    
    # 1. 收集所有存在的场景名称 (取并集)
    all_scenes = set()
    for folder_name in source_folders:
        folder_path = base_dir / folder_name
        if folder_path.exists():
            for d in folder_path.iterdir():
                if d.is_dir():
                    all_scenes.add(d.name)
                    
    if not all_scenes:
        print(f"[Error] 在 {base_dir} 下未找到任何场景数据。")
        sys.exit(1)

    print(f"共发现 {len(all_scenes)} 个独立场景，准备合并重命名至: {out_dir}\n")

    # 2. 逐个场景进行处理
    for scene_name in sorted(all_scenes):
        print(f"正在处理场景: {scene_name}")
        
        # 创建输出目录结构 output/scene_name/0 和 output/scene_name/1
        out_scene_0 = out_dir / scene_name / '0'
        out_scene_1 = out_dir / scene_name / '1'
        out_scene_0.mkdir(parents=True, exist_ok=True)
        out_scene_1.mkdir(parents=True, exist_ok=True)
        
        global_frame_idx = 0  # 该场景的全局帧索引，从 0 开始
        
        # 按 M1 -> M2 -> M3 的顺序读取
        for folder_name in source_folders:
            source_scene = base_dir / folder_name / scene_name
            source_0 = source_scene / '0'
            source_1 = source_scene / '1'
            
            if not source_0.exists() or not source_1.exists():
                continue # 如果该阶段（如 M2）没有这个场景，则跳过
            
            # 以 '0' 目录下的 .bin 文件作为基准进行遍历，保证严格的顺序
            # 找到所有的 bin 文件，提取前缀名并排序 (例如 '000123')
            bin_files = sorted([f.stem for f in source_0.iterdir() if f.suffix == '.bin'])
            
            for original_prefix in bin_files:
                # 构建原文件的完整路径
                orig_0_bin = source_0 / f"{original_prefix}.bin"
                orig_0_yaml = source_0 / f"{original_prefix}.yaml"
                orig_1_bin = source_1 / f"{original_prefix}.bin"
                orig_1_yaml = source_1 / f"{original_prefix}.yaml"
                
                # 安全校验：确保四个文件都存在才进行拷贝
                if all(f.exists() for f in [orig_0_bin, orig_0_yaml, orig_1_bin, orig_1_yaml]):
                    # 格式化新的文件名 (6位数字前补0)
                    new_filename = f"{global_frame_idx:06d}"
                    
                    # 拷贝并重命名文件
                    shutil.copy2(orig_0_bin, out_scene_0 / f"{new_filename}.bin")
                    shutil.copy2(orig_0_yaml, out_scene_0 / f"{new_filename}.yaml")
                    shutil.copy2(orig_1_bin, out_scene_1 / f"{new_filename}.bin")
                    shutil.copy2(orig_1_yaml, out_scene_1 / f"{new_filename}.yaml")
                    
                    global_frame_idx += 1
                else:
                    print(f"  [Warning] {folder_name} 中 {original_prefix} 的文件不完整，已跳过。")
        
        print(f"  -> 完成合并，该场景共生成 {global_frame_idx} 帧连续数据。\n")

    print("-" * 50)
    print("数据重命名与合并已全部完成！")

if __name__ == "__main__":
    # # 输入路径 (rq2_gen 文件夹)
    input_directory = "/mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset/rq2/rq2_gen"
    
    # 输出路径 (建议新建一个目录以防覆盖)
    output_directory = "/mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset/rq2/rq2_gen_merged"

    # 输入路径 (rq2_gen 文件夹)
    # input_directory = "/mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset/rq3/rq3_gen"
    
    # # 输出路径 (建议新建一个目录以防覆盖)
    # output_directory = "/mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset/rq3/rq3_gen_merged"
    
    merge_and_reindex_scenes(input_directory, output_directory)

    