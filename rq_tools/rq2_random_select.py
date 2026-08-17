import os
import shutil
import random
import argparse
from pathlib import Path

def sample_and_copy_per_scene(
    src_root: str,
    dst_root: str,
    ratio: float = 0.5,
    seed: int = 42
): 
    """
    对每个场景独立随机选取一定比例的数据帧，并同步复制 0/ 和 1/ 内的 bin 和 yaml 文件。

    Args:
        src_root: train 文件夹路径
        dst_root: 目标根目录路径
        ratio: 选取比例 (0.0 ~ 1.0)
        seed: 随机种子，保证可复现
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    random.seed(seed)

    # 获取所有场景文件夹（包含 0/ 和 1/ 子目录的）
    scene_dirs = sorted([
        d for d in src_root.iterdir()
        if d.is_dir() and (d / '0').exists() and (d / '1').exists()
    ])
    if not scene_dirs:
        raise ValueError(f"未找到有效的场景目录于 {src_root}")

    print(f"发现 {len(scene_dirs)} 个场景。")

    total_selected_frames = 0

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name

        # 从 0/ 目录提取该场景的所有 bin 文件序号
        bin_dir_0 = scene_dir / '0'
        bin_files_0 = sorted(bin_dir_0.glob('*.bin'))
        if not bin_files_0:
            print(f"警告：场景 {scene_name} 的 0/ 目录下没有 .bin 文件，跳过。")
            continue

        # 文件序号列表（无后缀的数字名）
        indices = [f.stem for f in bin_files_0]
        total = len(indices)

        # 按比例计算选取数量（至少 1 帧）
        select_count = max(1, int(total * ratio))
        # 打乱序号
        shuffled = indices.copy()
        random.shuffle(shuffled)
        selected = set(shuffled[:select_count])

        total_selected_frames += len(selected)
        print(f"场景 {scene_name}: 总 {total} 帧 -> 选取 {len(selected)} 帧")

        # 复制 0/ 和 1/ 下的相应文件
        for sub in ['0', '1']:
            src_sub = scene_dir / sub
            dst_sub = dst_root / scene_name / sub
            dst_sub.mkdir(parents=True, exist_ok=True)

            for idx in selected:
                bin_src = src_sub / f"{idx}.bin"
                yaml_src = src_sub / f"{idx}.yaml"
                if bin_src.exists():
                    shutil.copy2(bin_src, dst_sub / f"{idx}.bin")
                else:
                    print(f"警告: 缺失 {bin_src}")
                if yaml_src.exists():
                    shutil.copy2(yaml_src, dst_sub / f"{idx}.yaml")
                else:
                    print(f"警告: 缺失 {yaml_src}")

    print(f"全部完成！共选取 {total_selected_frames} 帧，保存至 {dst_root}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="按场景独立抽样复制 V2X 数据集（支持各场景数量不同）")
    parser.add_argument('--src', type=str, required=True, help='train 目录路径')
    parser.add_argument('--dst', type=str, required=True, help='目标输出目录')
    parser.add_argument('--ratio', type=float, default=0.5, help='抽样比例 (0~1)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    sample_and_copy_per_scene(args.src, args.dst, args.ratio, args.seed)