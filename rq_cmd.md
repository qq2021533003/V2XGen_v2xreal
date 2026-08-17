## Basic CMD

```shell
# 进入 wsl
wsl.exe -d Ubuntu

# conda 环境
conda activate v2xgen	# 项目通用环境
# conda activate v2xreal

CUDA Version: 13.2
```



## 模型训练

```shell
# 对现有模型进行训练
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_late_fusion.yaml --dataset_dir /mnt/g/v2x_dataset/V2X-Real/train_64 --model_dir trained_model/late_fusion

python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_early_fusion.yaml --dataset_dir /mnt/g/v2x_dataset/V2X-Real/train_64 --model_dir trained_model/early_fusion

python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml --dataset_dir /mnt/g/v2x_dataset/V2X-Real/train_64 
```



## 数据生成

```shell
# dataset path
python dataset_init_v2.py -d "/mnt/g/v2x_dataset/V2XGen_V2X-Real/test"

# 数据初始化
python dataset_init_v2.py -d "/mnt/g/v2x_dataset/v2_test/test"

# 数据生成测试
python v2/v2_gen_test.py -t insert
```



## v2x-real 项目

```shell
python opencood/tools/inference.py --model_dir trained_model/late_fusion --fusion_method late --dataset_dir /mnt/g/v2x_dataset/V2XGen_V2X-Real/test
```



```shell
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_late_fusion.yaml --half --dataset_dir /mnt/g/v2x_dataset/V2X-Real/train

python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_late_fusion.yaml --half --dataset_dir /mnt/g/v2x_dataset/V2X-Real/train_64
```



## RQ 实验

**生成数据 demo**

```shell
python v2/v2xreal_gen_demo.py -t insert -n 10
# -t 变换类型
# -n 每个场景选择的数据数量
```

**rq1 可视化**

```shell
python rq_tools/rq1_vis.py -t insert -n 1
# -t 变换类型
# -n 变换次数
```

**rq2 数据随机分为两半**

```shell
python rq_tools/rq2_dataset_split.py --dataset_dir "/mnt/g/v2x_dataset/V2XGen_V2X-Real/v2x-real_dataset_64"
```

**rq2 有效数据扩增**

```shell
python rq_tools/rq2_gen.py --dataset_dir "/mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset/rq2_gen"  --model_dir "trained_model"
```

**rq2 数据选择与保存**

```shell
# 1. 无融合获得仅 ego 视角下的检测结果
python rq_tools/rq2_inference.py --model_dir trained_model/late_fusion --dataset_dir /mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset/rq2_gen/rq2_gen_merged --fusion_method nofusion

# 2. 推理、选择并保存
python rq_tools/rq2_inference.py --dataset_dir /mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset/rq2/rq2_gen_merged --model_dir trained_model/late_fusion --fusion_method late
```

**rq3 重训练**

```shell
python rq_tools/train.py --dataset_dir /mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset/rq2/rq2_select --model_dir trained_model/late_fusion --method v2x_gen --scale 0.15

# method: v2x_gen/random/coo_test
# scale: 0.1/0.15
```

**rq3 推理**

```shell
python rq_tools/rq3_inference.py --dataset_dir /mnt/g/v2x_dataset/V2XGen_V2X-Real/rq_dataset/rq3/rq3_gen_merged --fusion_method late --model_dir trained_model/late_fusion --method v2x_gen --scale 0.15
```

























