# Highway-Parking-Detection

## 项目简介
基于无人机高空视频监控，结合深度学习技术（目标检测、多目标跟踪、语义分割等），实现高速公路违规停车车辆的自动检测与报警。系统特点：

- 实时识别并定位违规停车行为
- 基于[MMDetection](https://github.com/open-mmlab/mmdetection)框架开发
- 适配无人机视角复杂场景
- 支持端到端训练部署

## 目录结构
```text
.
├── configs/                        # 模型配置
│   └── highway_parking_detection/
│       └── rtm_det_custom.py       # RTMDet自定义配置
│
├── projects/                       # 核心代码
│   └── highway_parking_detection/
│       ├── __init__.py
│       ├── dataset.py              # 自定义数据集
│       ├── model.py                # 模型封装
│       ├── train.py                # 训练脚本
│       ├── inference.py            # 推理入口（检测+跟踪+违规判定）
│       ├── utils.py                # 工具函数
│       ├── tracking.py             # 多目标跟踪
│       ├── segmentation.py         # 语义分割
│       ├── violation.py            # 违规判定逻辑
│       └── alarm.py                # 报警处理
│
├── tools/                          # 辅助工具
│   ├── train_detection.py          # 检测训练入口
│   ├── train_segmentation.py       # 分割训练入口
│   ├── evaluate.py                 # 评估脚本
│   ├── demo_inference.py           # 演示脚本
│   └── convert_weights.py          # 权重转换
│
├── data/                           # 数据集（.gitignore）
│   ├── detection/                  # 车辆检测数据
│   ├── segmentation/               # 语义分割数据
│   └── raw_videos/                 # 原始视频
│
├── checkpoints/                    # 模型权重（.gitignore）
├── logs/                           # 训练日志
├── README.md                       # 说明文档
├── requirements.txt                # 依赖列表
└── LICENSE                         # 许可文件
```

## 快速开始

### 1. 环境配置
```bash
git clone https://github.com/Flowow-zjw/Highway-Parking-Detection
cd Highway-Parking-Detection

# 创建虚拟环境（任选一种）
conda create -n hpd python=3.8
python -m venv hpd-env

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
- 检测数据：`data/detection/`（COCO格式）
- 分割数据：`data/segmentation/`
- 视频样本：`data/raw_videos/`

### 3. 模型训练
```bash
# 训练检测模型
python tools/train_detection.py \
    --config configs/highway_parking_detection/rtm_det_custom.py

# 训练分割模型
python tools/train_segmentation.py \
    --config configs/segmentation_config.py
```

### 4. 推理演示
```bash
python projects/highway_parking_detection/inference.py \
    --video data/raw_videos/sample.mp4 \
    --config configs/highway_parking_detection/rtm_det_custom.py \
    --checkpoint checkpoints/rtm_det_tiny.pth
```

## 贡献指南
小组成员可以根据各自的分工，完善相应的内容，修改相应文件。先git clone 本框架，修改后上传到各自的新建分支。（请勿直接上传到main分支）


## 许可证
[MIT License](LICENSE) © 2023 Highway-Parking-Detection
```

