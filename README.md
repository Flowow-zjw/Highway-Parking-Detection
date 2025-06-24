```markdown
# Highway-Parking-Detection

## 项目简介

本项目基于无人机高空视频监控，结合先进的深度学习技术（目标检测、多目标跟踪、语义分割等），实现高速公路违规停车车辆的自动检测与报警。系统通过无人机采集的视频数据，实时识别并定位违规停车行为，有效提升交通管理和高速公路安全水平。

项目依托 [MMDetection](https://github.com/open-mmlab/mmdetection) 等开源深度学习工具，支持模型训练、推理及端到端部署，适配无人机视角的复杂交通场景，具备良好的鲁棒性和扩展性。

---

## 目录结构

```
├── configs/                    # 模型配置文件
│   └── highway_parking_detection/
│       └── rtm_det_custom.py   # RTMDet自定义配置
├── projects/                   # 主要项目源码
│   └── highway_parking_detection/
│       ├── __init__.py
│       ├── dataset.py          # 自定义数据集定义
│       ├── model.py            # 模型定义与封装
│       ├── train.py            # 训练主脚本
│       ├── inference.py        # 推理主脚本（检测+跟踪+违规判定）
│       ├── utils.py            # 工具函数（速度计算、图像处理等）
│       ├── tracking.py         # 多目标跟踪接口封装
│       ├── segmentation.py     # 语义分割模块调用
│       ├── violation.py        # 违规停车判定逻辑
│       └── alarm.py            # 报警事件处理
├── tools/                      # 辅助脚本
│   ├── train_detection.py      # 检测模型训练入口
│   ├── train_segmentation.py   # 语义分割训练入口
│   ├── evaluate.py             # 模型评估脚本
│   ├── demo_inference.py       # 演示推理脚本
│   └── convert_weights.py      # 权重格式转换工具
├── data/                       # 数据集（不纳入版本控制）
│   ├── detection/              # 车辆检测数据
│   ├── segmentation/           # 语义分割数据
│   └── raw_videos/             # 无人机原始视频备份
├── checkpoints/                # 模型权重文件（不纳入版本控制）
├── logs/                       # 训练日志及TensorBoard文件
├── README.md                   # 项目说明文档
├── requirements.txt            # 依赖环境列表
└── LICENSE                    # 版权许可文件
```

---

## 快速开始

### 环境搭建

1. 克隆仓库：
   ```bash
   git clone https://github.com/<your-username>/Highway-Parking-Detection.git
   cd Highway-Parking-Detection
   ```

2. 创建并激活Python虚拟环境（推荐使用conda或venv）

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

### 数据准备

- 按照目录结构准备车辆检测和语义分割数据，数据格式遵循COCO标准。
- 将无人机采集的视频备份放入 `data/raw_videos`。

### 模型训练

- 训练检测模型：
  ```bash
  python tools/train_detection.py --config configs/highway_parking_detection/rtm_det_custom.py
  ```

- 训练语义分割模型：
  ```bash
  python tools/train_segmentation.py --config configs/segmentation_config.py
  ```

### 模型推理与违规判定

- 使用推理脚本进行车辆检测、跟踪和违规停车判断：
  ```bash
  python projects/highway_parking_detection/inference.py --video path/to/video.mp4 --config configs/highway_parking_detection/rtm_det_custom.py --checkpoint checkpoints/rtm_det_tiny.pth
  ```

---

## 贡献指南

欢迎大家提交Issue和Pull Request，共同完善项目：

- Fork 本仓库，创建 feature 分支开发新功能
- 保持代码风格统一，提交前请运行代码检查和测试
- PR 请附带详细描述和测试步骤

---

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

---
