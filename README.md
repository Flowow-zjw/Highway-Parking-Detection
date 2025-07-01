# Highway Parking Detection

🚗 高速公路车辆违规停车检测系统 — 基于无人机俯视视角的多目标检测与跟踪

---

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型推理](#模型推理)
- [模块说明](#模块说明)
- [FAQ](#faq)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 项目简介

本项目基于OpenMMLab生态（MMDetection、MMRotate、MMSegmentation等），实现高速公路无人机视角的车辆目标检测、多目标跟踪、语义分割禁停区识别、速度估计及违规停车报警。适用于高速公路监控、智能交通和安全管理。

---

## 功能特性

- **旋转目标检测**：采用旋转版RetinaNet，适配无人机俯视车辆检测
- **多目标跟踪**：集成DeepSort/ByteTrack模块，实现车辆跨帧身份保持
- **语义分割**：精准识别禁停区域，实现基于空间的违规判定
- **速度估计**：根据轨迹点结合无人机高度和相机参数，计算车辆速度
- **违规停车报警**：基于速度与禁停区掩码，实时检测违规停车并触发报警
- **模块化设计**：检测、跟踪、分割、速度估计、违规判定分层解耦，易扩展

---

## 项目结构

```
HIGHWAY-PAYKING-DETECTION/
│
├── configs/ # 配置文件
│ ├── detection/ # 目标检测模型配置（基于MMDetection）
│ │ └── rtmdet_custom.py
│ ├── segmentation/ # 语义分割模型配置（基于MMSegmentation）
│ │ └── parking_zone_seg.py
│ └── tracking/ # 跟踪算法相关配置（可选）
│
├── data/ # 数据集目录
│ ├── detection/ # 车辆检测数据（COCO格式）
│ │ ├── annotations/
│ │ ├── train/
│ │ ├── val/
│ │ └── test/
│ ├── segmentation/ # 语义分割数据
│ │ ├── images/
│ │ ├── masks/
│ │ └── annotations/
│
├── checkpoints/ # 训练好的模型权重文件
│ ├── rtmdet_tiny.pth
│ └── parking_zone_seg.pth
│
├── projects/ # 自定义项目代码
│ ├── detection/ # 车辆检测模块
│ │ ├── dataset.py # 数据集定义
│ │ ├── model.py # 模型构建
│ │ ├── train.py # 训练脚本
│ │ └── inference.py # 检测推理脚本
│ │
│ ├── tracking/ # 多目标跟踪模块
│ │ ├── deep_sort.py # DeepSort或其他跟踪实现
│ │ └── tracker.py # 跟踪接口封装
│ │
│ ├── segmentation/ # 语义分割模块
│ │ ├── dataset.py # 语义分割数据定义（可选）
│ │ ├── train.py # 训练脚本
│ │ └── inference.py # 推理脚本
│ │
│ ├── utils/ # 公共工具函数
│ │ ├── speed_estimator.py # 速度估计函数
│ │ ├── geo_utils.py # 经纬度与像素映射转换
│ │ └── visualization.py # 可视化工具
│ │
│ ├── violation/ # 违规检测与报警模块
│ │ ├── violation_detector.py # 违规判定逻辑
│ │ └── alarm.py # 报警系统
│ │
│ └── main.py # 主流程集成脚本（视频流处理，模块串联）
│
├── tools/ # 辅助工具脚本
│ ├── train_detection.py # 训练检测模型的启动脚本
│ ├── train_segmentation.py # 训练分割模型的启动脚本
│ └── evaluate.py # 评估模型性能脚本
│
├── requirements.txt # 依赖包列表
├── README.md # 项目说明文档
└── LICENSE # 许可证
```

---

## 环境依赖

建议使用Python 3.8+，并根据PyTorch版本及CUDA版本安装对应的`mmcv-full`。

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.1/index.html
pip install mmdet mmrotate mmsegmentation
pip install opencv-python numpy
```

> 详细依赖请见 `requirements.txt`

---

## 数据准备

- **车辆检测数据**  
  采用COCO格式或DOTA格式，放置于`data/detection/`对应目录，包含图片和标注。

- **语义分割数据**  
  禁停区语义分割数据集，放置于`data/segmentation/`目录。

- **视频数据**  
  无人机采集视频放入`data/raw_videos/`目录。

---

## 模型训练

进入对应模块目录训练：

```bash
# 车辆检测训练
cd projects/detection
python train.py

# 语义分割训练
cd projects/segmentation
python train.py
```

训练配置文件位于`configs/`，可根据需求修改。

---

## 模型推理

```bash
# 车辆检测推理
cd projects/detection
python inference.py

# 语义分割推理
cd projects/segmentation
python inference.py
```

推理结果默认保存在`output/`目录。

---

## 模块说明

- **detection/**  
  车辆检测模型训练与推理，基于MMRotate。

- **tracking/**  
  多目标跟踪算法封装，支持DeepSort、ByteTrack。

- **segmentation/**  
  禁停区语义分割模型训练与推理。

- **utils/**  
  速度估计、经纬度转换、可视化工具。

- **violation/**  
  违规停车判定逻辑及报警推送模块。

- **main.py**  
  多模块串联主流程入口。

---

## FAQ

**Q1:** 如何确保环境和依赖正确？  
**A:** 推荐使用`conda`或`venv`虚拟环境，确认`mmcv-full`版本与PyTorch及CUDA匹配。

**Q2:** 数据格式如何？  
**A:** 车辆检测支持COCO/DOTA格式，语义分割支持常见分割标注格式。

**Q3:** 如何添加自己的模型？  
**A:** 按照MMDetection/MMRotate标准流程，修改`configs/`配置文件及`projects/`代码。

---

## 贡献指南

欢迎通过Issue和Pull Request贡献代码和建议。请遵守代码规范和提交说明。

---

## 许可证

本项目采用MIT许可证，详见 LICENSE 文件。

