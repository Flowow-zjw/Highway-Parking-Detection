import os
from mmcv import Config
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

def main():
    config_path = os.path.join('configs', 'highway_parking_detection', 'rtm_det_custom.py')
    cfg = Config.fromfile(config_path)

    # 设置随机种子保证可复现
    set_random_seed(0, deterministic=False)

    # 构建数据集
    datasets = [build_dataset(cfg.data.train)]

    # 构建模型
    model = build_detector(cfg.model)

    # 加载预训练权重
    if cfg.load_from:
        from mmcv.runner import load_checkpoint
        load_checkpoint(model, cfg.load_from, map_location='cpu')

    # 开始训练
    train_detector(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':
    main()