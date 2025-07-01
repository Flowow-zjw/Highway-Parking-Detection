import os
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import set_random_seed, train_detector

def main():
    # 配置文件相对于项目根目录的路径
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'detection', 'rtmdet_custom.py')
    cfg = Config.fromfile(config_path)

    set_random_seed(0, deterministic=False)

    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model)

    if cfg.load_from:
        from mmcv.runner import load_checkpoint
        load_checkpoint(model, cfg.load_from, map_location='cpu')

    train_detector(model, datasets, cfg, distributed=False, validate=True)

if __name__ == '__main__':
    main()