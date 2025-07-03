from mmdet.datasets import DOTADataset

class CODroneDataset(DOTADataset):
    CLASSES = (
        'car','people','motor','truck','traffic-sign','boat','traffic-light','ship','bicycle','tricycle','tricycle','bridge','bus','ignored'
    )