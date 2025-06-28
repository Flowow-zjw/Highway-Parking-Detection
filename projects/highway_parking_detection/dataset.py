from mmdet.datasets import DOTADataset

class CODroneDataset(DOTADataset):
    CLASSES = (
        'car', 'truck', 'bus', 'van', 'excavator', 'barrow',
        'tricycle', 'awning-tricycle', 'forklift', 'other', 'motor', 'person', 'bicycle'
    )