import math
from typing import List, Tuple
from typing import List, Dict, Optional, Tuple
class SpeedEstimator:
    """
    速度估计模块
    输入：单辆车的轨迹点序列（frame_id, bbox, timestamp）
    输出：车辆当前速度（单位：m/s）

    说明：
    - bbox格式：[x1, y1, x2, y2]，像素坐标
    - 需要外部注入像素到实际距离的转换函数 pixel_to_meter()
    """

    def __init__(self, pixel_to_meter_func):
        """
        :param pixel_to_meter_func: 像素距离转米距离函数，signature: float pixel_dist -> float meter_dist
        """
        self.pixel_to_meter = pixel_to_meter_func

    def bbox_center(self, bbox: List[int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def compute_speed(self, track_points: List[Dict]) -> Optional[float]:
        """
        计算速度，基于最近两个轨迹点的中心点距离和时间差计算速度

        :param track_points: 轨迹点序列，有序（时间升序），至少包含2个点
        :return: 速度 m/s，若数据不足返回 None
        """
        if len(track_points) < 2:
            return None

        p1 = track_points[-2]
        p2 = track_points[-1]

        c1 = self.bbox_center(p1['bbox'])
        c2 = self.bbox_center(p2['bbox'])

        dt = p2['timestamp'] - p1['timestamp']
        if dt <= 0:
            return None

        dist_pixel = math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)
        dist_meter = self.pixel_to_meter(dist_pixel)
        speed = dist_meter / dt
        return speed