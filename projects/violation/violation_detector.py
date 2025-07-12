# 文件路径：HIGHWAY-PAYKING-DETECTION/projects/violation/violation_detector.py

import numpy as np
from typing import List, Dict, Optional


class ViolationDetector:
    """
    违规停车判定模块

    判定条件：
    - 车辆速度低于阈值 speed_threshold（m/s）
    - 车辆持续静止时间超过 time_threshold（秒）
    - 车辆在禁停区域内（seg_mask中对应像素为1）

    输入：
    - tracks: 当前帧所有车辆信息 [{"track_id": int, "bbox": [x1,y1,x2,y2], "timestamp": float}, ...]
    - speed_dict: {track_id: speed_in_m_s, ...}
    - seg_mask: np.ndarray 二值禁停区掩码，shape与视频帧分辨率一致，1表示禁停区
    """

    def __init__(self, speed_threshold: float = 0.5, time_threshold: float = 180):
        """
        :param speed_threshold: 速度阈值，单位m/s，低于视为停车
        :param time_threshold: 静止持续时间阈值，单位秒，超过则判定违规
        """
        self.speed_threshold = speed_threshold
        self.time_threshold = time_threshold
        # 用于持续静止时间统计，格式：{track_id: [last_stop_start_timestamp, last_timestamp]}
        self.stop_time_records = {}

    def is_inside_no_parking_zone(self, bbox: List[int], seg_mask: np.ndarray) -> bool:
        """
        判断车辆是否处于禁停区
        判断规则：车辆中心点像素在禁停区掩码中值为1即判定为禁停区内

        :param bbox: 车辆边界框 [x1,y1,x2,y2]
        :param seg_mask: 禁停区掩码二值图
        :return: True/False
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        h, w = seg_mask.shape
        cx = np.clip(cx, 0, w - 1)
        cy = np.clip(cy, 0, h - 1)

        return seg_mask[cy, cx] == 1

    def update_stop_time(self, track_id: int, timestamp: float, is_stopped: bool):
        """
        更新车辆静止时间记录

        :param track_id: 车辆ID
        :param timestamp: 当前时间戳
        :param is_stopped: 车辆当前是否静止
        """
        if not is_stopped:
            if track_id in self.stop_time_records:
                del self.stop_time_records[track_id]
            return

        if track_id not in self.stop_time_records:
            self.stop_time_records[track_id] = [timestamp, timestamp]
        else:
            self.stop_time_records[track_id][1] = timestamp

    def get_stop_duration(self, track_id: int) -> float:
        """
        获取车辆持续静止时间（秒）

        :param track_id:
        :return: 持续静止时间，若无记录返回0
        """
        if track_id not in self.stop_time_records:
            return 0.0
        start_t, end_t = self.stop_time_records[track_id]
        return end_t - start_t

    def detect_violations(
            self,
            tracks: List[Dict],
            speed_dict: Dict[int, Optional[float]],
            seg_mask: np.ndarray
    ) -> List[Dict]:
        """
        进行违规停车判定

        :param tracks: 当前帧车辆状态列表
        :param speed_dict: 车辆速度字典，{track_id: speed}
        :param seg_mask: 禁停区掩码
        :return: 违规车辆列表，包含track_id, bbox, speed, stop_duration
        """
        violation_list = []

        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            timestamp = track['timestamp']
            speed = speed_dict.get(track_id, None)

            if speed is None:
                # 速度未知，跳过判定
                continue

            # 判断是否静止
            is_stopped = speed < self.speed_threshold
            self.update_stop_time(track_id, timestamp, is_stopped)
            stop_duration = self.get_stop_duration(track_id)

            # 判断是否禁停区内
            in_no_parking = self.is_inside_no_parking_zone(bbox, seg_mask)

            if is_stopped and stop_duration >= self.time_threshold and in_no_parking:
                violation_list.append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'speed': speed,
                    'stop_duration': stop_duration,
                    'timestamp': timestamp
                })

        return violation_list