import numpy as np
import cv2
import math
from typing import List, Dict, Any, Optional

# 导入ByteTracker
try:
    from yolox.tracker.byte_tracker import BYTETracker, STrack
    YOLOX_AVAILABLE = True
    print("ByteTracker可用")
except ImportError as e:
    YOLOX_AVAILABLE = False
    print(f"ByteTracker不可用，请安装YOLOX: pip install yolox\n错误: {e}")

def rotated_bbox_to_xyxy(bbox):
    """将旋转框转换为普通框 [x1, y1, x2, y2]"""
    if len(bbox) < 5:
        return bbox[:4]  # 已经是普通框
    
    # 旋转框格式 [cx, cy, w, h, angle]
    center = (bbox[0], bbox[1])
    size = (bbox[2], bbox[3])
    angle = bbox[4] * 180 / math.pi
    
    # 获取旋转矩形的四个角点
    rect = (center, size, angle)
    box_points = cv2.boxPoints(rect)
    
    # 计算包围盒
    x_coords = box_points[:, 0]
    y_coords = box_points[:, 1]
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    
    return [x1, y1, x2, y2]

class Detection:
    """检测结果类 - 兼容ByteTracker"""
    def __init__(self, bbox, confidence, class_name):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.confidence = confidence
        self.class_name = class_name
        
        # 转换为ByteTracker需要的xyxy格式
        self.xyxy = rotated_bbox_to_xyxy(self.bbox)
    
    @property
    def tlwh(self):
        """转换为tlwh格式"""
        xyxy = self.xyxy
        return np.array([xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]])
    
    @property
    def tlbr(self):
        """获取tlbr格式"""
        return np.array(self.xyxy)

class ByteTrackerWrapper:
    """ByteTracker包装器 - 保持与原DeepSORT接口兼容"""
    
    def __init__(self, frame_rate=30, track_thresh=0.6, track_buffer=30, 
                 match_thresh=0.9, min_box_area=10, mot20=False):
        
        if not YOLOX_AVAILABLE:
            raise ImportError("需要安装YOLOX才能使用ByteTracker: pip install yolox")
        
        # ByteTracker参数类
        class Args:
            def __init__(self):
                self.track_thresh = track_thresh      # 高置信度阈值
                self.track_buffer = track_buffer      # 跟踪缓冲帧数
                self.match_thresh = match_thresh      # 匹配阈值
                self.min_box_area = min_box_area      # 最小框面积
                self.mot20 = mot20                    # MOT20数据集格式
        
        self.args = Args()
        self.tracker = BYTETracker(self.args, frame_rate=frame_rate)
        
        # 轨迹管理
        self.all_tracks = {}  # 存储所有轨迹的历史
        self.class_counters = {}  # 每个类别的计数器
        self.track_class_map = {}  # 轨迹ID到类别的映射
        
        print(f"ByteTracker初始化完成")
        print(f"参数: track_thresh={track_thresh}, track_buffer={track_buffer}, match_thresh={match_thresh}")
    
    def update(self, detections, frame_id, timestamp, image=None):
        """
        更新ByteTracker
        
        Args:
            detections: Detection对象列表
            frame_id: 帧ID
            timestamp: 时间戳
            image: 原始图像 (ByteTracker不需要，保持接口兼容)
        
        Returns:
            tracks: 跟踪结果列表
        """
        
        # 按类别分组处理检测
        tracks_by_class = {}
        all_online_tracks = []
        
        # 按类别分别跟踪
        for class_name in set(det.class_name for det in detections):
            class_detections = [det for det in detections if det.class_name == class_name]
            
            if len(class_detections) == 0:
                continue
            
            # 转换检测格式为ByteTracker需要的numpy数组
            # 格式: [x1, y1, x2, y2, score]
            det_array = []
            for det in class_detections:
                xyxy = det.xyxy
                det_array.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], det.confidence])
            
            det_array = np.array(det_array, dtype=np.float32)
            
            # ByteTracker更新 (每个类别独立跟踪)
            online_targets = self.tracker.update(det_array, None, None)
            
            # 处理跟踪结果
            for target in online_targets:
                track_id = target.track_id
                
                # 更新类别映射
                self.track_class_map[track_id] = class_name
                
                # 获取类别内的序号
                if track_id not in self.all_tracks:
                    class_track_id = self._get_next_class_id(class_name)
                    self.all_tracks[track_id] = {
                        'track_id': track_id,
                        'class_name': class_name,
                        'class_track_id': class_track_id,
                        'formatted_id': f'{class_name}_{class_track_id}',
                        'points': [],
                        'bbox': target.tlbr.copy(),
                        'confidence': target.score
                    }
                else:
                    # 更新现有轨迹
                    self.all_tracks[track_id]['bbox'] = target.tlbr.copy()
                    self.all_tracks[track_id]['confidence'] = target.score
                
                # 保存轨迹点
                track_point = {
                    "frame_id": frame_id,
                    "bbox": target.tlbr.tolist(),  # [x1, y1, x2, y2]
                    "timestamp": timestamp
                }
                self.all_tracks[track_id]['points'].append(track_point)
                
                # 创建Track对象以保持兼容性
                track_obj = TrackObject(target, self.all_tracks[track_id])
                all_online_tracks.append(track_obj)
        
        return all_online_tracks
    
    def _get_next_class_id(self, class_name):
        """获取类别内的下一个序号"""
        if class_name not in self.class_counters:
            self.class_counters[class_name] = 0
        self.class_counters[class_name] += 1
        return self.class_counters[class_name]
    
    def get_all_track_points(self) -> List[Dict]:
        """获取所有轨迹点 - 按要求格式"""
        all_points = []
        
        for track_info in self.all_tracks.values():
            if len(track_info['points']) > 1:  # 至少2个点才算有效轨迹
                for point in track_info['points']:
                    point_copy = point.copy()
                    point_copy['track_id'] = track_info['formatted_id']
                    point_copy['class_name'] = track_info['class_name']
                    all_points.append(point_copy)
        
        # 按时间戳排序
        all_points.sort(key=lambda x: x['timestamp'])
        return all_points
    
    def get_vehicle_sequences(self) -> List[List[Dict]]:
        """获取每辆车的轨迹序列"""
        sequences = []
        
        for track_info in self.all_tracks.values():
            if len(track_info['points']) > 1:
                sequence = []
                for point in track_info['points']:
                    point_copy = point.copy()
                    point_copy['track_id'] = track_info['formatted_id']
                    point_copy['class_name'] = track_info['class_name']
                    sequence.append(point_copy)
                
                # 按时间排序
                sequence.sort(key=lambda x: x['timestamp'])
                sequences.append(sequence)
        
        return sequences
    
    def get_track_points_by_class(self) -> Dict[str, List[Dict]]:
        """按类别分组获取轨迹点"""
        track_points_by_class = {}
        
        for track_info in self.all_tracks.values():
            if len(track_info['points']) > 1:
                class_name = track_info['class_name']
                if class_name not in track_points_by_class:
                    track_points_by_class[class_name] = []
                
                track_sequence = {
                    "track_id": track_info['formatted_id'],
                    "class_name": class_name,
                    "points": track_info['points']
                }
                track_points_by_class[class_name].append(track_sequence)
        
        return track_points_by_class
    
    def get_statistics(self):
        """获取跟踪统计信息"""
        valid_tracks = [t for t in self.all_tracks.values() if len(t['points']) > 1]
        
        stats = {
            "total_tracks": len(self.all_tracks),
            "confirmed_tracks": len(valid_tracks),
            "class_counts": self.class_counters.copy(),
            "active_tracks_by_class": {}
        }
        
        # 统计每个类别的活跃轨迹数
        for track_info in valid_tracks:
            class_name = track_info['class_name']
            if class_name not in stats["active_tracks_by_class"]:
                stats["active_tracks_by_class"][class_name] = 0
            stats["active_tracks_by_class"][class_name] += 1
        
        return stats

class TrackObject:
    """轨迹对象 - 保持与原接口兼容"""
    
    def __init__(self, strack, track_info):
        self.track_id = strack.track_id
        self.bbox = strack.tlbr  # [x1, y1, x2, y2]
        self.confidence = strack.score
        self.class_name = track_info['class_name']
        self.class_track_id = track_info['class_track_id']
        self.formatted_id = track_info['formatted_id']
        self.track_points = track_info['points']
        
        # 兼容性属性
        self.hits = len(track_info['points'])
        self.age = len(track_info['points'])
        self.time_since_update = 0
        self.confirmed = len(track_info['points']) > 1
    
    @property
    def is_confirmed(self):
        return self.confirmed
    
    @property
    def tlbr(self):
        return self.bbox
    
    @property
    def tlwh(self):
        """转换为tlwh格式"""
        return np.array([self.bbox[0], self.bbox[1], 
                        self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]])

# 为了保持兼容性，创建别名
DeepSORT = ByteTrackerWrapper

if __name__ == "__main__":
    # 测试ByteTracker
    if YOLOX_AVAILABLE:
        print("ByteTracker测试...")
        tracker = ByteTrackerWrapper()
        
        # 创建模拟检测
        detections = [
            Detection([100, 100, 200, 150], 0.9, 'car'),
            Detection([300, 200, 400, 300], 0.8, 'truck')
        ]
        
        # 更新跟踪器
        tracks = tracker.update(detections, 0, 0.0)
        print(f"跟踪到 {len(tracks)} 个目标")
        
        for track in tracks:
            print(f"  {track.formatted_id}: bbox={track.bbox}, conf={track.confidence:.2f}")
    else:
        print("无法测试ByteTracker，请安装YOLOX")