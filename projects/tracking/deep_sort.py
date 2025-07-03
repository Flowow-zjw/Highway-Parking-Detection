import numpy as np
import cv2
from collections import deque
from typing import List, Dict, Any, Optional
import math
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using simple assignment")

def linear_assignment(cost_matrix):
    """匈牙利算法进行线性分配"""
    if SCIPY_AVAILABLE:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
    else:
        # 简单的贪心分配作为备选
        matches = []
        cost_matrix = cost_matrix.copy()
        for _ in range(min(cost_matrix.shape)):
            min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            matches.append(min_idx)
            cost_matrix[min_idx[0], :] = 1e6
            cost_matrix[:, min_idx[1]] = 1e6
        return np.array(matches)

def rotated_bbox_iou(bbox1, bbox2):
    """计算旋转边界框的IoU"""
    try:
        # 如果是旋转框格式 [cx, cy, w, h, angle]
        if len(bbox1) >= 5 and len(bbox2) >= 5:
            # 转换为RotatedRect格式
            rect1 = ((bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), bbox1[4] * 180 / math.pi)
            rect2 = ((bbox2[0], bbox2[1]), (bbox2[2], bbox2[3]), bbox2[4] * 180 / math.pi)
            
            # 计算旋转框IoU
            return rotated_rect_iou(rect1, rect2)
        else:
            # 回退到普通IoU
            return bbox_iou(bbox1, bbox2)
    except:
        return bbox_iou(bbox1, bbox2)

def rotated_rect_iou(rect1, rect2):
    """计算两个旋转矩形的IoU"""
    try:
        # 计算中心距离
        center_dist = np.sqrt((rect1[0][0] - rect2[0][0])**2 + (rect1[0][1] - rect2[0][1])**2)
        
        # 尺寸相似度
        size_sim = min(rect1[1][0]/max(rect2[1][0], 1e-6), rect2[1][0]/max(rect1[1][0], 1e-6)) * \
                   min(rect1[1][1]/max(rect2[1][1], 1e-6), rect2[1][1]/max(rect1[1][1], 1e-6))
        
        # 角度相似度
        angle_diff = abs(rect1[2] - rect2[2])
        angle_diff = min(angle_diff, 180 - angle_diff)
        angle_sim = 1 - angle_diff / 90.0
        
        # 距离阈值
        max_dist = (max(rect1[1][0], rect1[1][1]) + max(rect2[1][0], rect2[1][1])) / 2
        
        if center_dist < max_dist:
            # 综合相似度作为IoU近似
            dist_factor = 1 - (center_dist / max_dist)
            similarity = size_sim * angle_sim * dist_factor
            return max(0, min(1, similarity))
        else:
            return 0.0
    except:
        return 0.0

def bbox_iou(box1, box2):
    """计算普通边界框IoU"""
    if len(box1) >= 4 and len(box2) >= 4:
        x1, y1, x2, y2 = box1[:4]
        x1_b, y1_b, x2_b, y2_b = box2[:4]
    else:
        return 0.0

    xx1 = max(x1, x1_b)
    yy1 = max(y1, y1_b)
    xx2 = min(x2, x2_b)
    yy2 = min(y2, y2_b)
    
    if xx2 <= xx1 or yy2 <= yy1:
        return 0.0
        
    inter = (xx2 - xx1) * (yy2 - yy1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_b - x1_b) * (y2_b - y1_b)
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0

class SimpleFeatureExtractor:
    """改进的特征提取器"""
    def __init__(self):
        pass
    
    def extract_features(self, image, bbox):
        """从图像的边界框区域提取特征"""
        try:
            if len(bbox) >= 5:  # 旋转框 [cx, cy, w, h, angle]
                # 转换旋转框为四个角点
                center = (bbox[0], bbox[1])
                size = (bbox[2], bbox[3])
                angle = bbox[4] * 180 / math.pi
                rect = (center, size, angle)
                box_points = cv2.boxPoints(rect)
                
                # 获取包围盒
                x_coords = box_points[:, 0]
                y_coords = box_points[:, 1]
                x1, y1 = int(max(0, min(x_coords))), int(max(0, min(y_coords)))
                x2, y2 = int(min(image.shape[1], max(x_coords))), int(min(image.shape[0], max(y_coords)))
            else:  # 普通框 [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, bbox[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(128)
                
            # 提取ROI区域
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                return np.zeros(128)
                
            # 调整大小
            roi = cv2.resize(roi, (64, 128))
            
            # 多种特征组合
            features_list = []
            
            # 1. 颜色直方图特征
            hist_b = cv2.calcHist([roi], [0], None, [16], [0, 256])
            hist_g = cv2.calcHist([roi], [1], None, [16], [0, 256])
            hist_r = cv2.calcHist([roi], [2], None, [16], [0, 256])
            hist_features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
            
            # 2. LBP纹理特征 (简化版)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            lbp_hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            
            # 合并所有特征
            all_features = np.concatenate([hist_features, lbp_hist.flatten()])
            
            # 归一化
            all_features = all_features / (np.linalg.norm(all_features) + 1e-6)
            
            # 调整到128维
            if len(all_features) > 128:
                features = all_features[:128]
            else:
                features = np.zeros(128)
                features[:len(all_features)] = all_features
                
            return features
        except Exception as e:
            print(f"特征提取失败: {e}")
            return np.zeros(128)

def cosine_distance(features1, features2):
    """计算余弦距离"""
    try:
        dot_product = np.dot(features1, features2)
        norm_a = np.linalg.norm(features1)
        norm_b = np.linalg.norm(features2)
        
        if norm_a == 0 or norm_b == 0:
            return 1.0
        
        cosine_sim = dot_product / (norm_a * norm_b)
        return 1 - max(0, min(1, cosine_sim))
    except:
        return 1.0

class TrackState:
    """轨迹状态枚举"""
    New = 0
    Tracked = 1
    Lost = 2
    Deleted = 3

class Detection:
    """检测结果类 - 支持旋转框"""
    def __init__(self, bbox, confidence, class_name, feature=None):
        self.bbox = np.array(bbox, dtype=np.float32)  # 保持原始格式
        self.confidence = confidence
        self.class_name = class_name
        self.feature = feature
        
    @property
    def tlwh(self):
        """转换为tlwh格式"""
        if len(self.bbox) >= 5:  # 旋转框
            # 对于旋转框，返回包围盒的tlwh
            center = (self.bbox[0], self.bbox[1])
            size = (self.bbox[2], self.bbox[3])
            angle = self.bbox[4] * 180 / math.pi
            rect = (center, size, angle)
            box_points = cv2.boxPoints(rect)
            
            x_coords = box_points[:, 0]
            y_coords = box_points[:, 1]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            return np.array([x1, y1, x2-x1, y2-y1])
        else:  # 普通框
            ret = self.bbox.copy()
            ret[2:] -= ret[:2]
            return ret
        
    @property
    def tlbr(self):
        """tlbr格式"""
        if len(self.bbox) >= 5:  # 旋转框
            # 返回包围盒
            center = (self.bbox[0], self.bbox[1])
            size = (self.bbox[2], self.bbox[3])
            angle = self.bbox[4] * 180 / math.pi
            rect = (center, size, angle)
            box_points = cv2.boxPoints(rect)
            
            x_coords = box_points[:, 0]
            y_coords = box_points[:, 1]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            return np.array([x1, y1, x2, y2])
        else:  # 普通框
            return self.bbox

class Track:
    """改进的轨迹类"""
    def __init__(self, detection, track_id, class_name, n_init=3, max_age=50):
        self.track_id = track_id
        self.class_name = class_name
        self.class_track_id = None
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.New
        self.n_init = n_init
        self.max_age = max_age
        
        # 轨迹历史 - 按要求的格式
        self.track_points = []
        self.features = deque(maxlen=50)  # 减少特征数量提高性能
        
        # 位置历史用于预测
        self.position_history = deque(maxlen=10)
        
        # 初始化位置
        self.bbox = detection.bbox.copy()
        self.confidence = detection.confidence
        
        # 添加特征
        if detection.feature is not None:
            self.features.append(detection.feature)
    
    def set_class_track_id(self, class_track_id):
        """设置类别内的序号"""
        self.class_track_id = class_track_id
    
    @property
    def formatted_id(self):
        """获取格式化的ID，如 car_1, truck_2"""
        if self.class_track_id is not None:
            return f"{self.class_name}_{self.class_track_id}"
        else:
            return f"{self.class_name}_{self.track_id}"
            
    def predict(self):
        """改进的位置预测"""
        self.age += 1
        
        # 如果有足够的历史位置，进行线性预测
        if len(self.position_history) >= 2:
            last_pos = self.position_history[-1]
            prev_pos = self.position_history[-2]
            
            if len(self.bbox) >= 5:  # 旋转框
                # 预测中心点
                dx = last_pos[0] - prev_pos[0]
                dy = last_pos[1] - prev_pos[1]
                self.bbox[0] += dx * 0.5  # 减少预测强度
                self.bbox[1] += dy * 0.5
            else:  # 普通框
                # 预测中心点
                last_center = [(last_pos[0] + last_pos[2])/2, (last_pos[1] + last_pos[3])/2]
                prev_center = [(prev_pos[0] + prev_pos[2])/2, (prev_pos[1] + prev_pos[3])/2]
                
                dx = last_center[0] - prev_center[0]
                dy = last_center[1] - prev_center[1]
                
                self.bbox[0] += dx * 0.5
                self.bbox[1] += dy * 0.5
                self.bbox[2] += dx * 0.5
                self.bbox[3] += dy * 0.5
        
    def update(self, detection, frame_id, timestamp):
        """更新轨迹"""
        self.time_since_update = 0
        self.hits += 1
        
        # 平滑更新位置 (减少抖动)
        alpha = 0.8  # 平滑系数
        self.bbox = alpha * detection.bbox + (1 - alpha) * self.bbox
        self.confidence = detection.confidence
        
        # 更新位置历史
        self.position_history.append(self.bbox.copy())
        
        # 更新特征库
        if detection.feature is not None:
            self.features.append(detection.feature)
        
        # 更新状态 - 降低确认阈值提高精度
        if self.state == TrackState.New and self.hits >= self.n_init:
            self.state = TrackState.Tracked
            
        # 添加轨迹点 - 按要求的格式
        track_point = {
            "frame_id": frame_id,
            "bbox": self.bbox.tolist(),  # 保持原始格式(旋转框或普通框)
            "timestamp": timestamp
        }
        self.track_points.append(track_point)
        
    def mark_missed(self):
        """标记为错过"""
        self.time_since_update += 1
        if self.state == TrackState.New:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Deleted
        else:
            self.state = TrackState.Lost
            
    @property
    def tlwh(self):
        """获取tlwh格式边界框"""
        if len(self.bbox) >= 5:  # 旋转框
            center = (self.bbox[0], self.bbox[1])
            size = (self.bbox[2], self.bbox[3])
            angle = self.bbox[4] * 180 / math.pi
            rect = (center, size, angle)
            box_points = cv2.boxPoints(rect)
            
            x_coords = box_points[:, 0]
            y_coords = box_points[:, 1]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            return np.array([x1, y1, x2-x1, y2-y1])
        else:  # 普通框
            ret = self.bbox.copy()
            ret[2:] -= ret[:2]
            return ret
        
    @property
    def tlbr(self):
        """获取tlbr格式边界框"""
        if len(self.bbox) >= 5:  # 旋转框
            center = (self.bbox[0], self.bbox[1])
            size = (self.bbox[2], self.bbox[3])
            angle = self.bbox[4] * 180 / math.pi
            rect = (center, size, angle)
            box_points = cv2.boxPoints(rect)
            
            x_coords = box_points[:, 0]
            y_coords = box_points[:, 1]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            return np.array([x1, y1, x2, y2])
        else:  # 普通框
            return self.bbox
        
    def get_mean_feature(self):
        """获取平均特征"""
        if not self.features:
            return None
        return np.mean(list(self.features), axis=0)
        
    @property
    def is_confirmed(self):
        """轨迹是否已确认"""
        return self.state == TrackState.Tracked

class DeepSORT:
    """改进的DeepSORT跟踪器"""
    def __init__(self, max_age=50, n_init=3, max_iou_distance=0.5, max_cosine_distance=0.3):
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance  # 降低IoU阈值提高精度
        self.max_cosine_distance = max_cosine_distance
        
        self.tracks = []
        self.next_id = 1
        self.feature_extractor = SimpleFeatureExtractor()
        
        # 每个类别的计数器
        self.class_counters = {}
        
    def _get_next_class_id(self, class_name):
        """获取类别内的下一个序号"""
        if class_name not in self.class_counters:
            self.class_counters[class_name] = 0
        self.class_counters[class_name] += 1
        return self.class_counters[class_name]
        
    def update(self, detections, frame_id, timestamp, image=None):
        """更新跟踪器"""
        # 提取特征
        for det in detections:
            if image is not None:
                det.feature = self.feature_extractor.extract_features(image, det.bbox)
        
        # 预测所有轨迹
        for track in self.tracks:
            track.predict()
            
        # 匹配
        matched, unmatched_tracks, unmatched_detections = self._match(detections)
        
        # 更新匹配的轨迹
        for track_idx, detection_idx in matched:
            self.tracks[track_idx].update(detections[detection_idx], frame_id, timestamp)
            
        # 标记未匹配的轨迹
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            
        # 创建新轨迹
        for detection_idx in unmatched_detections:
            self._create_track(detections[detection_idx], frame_id, timestamp)
            
        # 删除旧轨迹
        self.tracks = [t for t in self.tracks if t.state != TrackState.Deleted]
        
        # 返回确认的轨迹
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed]
        return confirmed_tracks
        
    def _match(self, detections):
        """匹配检测和轨迹"""
        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))
            
        # 计算成本矩阵
        cost_matrix = self._calculate_cost_matrix(detections)
        
        # 使用匈牙利算法匹配
        if cost_matrix.size == 0:
            return [], list(range(len(self.tracks))), list(range(len(detections)))
            
        matches = linear_assignment(cost_matrix)
        
        # 过滤匹配结果
        valid_matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        
        for track_idx, det_idx in matches:
            if track_idx < len(self.tracks) and det_idx < len(detections):
                if cost_matrix[track_idx, det_idx] < self.max_iou_distance:
                    valid_matches.append([track_idx, det_idx])
                    if track_idx in unmatched_tracks:
                        unmatched_tracks.remove(track_idx)
                    if det_idx in unmatched_detections:
                        unmatched_detections.remove(det_idx)
                        
        return valid_matches, unmatched_tracks, unmatched_detections
        
    def _calculate_cost_matrix(self, detections):
        """计算成本矩阵 - 支持旋转框"""
        if len(self.tracks) == 0 or len(detections) == 0:
            return np.empty((len(self.tracks), len(detections)))
            
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                # 只匹配相同类别的检测和轨迹
                if track.class_name != detection.class_name:
                    cost_matrix[i, j] = 1.0
                    continue
                    
                # IoU距离 - 支持旋转框
                if len(track.bbox) >= 5 and len(detection.bbox) >= 5:
                    iou = rotated_bbox_iou(track.bbox, detection.bbox)
                else:
                    iou = bbox_iou(track.tlbr, detection.tlbr)
                
                iou_distance = 1 - iou
                
                # 特征距离
                feature_distance = 0.5
                if len(track.features) > 0 and detection.feature is not None:
                    mean_feature = track.get_mean_feature()
                    if mean_feature is not None:
                        feature_distance = cosine_distance(mean_feature, detection.feature)
                
                # 综合距离 - 更重视IoU
                cost_matrix[i, j] = 0.8 * iou_distance + 0.2 * feature_distance
                
        return cost_matrix
        
    def _create_track(self, detection, frame_id, timestamp):
        """创建新轨迹"""
        # 获取类别内的序号
        class_track_id = self._get_next_class_id(detection.class_name)
        
        # 创建轨迹
        track = Track(detection, self.next_id, detection.class_name, self.n_init, self.max_age)
        track.set_class_track_id(class_track_id)
        track.update(detection, frame_id, timestamp)
        
        self.tracks.append(track)
        self.next_id += 1
        
        print(f"创建新轨迹: {track.formatted_id}")
        
    def get_all_track_points(self) -> List[Dict]:
        """获取所有轨迹点 - 按要求格式"""
        all_track_points = []
        
        for track in self.tracks:
            if len(track.track_points) > 1 and track.is_confirmed:
                # 为每个轨迹添加标识符
                vehicle_track_points = []
                for point in track.track_points:
                    vehicle_point = point.copy()
                    vehicle_point['track_id'] = track.formatted_id
                    vehicle_point['class_name'] = track.class_name
                    vehicle_track_points.append(vehicle_point)
                
                all_track_points.extend(vehicle_track_points)
                
        # 按时间戳排序
        all_track_points.sort(key=lambda x: x['timestamp'])
        return all_track_points
    
    def get_vehicle_sequences(self) -> List[List[Dict]]:
        """获取每辆车的轨迹序列 - 按要求格式"""
        vehicle_sequences = []
        
        for track in self.tracks:
            if len(track.track_points) > 1 and track.is_confirmed:
                # 每辆车的轨迹点序列
                sequence = []
                for point in track.track_points:
                    vehicle_point = point.copy()
                    vehicle_point['track_id'] = track.formatted_id
                    vehicle_point['class_name'] = track.class_name
                    sequence.append(vehicle_point)
                
                # 按时间排序
                sequence.sort(key=lambda x: x['timestamp'])
                vehicle_sequences.append(sequence)
        
        return vehicle_sequences
    
    def get_track_points_by_class(self) -> Dict[str, List[Dict]]:
        """按类别分组获取轨迹点"""
        track_points_by_class = {}
        
        for track in self.tracks:
            if len(track.track_points) > 1 and track.is_confirmed:
                class_name = track.class_name
                if class_name not in track_points_by_class:
                    track_points_by_class[class_name] = []
                
                # 为每个轨迹创建一个独立的轨迹序列
                track_sequence = {
                    "track_id": track.formatted_id,
                    "class_name": class_name,
                    "points": track.track_points
                }
                track_points_by_class[class_name].append(track_sequence)
                
        return track_points_by_class
    
    def get_statistics(self):
        """获取跟踪统计信息"""
        stats = {
            "total_tracks": len(self.tracks),
            "confirmed_tracks": len([t for t in self.tracks if t.is_confirmed]),
            "class_counts": self.class_counters.copy(),
            "active_tracks_by_class": {}
        }
        
        # 统计每个类别的活跃轨迹数
        for track in self.tracks:
            if track.is_confirmed:
                class_name = track.class_name
                if class_name not in stats["active_tracks_by_class"]:
                    stats["active_tracks_by_class"][class_name] = 0
                stats["active_tracks_by_class"][class_name] += 1
                
        return stats