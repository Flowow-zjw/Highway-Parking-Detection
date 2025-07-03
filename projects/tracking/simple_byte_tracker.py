import numpy as np
import cv2
import math
from typing import List, Dict, Any, Optional
from collections import OrderedDict
import itertools

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using simple assignment")

def linear_assignment(cost_matrix, thresh=0.7):
    """匈牙利算法进行线性分配"""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    if SCIPY_AVAILABLE:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = np.array(list(zip(row_ind, col_ind)))
    else:
        # 简单的贪心分配作为备选
        matches = []
        cost_matrix_copy = cost_matrix.copy()
        for _ in range(min(cost_matrix.shape)):
            min_idx = np.unravel_index(np.argmin(cost_matrix_copy), cost_matrix_copy.shape)
            matches.append(min_idx)
            cost_matrix_copy[min_idx[0], :] = 1e6
            cost_matrix_copy[:, min_idx[1]] = 1e6
        matches = np.array(matches)
    
    # 过滤高成本匹配
    if len(matches) > 0:
        valid_matches = matches[cost_matrix[matches[:, 0], matches[:, 1]] <= thresh]
        unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in valid_matches[:, 0]]
        unmatched_detections = [j for j in range(cost_matrix.shape[1]) if j not in valid_matches[:, 1]]
    else:
        valid_matches = np.empty((0, 2), dtype=int)
        unmatched_tracks = list(range(cost_matrix.shape[0]))
        unmatched_detections = list(range(cost_matrix.shape[1]))
    
    return valid_matches, unmatched_tracks, unmatched_detections

def bbox_iou(box1, box2):
    """计算IoU"""
    x1, y1, x2, y2 = box1[:4]
    x1_b, y1_b, x2_b, y2_b = box2[:4]

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

def rotated_bbox_to_xyxy(bbox):
    """将旋转框转换为普通框"""
    if len(bbox) < 5:
        return bbox[:4]
    
    center = (bbox[0], bbox[1])
    size = (bbox[2], bbox[3])
    angle = bbox[4] * 180 / math.pi
    
    rect = (center, size, angle)
    box_points = cv2.boxPoints(rect)
    
    x_coords = box_points[:, 0]
    y_coords = box_points[:, 1]
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    
    return [x1, y1, x2, y2]

def tlbr_to_tlwh(tlbr):
    """转换tlbr到tlwh格式"""
    return [tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]]

class Detection:
    """检测结果类"""
    def __init__(self, bbox, confidence, class_name):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.confidence = confidence
        self.class_name = class_name
        self.xyxy = rotated_bbox_to_xyxy(self.bbox)
    
    @property
    def tlwh(self):
        xyxy = self.xyxy
        return np.array([xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]])
    
    @property
    def tlbr(self):
        return np.array(self.xyxy)

class TrackState:
    """轨迹状态"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class STrack:
    """简化的轨迹类 - 模仿ByteTracker的STrack"""
    
    shared_track_id = 0
    
    def __init__(self, tlwh, score, class_name='vehicle'):
        # 基本属性
        self.track_id = None
        self.class_name = class_name
        self.score = score
        self.state = TrackState.New
        
        # 位置信息
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        
        # 跟踪历史
        self.tracklet_len = 0
        self.time_since_update = 0
        self.start_frame = 0
        self.frame_id = 0
        
    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """静态方法：转换tlbr到tlwh格式"""
        return [tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]]
        
    def activate(self, kalman_filter, frame_id):
        """激活轨迹"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        
        # 初始化卡尔曼滤波器
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah())
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.start_frame = frame_id
        
    def predict(self):
        """预测下一状态"""
        if self.kalman_filter is not None:
            self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
            
    def update(self, new_track, frame_id):
        """更新轨迹"""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self.score = new_track.score
        
        # 卡尔曼滤波更新
        if self.kalman_filter is not None:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        
        self.state = TrackState.Tracked
        self.time_since_update = 0
        
    def mark_lost(self):
        """标记为丢失"""
        self.state = TrackState.Lost
        
    def mark_removed(self):
        """标记为删除"""
        self.state = TrackState.Removed
        
    @staticmethod
    def next_id():
        """获取下一个ID"""
        STrack.shared_track_id += 1
        return STrack.shared_track_id
        
    @staticmethod
    def reset_id():
        """重置ID计数器"""
        STrack.shared_track_id = 0
        
    def tlwh_to_xyah(self, tlwh=None):
        """转换tlwh到xyah格式"""
        if tlwh is None:
            tlwh = self._tlwh
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2  # 转换为中心点
        ret[2] /= ret[3]        # 宽高比
        return ret
        
    @property
    def tlwh(self):
        """获取tlwh"""
        if self.mean is None:
            return self._tlwh.copy()
        
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # 从宽高比恢复高度
        ret[:2] -= ret[2:] / 2  # 从中心点恢复左上角
        return ret
        
    @property 
    def tlbr(self):
        """获取tlbr"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @property
    def is_activated(self):
        """检查是否已激活"""
        return self.state == TrackState.Tracked

class SimpleKalmanFilter:
    """简化的卡尔曼滤波器"""
    
    def __init__(self):
        ndim, dt = 4, 1.
        
        # 状态转移矩阵
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
            
        # 观测矩阵
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # 运动和观测不确定性的权重
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        
    def initiate(self, measurement):
        """初始化状态"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
        
    def predict(self, mean, covariance):
        """预测状态"""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
            
        return mean, covariance
        
    def update(self, mean, covariance, measurement):
        """更新状态"""
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        
        chol_factor = np.linalg.cholesky(projected_cov + np.eye(4) * 1e-6)
        kalman_gain = np.linalg.multi_dot((
            covariance, self._update_mat.T, np.linalg.inv(chol_factor).T, 
            np.linalg.inv(chol_factor)))
            
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, self._update_mat, covariance))
            
        return new_mean, new_covariance

class SimpleBYTETracker:
    """简化版ByteTracker实现"""
    
    def __init__(self, frame_rate=30, track_thresh=0.6, track_buffer=30, 
                 match_thresh=0.8, high_thresh=0.6, low_thresh=0.1, min_box_area=10):
        
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.min_box_area = min_box_area
        
        self.frame_id = 0
        self.tracked_stracks = []  # 已跟踪轨迹
        self.lost_stracks = []     # 丢失轨迹
        self.removed_stracks = []  # 删除轨迹
        
        self.kalman_filter = SimpleKalmanFilter()
        
        # 重置ID计数器
        STrack.reset_id()
        
    def update(self, output_results, img_info=None, img_size=None):
        """更新跟踪器"""
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # 检查输入格式
        if len(output_results) == 0:
            # 没有检测结果
            scores = np.array([])
            bboxes = np.empty((0, 4))
        else:
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
            else:
                # 兼容其他格式
                scores = output_results[:, -1]
                bboxes = output_results[:, :-1]
        
        # 过滤小框和低置信度检测
        if len(scores) > 0:
            remain_inds = np.logical_and(
                (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) > self.min_box_area,
                scores > 0.1
            )
            
            if remain_inds.sum() > 0:
                inds_low = scores > self.low_thresh
                inds_high = scores > self.high_thresh
                inds_second = np.logical_and(inds_low, np.logical_not(inds_high))
                
                dets_second = bboxes[inds_second]
                dets = bboxes[inds_high]
                scores_keep = scores[inds_high]
                scores_second = scores[inds_second]
            else:
                dets = np.empty((0, 4))
                dets_second = np.empty((0, 4))
                scores_keep = np.empty((0,))
                scores_second = np.empty((0,))
        else:
            dets = np.empty((0, 4))
            dets_second = np.empty((0, 4))
            scores_keep = np.empty((0,))
            scores_second = np.empty((0,))
        
        # 将检测转换为轨迹
        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), score, 'vehicle') 
                         for (tlbr, score) in zip(dets, scores_keep)]
        else:
            detections = []
        
        # 预测现有轨迹
        for strack in itertools.chain(self.tracked_stracks, self.lost_stracks):
            strack.predict()
        
        # 第一次关联：高置信度检测与跟踪轨迹
        strack_pool = self.tracked_stracks
        dists = self.iou_distance(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            track.update(det, self.frame_id)
            activated_starcks.append(track)
        
        # 第二次关联：剩余轨迹与低置信度检测
        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), score, 'vehicle')
                               for (tlbr, score) in zip(dets_second, scores_second)]
        else:
            detections_second = []
            
        r_tracked_stracks = [strack_pool[i] for i in u_track 
                           if i < len(strack_pool) and strack_pool[i].state == TrackState.Tracked]
        
        dists = self.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            track.update(det, self.frame_id)
            activated_starcks.append(track)
        
        # 处理未匹配轨迹
        for it in u_track:
            if it < len(r_tracked_stracks):
                track = r_tracked_stracks[it]
                track.mark_lost()
                lost_stracks.append(track)
        
        # 第三次关联：剩余检测与丢失轨迹
        detections = [detections[i] for i in u_detection if i < len(detections)]
        dists = self.iou_distance(self.lost_stracks, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        
        for itracked, idet in matches:
            track = self.lost_stracks[itracked]
            det = detections[idet]
            track.update(det, self.frame_id)
            activated_starcks.append(track)
            refind_stracks.append(track)
        
        # 创建新轨迹
        for inew in u_detection:
            if inew < len(detections):
                track = detections[inew]
                if track.score < self.track_thresh:
                    continue
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)
        
        # 更新状态
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)
        
        # 更新轨迹列表
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        
        return [track for track in self.tracked_stracks if track.is_activated]
    
    def iou_distance(self, atracks, btracks):
        """计算IoU距离矩阵"""
        if len(atracks) == 0 or len(btracks) == 0:
            return np.empty((len(atracks), len(btracks)))
        
        cost_matrix = np.zeros((len(atracks), len(btracks)))
        for i, atrack in enumerate(atracks):
            for j, btrack in enumerate(btracks):
                cost_matrix[i, j] = 1 - bbox_iou(atrack.tlbr, btrack.tlbr)
        
        return cost_matrix

def joint_stracks(tlista, tlistb):
    """合并轨迹列表"""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    """从列表a中移除列表b中的轨迹"""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

class ByteTrackerWrapper:
    """ByteTracker包装器 - 不依赖YOLOX"""
    
    def __init__(self, frame_rate=30, track_thresh=0.6, track_buffer=30, 
                 match_thresh=0.8, min_box_area=10, mot20=False):
        
        self.tracker = SimpleBYTETracker(
            frame_rate=frame_rate,
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            min_box_area=min_box_area
        )
        
        self.all_tracks = {}
        self.class_counters = {}
        self.track_class_map = {}
        
        print("SimpleBYTETracker初始化完成")
    
    def update(self, detections, frame_id, timestamp, image=None):
        """更新跟踪器"""
        # 转换检测格式
        if isinstance(detections, list) and len(detections) > 0:
            det_array = []
            det_classes = []
            
            for det in detections:
                xyxy = det.xyxy if hasattr(det, 'xyxy') else rotated_bbox_to_xyxy(det.bbox)
                det_array.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], det.confidence])
                det_classes.append(det.class_name)
            
            det_array = np.array(det_array)
        else:
            det_array = np.empty((0, 5))
            det_classes = []
        
        # 更新跟踪器
        online_targets = self.tracker.update(det_array)
        
        # 处理结果
        tracks = []
        for i, target in enumerate(online_targets):
            track_id = target.track_id
            
            # 分配类别 - 改进的类别匹配逻辑
            if det_classes:
                # 根据位置匹配最近的检测类别
                target_center = [(target.tlbr[0] + target.tlbr[2])/2, (target.tlbr[1] + target.tlbr[3])/2]
                min_dist = float('inf')
                best_class = det_classes[0]
                
                for j, det in enumerate(detections):
                    det_center = [(det.xyxy[0] + det.xyxy[2])/2, (det.xyxy[1] + det.xyxy[3])/2]
                    dist = ((target_center[0] - det_center[0])**2 + (target_center[1] - det_center[1])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_class = det.class_name
                
                class_name = best_class
            else:
                class_name = 'vehicle'
            
            if track_id not in self.all_tracks:
                class_track_id = self._get_next_class_id(class_name)
                self.all_tracks[track_id] = {
                    'track_id': track_id,
                    'class_name': class_name,
                    'class_track_id': class_track_id,
                    'formatted_id': f'{class_name}_{class_track_id}',
                    'points': []
                }
            
            # 保存轨迹点
            track_point = {
                "frame_id": frame_id,
                "bbox": target.tlbr.tolist(),
                "timestamp": timestamp
            }
            self.all_tracks[track_id]['points'].append(track_point)
            
            # 创建轨迹对象
            track_obj = TrackObject(target, self.all_tracks[track_id])
            tracks.append(track_obj)
        
        return tracks
    
    def _get_next_class_id(self, class_name):
        if class_name not in self.class_counters:
            self.class_counters[class_name] = 0
        self.class_counters[class_name] += 1
        return self.class_counters[class_name]
    
    def get_all_track_points(self):
        """获取所有轨迹点"""
        all_points = []
        for track_info in self.all_tracks.values():
            if len(track_info['points']) > 1:
                for point in track_info['points']:
                    point_copy = point.copy()
                    point_copy['track_id'] = track_info['formatted_id']
                    point_copy['class_name'] = track_info['class_name']
                    all_points.append(point_copy)
        
        all_points.sort(key=lambda x: x['timestamp'])
        return all_points
    
    def get_vehicle_sequences(self):
        """获取车辆序列"""
        sequences = []
        for track_info in self.all_tracks.values():
            if len(track_info['points']) > 1:
                sequence = []
                for point in track_info['points']:
                    point_copy = point.copy()
                    point_copy['track_id'] = track_info['formatted_id']
                    point_copy['class_name'] = track_info['class_name']
                    sequence.append(point_copy)
                
                sequence.sort(key=lambda x: x['timestamp'])
                sequences.append(sequence)
        
        return sequences
    
    def get_track_points_by_class(self):
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
        """获取统计信息"""
        valid_tracks = [t for t in self.all_tracks.values() if len(t['points']) > 1]
        
        stats = {
            "total_tracks": len(self.all_tracks),
            "confirmed_tracks": len(valid_tracks),
            "class_counts": self.class_counters.copy(),
            "active_tracks_by_class": {}
        }
        
        for track_info in valid_tracks:
            class_name = track_info['class_name']
            if class_name not in stats["active_tracks_by_class"]:
                stats["active_tracks_by_class"][class_name] = 0
            stats["active_tracks_by_class"][class_name] += 1
        
        return stats

class TrackObject:
    """轨迹对象"""
    def __init__(self, strack, track_info):
        self.track_id = strack.track_id
        self.bbox = strack.tlbr
        self.confidence = strack.score
        self.class_name = track_info['class_name']
        self.class_track_id = track_info['class_track_id']
        self.formatted_id = track_info['formatted_id']
        self.track_points = track_info['points']
        self.confirmed = True
    
    @property
    def is_confirmed(self):
        return self.confirmed
    
    @property
    def tlbr(self):
        return self.bbox
    
    @property
    def tlwh(self):
        return np.array([self.bbox[0], self.bbox[1], 
                        self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]])

# 别名
DeepSORT = ByteTrackerWrapper

if __name__ == "__main__":
    print("SimpleBYTETracker测试...")
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