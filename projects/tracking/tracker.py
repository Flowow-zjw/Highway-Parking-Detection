import os
import sys
import numpy as np
import time
import json
import cv2
import mmcv
import math
from mmdet.apis import init_detector, inference_detector
from typing import List, Dict, Any

# 导入简化版ByteTracker (不依赖YOLOX)
try:
    from simple_byte_tracker import ByteTrackerWrapper as DeepSORT, Detection
    BYTETRACK_AVAILABLE = True
    print("使用SimpleBYTETracker作为跟踪后端")
except ImportError:
    try:
        # 备选：使用原来的DeepSORT
        from deep_sort import DeepSORT, Detection
        BYTETRACK_AVAILABLE = False
        print("ByteTracker不可用，使用原DeepSORT")
    except ImportError:
        raise ImportError("无法导入跟踪器，请检查安装")

def get_absolute_path(*path_parts):
    """获取绝对路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    return os.path.join(project_root, *path_parts)

def draw_rotated_bbox(image, bbox, color, thickness=2):
    """新增：绘制旋转边界框的辅助函数"""
    if len(bbox) < 5:
        # 如果不是旋转框，绘制普通矩形
        x1, y1, x2, y2 = bbox[:4]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return (int(x1), int(y1))
    
    # 旋转框: [cx, cy, w, h, angle]
    center = (int(bbox[0]), int(bbox[1]))
    size = (int(bbox[2]), int(bbox[3]))
    angle = bbox[4] * 180 / math.pi
    
    # 获取旋转矩形的四个角点
    rect = (center, size, angle)
    box_points = cv2.boxPoints(rect)
    box_points = np.int0(box_points)
    
    # 绘制旋转矩形
    cv2.drawContours(image, [box_points], 0, color, thickness)
    
    # 绘制中心点
    cv2.circle(image, center, 3, color, -1)
    
    # 绘制方向线（表示车辆朝向）
    direction_length = min(size) // 4
    end_x = int(center[0] + direction_length * math.cos(bbox[4]))
    end_y = int(center[1] + direction_length * math.sin(bbox[4]))
    cv2.line(image, center, (end_x, end_y), color, thickness)
    
    return center

class MultiObjectTracker:
    """改进的多目标跟踪器类 - 使用SimpleBYTETracker"""
    
    def __init__(self, config_file, checkpoint_file, device='cuda:0', 
                 target_classes=['car', 'truck', 'bus'],
                 score_thr=0.5,
                 tracker_params=None):
        """
        初始化多目标跟踪器
        
        Args:
            config_file: 模型配置文件路径
            checkpoint_file: 模型权重文件路径
            device: 设备
            target_classes: 要跟踪的目标类别
            score_thr: 检测置信度阈值
            tracker_params: 跟踪器参数
        """
        # 初始化检测模型
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.target_classes = target_classes
        self.score_thr = score_thr
        
        # 初始化SimpleBYTETracker
        if tracker_params is None:
            if BYTETRACK_AVAILABLE:
                # SimpleBYTETracker参数 - 针对车辆跟踪优化
                tracker_params = {
                    'frame_rate': 30,
                    'track_thresh': 0.3,        # 高置信度阈值
                    'track_buffer': 60,         # 跟踪缓冲帧数
                    'match_thresh': 0.6,        # 匹配阈值
                    'min_box_area': 20,        # 最小框面积
                    'mot20': False
                }
            else:
                # DeepSORT参数
                tracker_params = {
                    'max_age': 50,
                    'n_init': 3,
                    'max_iou_distance': 0.3,
                    'max_cosine_distance': 0.2
                }
        
        self.tracker = DeepSORT(**tracker_params)
        
        # 获取模型类别
        if hasattr(self.model, 'CLASSES'):
            self.model_classes = self.model.CLASSES
        elif hasattr(self.model, 'dataset_meta') and 'classes' in self.model.dataset_meta:
            self.model_classes = self.model.dataset_meta['classes']
        else:
            # 默认类别
            self.model_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
        
        backend_name = "SimpleBYTETracker" if BYTETRACK_AVAILABLE else "DeepSORT"
        print(f"跟踪后端: {backend_name}")
        print(f"模型类别: {self.model_classes}")
        print(f"跟踪类别: {self.target_classes}")
        print(f"跟踪器参数: {tracker_params}")
    
    def extract_detections(self, result, frame):
        """从检测结果中提取目标类别的检测"""
        detections = []
        
        if isinstance(result, tuple):
            bbox_result, _ = result
        else:
            bbox_result = result
            
        for class_id, class_name in enumerate(self.model_classes):
            if class_name in self.target_classes and class_id < len(bbox_result):
                class_bboxes = bbox_result[class_id]
                for bbox in class_bboxes:
                    if len(bbox) >= 6:  # 旋转框格式
                        if bbox[5] > self.score_thr:
                            detection = Detection(bbox[:5], bbox[5], class_name)
                            detections.append(detection)
                    elif len(bbox) >= 5:  # 普通框格式
                        if bbox[4] > self.score_thr:
                            detection = Detection(bbox[:4], bbox[4], class_name)
                            detections.append(detection)
        
        return detections
    
    def process_frame(self, frame, frame_id, timestamp):
        """处理单帧"""
        result = inference_detector(self.model, frame)
        detections = self.extract_detections(result, frame)
        tracks = self.tracker.update(detections, frame_id, timestamp, frame)
        return tracks, result
    
    def visualize_tracks(self, frame, tracks, result=None):
        """修改：可视化跟踪结果 - 支持旋转框显示"""
        vis_frame = frame.copy()
        
        class_colors = {
            'car': (0, 255, 0),
            'truck': (255, 0, 0),
            'bus': (0, 0, 255),
            'van': (255, 255, 0),
            'motorcycle': (255, 0, 255),
            'bicycle': (0, 255, 255),
        }
        
        for track in tracks:
            color = class_colors.get(track.class_name, (128, 128, 128))
            
            # 修改：检查是否有原始旋转框信息
            if hasattr(track, 'is_rotated') and track.is_rotated and hasattr(track, 'original_bbox'):
                text_pos = draw_rotated_bbox(vis_frame, track.original_bbox, color)
            else:
                # 使用水平框
                bbox = track.bbox if hasattr(track, 'bbox') else track.tlbr
                text_pos = draw_rotated_bbox(vis_frame, bbox, color)
            
            # 显示轨迹信息
            track_id = track.formatted_id if hasattr(track, 'formatted_id') else f"{track.class_name}_{track.track_id}"
            cv2.putText(vis_frame, track_id, 
                       (text_pos[0], text_pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, color, 2)
            
            confidence = track.confidence if hasattr(track, 'confidence') else 0.0
            conf_text = f'{confidence:.2f}'
            cv2.putText(vis_frame, conf_text, 
                       (text_pos[0], text_pos[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
        
        return vis_frame
    
    def get_track_points(self) -> List[Dict]:
        """获取所有轨迹点 - 按要求格式"""
        return self.tracker.get_all_track_points()
    
    def get_vehicle_sequences(self) -> List[List[Dict]]:
        """获取每辆车的轨迹序列 - 按要求格式"""
        return self.tracker.get_vehicle_sequences()
    
    def get_track_points_by_class(self) -> Dict[str, List[Dict]]:
        """按类别分组获取轨迹点"""
        return self.tracker.get_track_points_by_class()
    
    def get_statistics(self):
        """获取跟踪统计信息"""
        return self.tracker.get_statistics()

def track_video(config_file, checkpoint_file, input_video_path, output_dir,
               device='cuda:0', target_classes=['car', 'truck', 'bus'], 
               score_thr=0.5, save_video=True, save_data=True):
    """使用SimpleBYTETracker进行视频跟踪"""
    
    # 文件检查
    for file_path in [config_file, checkpoint_file, input_video_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化跟踪器
    tracker = MultiObjectTracker(
        config_file=config_file,
        checkpoint_file=checkpoint_file,
        device=device,
        target_classes=target_classes,
        score_thr=score_thr
    )
    
    # 视频处理
    video_reader = mmcv.VideoReader(input_video_path)
    if not video_reader.opened:
        raise RuntimeError(f"无法打开视频: {input_video_path}")

    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    backend_suffix = "simple_bytetrack" if BYTETRACK_AVAILABLE else "deepsort"
    
    video_writer = None
    if save_video:
        output_video_file = os.path.join(output_dir, f"{base_name}_{backend_suffix}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_file, fourcc, 
                                     video_reader.fps, video_reader.resolution)

    frame_id = 0
    start_time = time.time()
    
    backend_name = "SimpleBYTETracker" if BYTETRACK_AVAILABLE else "DeepSORT"
    print(f"开始使用{backend_name}处理视频: {input_video_path}")
    
    try:
        for frame in video_reader:
            timestamp = start_time + frame_id / video_reader.fps
            tracks, result = tracker.process_frame(frame, frame_id, timestamp)
            
            if save_video:
                vis_frame = tracker.visualize_tracks(frame, tracks, result)
                video_writer.write(vis_frame)
            
            frame_id += 1
            if frame_id % 30 == 0:
                stats = tracker.get_statistics()
                print(f"已处理 {frame_id} 帧，当前跟踪 {len(tracks)} 个目标")
                print(f"  类别统计: {stats.get('active_tracks_by_class', {})}")

    finally:
        if video_writer:
            video_writer.release()
    
    # 保存数据
    all_track_points = tracker.get_track_points()
    vehicle_sequences = tracker.get_vehicle_sequences()
    
    if save_data:
        all_points_file = os.path.join(output_dir, f"{base_name}_{backend_suffix}_track_points.json")
        with open(all_points_file, 'w', encoding='utf-8') as f:
            json.dump(all_track_points, f, indent=2, ensure_ascii=False)
        
        sequences_file = os.path.join(output_dir, f"{base_name}_{backend_suffix}_vehicle_sequences.json")
        with open(sequences_file, 'w', encoding='utf-8') as f:
            json.dump(vehicle_sequences, f, indent=2, ensure_ascii=False)
        
        track_points_by_class = tracker.get_track_points_by_class()
        for class_name, class_tracks in track_points_by_class.items():
            class_output_file = os.path.join(output_dir, f"{base_name}_{backend_suffix}_{class_name}_tracks.json")
            with open(class_output_file, 'w', encoding='utf-8') as f:
                json.dump(class_tracks, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    final_stats = tracker.get_statistics()
    print(f"\n=== {backend_name}跟踪完成 ===")
    if save_video:
        print(f"跟踪视频: {output_video_file}")
    if save_data:
        print(f"轨迹数据: {all_points_file}")
        print(f"车辆序列: {sequences_file}")
    
    print(f"总帧数: {frame_id}")
    print(f"总轨迹点数: {len(all_track_points)}")
    print(f"车辆数量: {len(vehicle_sequences)}")
    print(f"总轨迹数: {final_stats.get('total_tracks', 0)}")
    print(f"确认轨迹数: {final_stats.get('confirmed_tracks', 0)}")
    
    for class_name, count in final_stats.get('class_counts', {}).items():
        class_vehicles = [seq for seq in vehicle_sequences if seq[0]['class_name'] == class_name]
        print(f"{class_name}: 创建 {count} 个轨迹，有效车辆 {len(class_vehicles)} 辆")
    
    return all_track_points, vehicle_sequences

if __name__ == '__main__':
    # 配置路径
    cfg_path = get_absolute_path('configs', 'detection', 'rtmdet_custom.py')
    ckpt_path = get_absolute_path('checkpoints', 'rtmdet_tiny.pth')
    input_video = get_absolute_path('data', 'raw_videos', '屏幕录制 2025-05-31 170124.mp4')
    output_folder = get_absolute_path('output')

    # 执行ByteTracker跟踪
    track_points, vehicle_sequences = track_video(
        config_file=cfg_path,
        checkpoint_file=ckpt_path,
        input_video_path=input_video,
        output_dir=output_folder,
        device='cuda:0',
        target_classes=['car', 'truck', 'bus'],
        score_thr=0.25,
        save_video=True,
        save_data=True
    )
    
    backend_name = "SimpleBYTETracker" if BYTETRACK_AVAILABLE else "DeepSORT"
    print(f"\n=== {backend_name}最终结果 ===")
    print(f"总轨迹点数: {len(track_points)}")
    print(f"车辆数量: {len(vehicle_sequences)}")
    
    # 输出符合要求的格式示例
    if vehicle_sequences:
        print(f"\n=== 输出格式示例 ===")
        example_vehicle = vehicle_sequences[0]
        print(f"单车轨迹格式示例 ({example_vehicle[0]['track_id']}):")
        for point in example_vehicle[:2]:
            print(f"  {point}")