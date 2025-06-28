import mmrotate.models
from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import cv2  # 引入 OpenCV

# 指定 config 文件和 checkpoint 权重
config_file = r'configs\highway_parking_detection\car_detedt_config.py'
checkpoint_file = r'checkpoint\latest.pth'

# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # 如果没有GPU可改为 'cpu'

# 视频路径
video_path = r'D:\挑战杯程序\视频素材\屏幕录制 2025-05-31 170606.mp4'
output_video_path = r'D:\挑战杯程序\视频素材\result_video2.mp4'
video_reader = mmcv.VideoReader(video_path)

# 使用 OpenCV 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 指定视频编码格式
video_writer = cv2.VideoWriter(output_video_path, fourcc, video_reader.fps, video_reader.resolution)

# 对视频逐帧推理
for frame in video_reader:
    # 推理
    result = inference_detector(model, frame)
    
    # 可视化结果
    frame_with_result = model.show_result(frame, result, score_thr=0.3, show=False)
    
    # 写入结果到新视频
    video_writer.write(frame_with_result)

video_writer.release()
print(f"✅ 视频推理完成，结果保存在：{output_video_path}")
