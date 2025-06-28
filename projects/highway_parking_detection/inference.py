import os
import mmcv
import cv2
from mmdet.apis import init_detector, inference_detector

def get_absolute_path(*path_parts):
    """修正版路径转换函数"""
    # 获取当前脚本所在目录（projects/highway_parking_detection/）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录（Highway-Parking-Detection/）
    project_root = os.path.dirname(os.path.dirname(script_dir))  # 关键修改：多向上退一级
    return os.path.join(project_root, *path_parts)

def inference_video(config_file, checkpoint_file, input_video_path, output_dir, device='cuda:0', score_thr=0.3):
    # 标准化路径（确保使用系统正确的分隔符）
    config_file = os.path.normpath(config_file)
    checkpoint_file = os.path.normpath(checkpoint_file)
    input_video_path = os.path.normpath(input_video_path)
    
    print(f"🔍 配置文件路径: {config_file}")
    print(f"🔍 模型权重路径: {checkpoint_file}")
    print(f"🔍 输入视频路径: {input_video_path}")
    
    # 检查文件是否存在
    if not os.path.exists(config_file):
        available = os.path.exists(os.path.dirname(config_file))
        raise FileNotFoundError(
            f"配置文件不存在: {config_file}\n"
            f"目录是否存在: {'是' if available else '否'}\n"
            f"目录内容: {os.listdir(os.path.dirname(config_file)) if available else '无'}"
        )
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"模型权重不存在: {checkpoint_file}")

    # 初始化模型
    model = init_detector(config_file, checkpoint_file, device=device)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取视频
    video_reader = mmcv.VideoReader(input_video_path)
    if not video_reader.isOpened():
        raise RuntimeError(f"无法打开视频文件: {input_video_path}")

    # 构造输出视频文件路径
    base_name = os.path.basename(input_video_path)
    output_file = os.path.join(output_dir, base_name)

    # 视频编码器和写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, video_reader.fps, video_reader.resolution)

    # 推理并写入
    for frame in video_reader:
        result = inference_detector(model, frame)
        vis_frame = model.show_result(frame, result, score_thr=score_thr, show=False)
        video_writer.write(vis_frame)

    video_writer.release()
    print(f'✅ 视频推理完成，结果保存于: {output_file}')

if __name__ == '__main__':
    # 使用修正后的路径获取方式
    cfg_path = get_absolute_path('configs', 'highway_parking_detection', 'rtm_det_custom.py')
    ckpt_path = get_absolute_path('checkpoints', 'rtm_det_tiny.pth')
    input_video = get_absolute_path('data', 'raw_videos', '屏幕录制 2025-05-31 170124.mp4')
    output_folder = get_absolute_path('output')

    # 打印验证路径
    print("🔄 修正后的路径验证：")
    print(f"• 配置文件: {cfg_path} → {'✅存在' if os.path.exists(cfg_path) else '❌不存在'}")
    print(f"• 模型权重: {ckpt_path} → {'✅存在' if os.path.exists(ckpt_path) else '❌不存在'}")
    print(f"• 输入视频: {input_video} → {'✅存在' if os.path.exists(input_video) else '❌不存在'}")

    inference_video(cfg_path, ckpt_path, input_video, output_folder, device='cuda:0')