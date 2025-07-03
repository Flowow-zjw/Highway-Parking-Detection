import os
import mmcv
import cv2
from mmdet.apis import init_detector, inference_detector

def get_absolute_path(*path_parts):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    return os.path.join(project_root, *path_parts)

def inference_video(config_file, checkpoint_file, input_video_path, output_dir, device='cuda:0', score_thr=0.3):
    config_file = os.path.normpath(config_file)
    checkpoint_file = os.path.normpath(checkpoint_file)
    input_video_path = os.path.normpath(input_video_path)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"模型权重不存在: {checkpoint_file}")

    model = init_detector(config_file, checkpoint_file, device=device)
    os.makedirs(output_dir, exist_ok=True)

    video_reader = mmcv.VideoReader(input_video_path)
    if not video_reader.opened:
        raise RuntimeError(f"无法打开视频文件: {input_video_path}")

    base_name = os.path.basename(input_video_path)
    output_file = os.path.join(output_dir, base_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, video_reader.fps, video_reader.resolution)

    for frame in video_reader:
        result = inference_detector(model, frame)
        vis_frame = model.show_result(frame, result, score_thr=score_thr, show=False)
        video_writer.write(vis_frame)

    video_writer.release()
    print(f'视频推理完成，结果保存于: {output_file}')

if __name__ == '__main__':
    cfg_path = get_absolute_path('configs', 'detection', 'rtmdet_custom.py')
    ckpt_path = get_absolute_path('checkpoints', 'rtmdet_tiny.pth')
    input_video = get_absolute_path('data', 'raw_videos', '屏幕录制 2025-05-31 170124.mp4')
    output_folder = get_absolute_path('output')

    inference_video(cfg_path, ckpt_path, input_video, output_folder, device='cuda:0')