import cv2
import numpy as np
import argparse
import os
import time
from typing import List, Tuple, Optional

class AerialLaneDetector:
    def __init__(self):
        # 初始化参数
        # Canny边缘检测参数
        self.canny_threshold1 = 80
        self.canny_threshold2 = 150
        self.hough_rho = 1
        self.hough_theta = np.pi/180
        self.hough_threshold = 70
        self.min_line_length = 100
        self.max_line_gap = 10
        
        # 上一帧车道线基点
        self.prev_lane_bases = None  # (left1_base, left2_base, right1_base, right2_base)
        self.change_threshold = 10  # 降低变化阈值到10像素
        self.smoothing_factor = 0.7  # 平滑因子(0-1)
        
        # 颜色和纹理参数
        self.road_color_lower = np.array([50, 0, 20])   
        self.road_color_upper = np.array([160, 120, 180])
        self.texture_threshold = 40
        self.line_color = (0, 255, 0)  # BGR绿色
        self.emergency_color = (0, 0, 255)  # BGR红色
        
        # 俯瞰图变换参数
        self.perspective_src = None
        self.perspective_dst = None
        self.M = None
        self.M_inv = None
        
        # 滑动窗口参数
        self.window_width = 50  # 保持窗口宽度不变
        self.window_height = 10  # 将高度减半
        self.min_pix =20      # 降低最小像素阈值以适应更小窗口
        
        # 输出目录
        self.output_dir = "data\output"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_road_roi(self, img: np.ndarray) -> np.ndarray:
        """获取基础ROI区域（梯形区域）"""
        height, width = img.shape[:2]
        roi_vertices = np.array([[
            (int(width * 0), int(height * 1)),  # 左下
            (int(width * 0.3), int(height * 0.5)),  # 左上
            (int(width * 0.7), int(height * 0.5)),  # 右上
            (int(width * 1), int(height * 1))   # 右下
        ]], dtype=np.int32)
        mask = np.zeros_like(img[:,:,0])
        cv2.fillPoly(mask, [roi_vertices], 255)
        
        # 设置俯瞰图变换的源点
        self.perspective_src = np.float32(roi_vertices[0])
        
        # 设置俯瞰图变换的目标点
        dst_width = width
        dst_height = height
        self.perspective_dst = np.float32([
            [0, dst_height],
            [0, 0],
            [dst_width, 0],
            [dst_width, dst_height]
        ])
        
        # 计算变换矩阵和逆矩阵
        self.M = cv2.getPerspectiveTransform(self.perspective_src, self.perspective_dst)
        self.M_inv = cv2.getPerspectiveTransform(self.perspective_dst, self.perspective_src)
        
        return mask

    def detect_road_features(self, img: np.ndarray) -> np.ndarray:
        """结合颜色和纹理特征检测道路区域"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.road_color_lower, self.road_color_upper)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        gradient = np.uint8(255 * gradient / gradient.max())
        _, texture_mask = cv2.threshold(gradient, self.texture_threshold, 255, cv2.THRESH_BINARY_INV)
        
        road_mask = cv2.bitwise_and(color_mask, texture_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        return cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

    def preprocess(self, img: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, float]:
        """改进的预处理流程"""
        original_h, original_w = img.shape[:2]
        target_w = 2400  # 进一步放大图像
        ratio = 1.0
        if original_w > target_w:
            ratio = target_w / original_w
            img = cv2.resize(img, (target_w, int(original_h * ratio)))

        roi_mask = self.get_road_roi(img)
        feature_mask = self.detect_road_features(img)
        combined_mask = cv2.bitwise_and(roi_mask, feature_mask)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0,0,50]), np.array([180,100,255]))
        yellow_mask = cv2.inRange(hsv, np.array([5,30,30]), np.array([50,255,255]))
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        color_mask = cv2.bitwise_and(color_mask, combined_mask)
        edges = cv2.bitwise_and(edges, combined_mask)
        final_edges = cv2.bitwise_and(edges, color_mask)

        if debug:
            debug_img = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
            debug_img = cv2.addWeighted(img, 0.7, debug_img, 0.3, 0)
            cv2.imshow("Combined Mask", debug_img)
            cv2.imshow("Color Mask", color_mask)
            cv2.imshow("Final Edges", final_edges)
            
        return final_edges, ratio

    def detect_lanes(self, edges: np.ndarray) -> List[np.ndarray]:
        """霍夫变换检测线段"""
        lines = cv2.HoughLinesP(
            edges, self.hough_rho, self.hough_theta, self.hough_threshold,
            minLineLength=self.min_line_length, maxLineGap=self.max_line_gap
        )
        return lines if lines is not None else []

    def filter_lines(self, lines: List[np.ndarray], img_shape: tuple) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """按角度分类线段"""
        left, right = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2-y1, x2-x1) * 180/np.pi
            (left if angle < 0 else right).append(line[0])
        return left, right

    def merge_close_lines(self, lines: List[np.ndarray], max_dist: int = 30, max_angle_diff: float = 5.0) -> List[np.ndarray]:
        """合并相近的车道线"""
        if not lines:
            return []

        line_info = []
        for line in lines:
            x1, y1, x2, y2 = line
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            angle = np.arctan2(y2-y1, x2-x1) * 180/np.pi
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            line_info.append((line, mid_x, mid_y, angle, length))

        line_info.sort(key=lambda x: x[1])

        merged = []
        current = line_info[0]
        
        for next_line in line_info[1:]:
            dist = np.sqrt((next_line[1]-current[1])**2 + (next_line[2]-current[2])**2)
            angle_diff = abs(next_line[3] - current[3])
            
            if dist < max_dist and angle_diff < max_angle_diff:
                current = current if current[4] > next_line[4] else next_line
            else:
                merged.append(current[0])
                current = next_line
        
        merged.append(current[0])
        return merged

    def perspective_transform(self, img: np.ndarray) -> np.ndarray:
        """执行俯瞰图变换"""
        if self.M is None:
            return img
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def inverse_perspective_transform(self, img: np.ndarray) -> np.ndarray:
        """执行逆俯瞰图变换"""
        if self.M_inv is None:
            return img
        return cv2.warpPerspective(img, self.M_inv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def find_lane_line_base(self, binary_warped: np.ndarray, left_lines: List[np.ndarray], right_lines: List[np.ndarray]) -> Tuple[int, int, int, int]:
        """找到左右车道线的起点"""
        # 对左右车道线排序
        left_sorted = sorted(left_lines, key=lambda x: (x[0] + x[2])/2)
        right_sorted = sorted(right_lines, key=lambda x: (x[0] + x[2])/2)
        
        # 获取左一、左二、右一、右二的起点
        left1_base = right1_base = left2_base = right2_base = None
        
        if len(left_sorted) >= 1:
            left1_base = self.get_line_base(binary_warped, left_sorted[0])
            # 寻找左二，确保与左一距离至少30像素
            if len(left_sorted) >= 2:
                for i in range(1, len(left_sorted)):
                    candidate_base = self.get_line_base(binary_warped, left_sorted[i])
                    if abs(candidate_base - left1_base) >= 30:
                        left2_base = candidate_base
                        break
        
        if len(right_sorted) >= 1:
            right1_base = self.get_line_base(binary_warped, right_sorted[-1])
            # 寻找右二，确保与右一距离至少30像素
            if len(right_sorted) >= 2:
                for i in range(len(right_sorted)-2, -1, -1):
                    candidate_base = self.get_line_base(binary_warped, right_sorted[i])
                    if abs(candidate_base - right1_base) >= 30:
                        right2_base = candidate_base
                        break
            
        return left1_base, left2_base, right1_base, right2_base

    def get_line_base(self, binary_warped: np.ndarray, line: np.ndarray) -> int:
        """获取车道线在底部的x坐标"""
        x1, y1, x2, y2 = line
        if y1 > y2:  # 确保y1是较低的点
            x1, y1, x2, y2 = x2, y2, x1, y1
        
        # 延长线到底部
        height = binary_warped.shape[0]
        if x1 == x2:  # 垂直线
            return x1
        
        k = (y2 - y1) / (x2 - x1)
        if k == 0:  # 水平线
            return x1  # 水平线无法延长到底部，返回任意x坐标
            
        base_x = int(x1 + (height - y1) / k)
        return base_x

    def sliding_window_search(self, binary_warped: np.ndarray, x_base: int) -> Tuple[Optional[np.ndarray], List[Tuple[int, int, int, int]]]:
        """使用滑动窗口搜索车道线
        返回: (拟合多项式系数, 窗口位置列表[(x1,y1,x2,y2), ...])
        """
        if x_base is None:
            return None, []
            
        # 设置滑动窗口的数量
        nwindows = 40  # 增加窗口数量使实际高度减半
        window_height = max(binary_warped.shape[0] // nwindows, self.window_height)  # 确保不小于最小高度
        
        # 识别二进制图像中非零像素的x和y位置
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # 当前窗口的边界位置
        current_x = x_base
        
        # 设置最小像素数来重新定位窗口
        minpix = self.min_pix
        
        # 创建空列表来接收车道像素索引和窗口位置
        lane_inds = []
        window_positions = []
        
        # 逐步遍历每个窗口
        for window in range(nwindows):
            # 识别窗口边界
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_x_low = current_x - self.window_width // 2
            win_x_high = current_x + self.window_width // 2
            
            # 记录窗口位置
            window_positions.append((win_x_low, win_y_low, win_x_high, win_y_high))
            
            # 识别窗口中的非零像素
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                        (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            
            # 将这些索引追加到列表中
            lane_inds.append(good_inds)
            
            # 如果找到的像素数大于minpix，则重新定位下一个窗口
            if len(good_inds) > minpix:
                current_x = int(np.mean(nonzerox[good_inds]))
        
        # 连接索引数组
        lane_inds = np.concatenate(lane_inds)
        
        # 提取车道线像素位置
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        
        if len(x) == 0 or len(y) == 0:
            return None, window_positions
        
        # 拟合二阶多项式
        fit = np.polyfit(y, x, 2)
        return fit, window_positions

    def draw_lane_lines(self, img: np.ndarray, binary_warped: np.ndarray, 
                      left1_fit: Optional[np.ndarray], left2_fit: Optional[np.ndarray],
                      right1_fit: Optional[np.ndarray], right2_fit: Optional[np.ndarray],
                      left1_windows: List[Tuple[int, int, int, int]] = [],
                      left2_windows: List[Tuple[int, int, int, int]] = [],
                      right1_windows: List[Tuple[int, int, int, int]] = [],
                      right2_windows: List[Tuple[int, int, int, int]] = []) -> Tuple[np.ndarray, np.ndarray]:
        """绘制车道线和应急车道区域，并返回俯瞰图窗口"""
        # 创建一个空白图像来绘制车道线和窗口
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # 绘制滑动窗口
        window_color = (255, 0, 0)  # BGR蓝色
        for win in left1_windows:
            cv2.rectangle(color_warp, (win[0], win[1]), (win[2], win[3]), window_color, 2)
        for win in left2_windows:
            cv2.rectangle(color_warp, (win[0], win[1]), (win[2], win[3]), window_color, 2)
        for win in right1_windows:
            cv2.rectangle(color_warp, (win[0], win[1]), (win[2], win[3]), window_color, 2)
        for win in right2_windows:
            cv2.rectangle(color_warp, (win[0], win[1]), (win[2], win[3]), window_color, 2)
        
        # 生成y值
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        
        # 绘制左一车道线
        if left1_fit is not None:
            left1_fitx = left1_fit[0]*ploty**2 + left1_fit[1]*ploty + left1_fit[2]
            pts_left1 = np.array([np.transpose(np.vstack([left1_fitx, ploty]))])
            cv2.polylines(color_warp, np.int32([pts_left1]), False, self.line_color, thickness=10)
        
        # 绘制左二车道线
        if left2_fit is not None:
            left2_fitx = left2_fit[0]*ploty**2 + left2_fit[1]*ploty + left2_fit[2]
            pts_left2 = np.array([np.transpose(np.vstack([left2_fitx, ploty]))])
            cv2.polylines(color_warp, np.int32([pts_left2]), False, self.line_color, thickness=10)
        
        # 绘制右一车道线
        if right1_fit is not None:
            right1_fitx = right1_fit[0]*ploty**2 + right1_fit[1]*ploty + right1_fit[2]
            pts_right1 = np.array([np.transpose(np.vstack([right1_fitx, ploty]))])
            cv2.polylines(color_warp, np.int32([pts_right1]), False, self.line_color, thickness=10)
        
        # 绘制右二车道线
        if right2_fit is not None:
            right2_fitx = right2_fit[0]*ploty**2 + right2_fit[1]*ploty + right2_fit[2]
            pts_right2 = np.array([np.transpose(np.vstack([right2_fitx, ploty]))])
            cv2.polylines(color_warp, np.int32([pts_right2]), False, self.line_color, thickness=10)
        
        # 填充应急车道区域
        if left1_fit is not None and left2_fit is not None:
            left1_fitx = left1_fit[0]*ploty**2 + left1_fit[1]*ploty + left1_fit[2]
            left2_fitx = left2_fit[0]*ploty**2 + left2_fit[1]*ploty + left2_fit[2]
            pts_left = np.array([np.transpose(np.vstack([left1_fitx, ploty]))])
            pts_left = np.hstack((pts_left, np.array([np.flipud(np.transpose(np.vstack([left2_fitx, ploty])))])))
            cv2.fillPoly(color_warp, np.int_([pts_left]), self.emergency_color)
        
        if right1_fit is not None and right2_fit is not None:
            right1_fitx = right1_fit[0]*ploty**2 + right1_fit[1]*ploty + right1_fit[2]
            right2_fitx = right2_fit[0]*ploty**2 + right2_fit[1]*ploty + right2_fit[2]
            pts_right = np.array([np.transpose(np.vstack([right1_fitx, ploty]))])
            pts_right = np.hstack((pts_right, np.array([np.flipud(np.transpose(np.vstack([right2_fitx, ploty])))])))
            cv2.fillPoly(color_warp, np.int_([pts_right]), self.emergency_color)
        
        # 将俯瞰图变换回原始图像空间
        newwarp = self.inverse_perspective_transform(color_warp)
        
        # 确保newwarp和原始图像尺寸相同
        if newwarp.shape != img.shape:
            newwarp = cv2.resize(newwarp, (img.shape[1], img.shape[0]))
        
        # 将结果与原始图像结合
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        
        # 创建俯瞰图窗口图像
        birdseye_view = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)
        birdseye_view = cv2.addWeighted(birdseye_view, 0.7, color_warp, 0.3, 0)
        
        return result, birdseye_view

    def process_frame(self, img: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        original_height, original_width = img.shape[:2]
        edges, ratio = self.preprocess(img, debug)
        lines = self.detect_lanes(edges)
        left_lines, right_lines = self.filter_lines(lines, img.shape)
        left_lines = self.merge_close_lines(left_lines)
        right_lines = self.merge_close_lines(right_lines)
        
        # 初始化seg_mask (0表示非禁停区)
        seg_mask = np.zeros_like(img[:,:,0], dtype=np.uint8)
        
        # 创建霍夫直线二值图
        hough_binary = np.zeros_like(edges)
        for line in left_lines + right_lines:
            x1, y1, x2, y2 = line
            cv2.line(hough_binary, (x1, y1), (x2, y2), 255, 2)
        
        # 执行俯瞰图变换
        binary_warped = self.perspective_transform(hough_binary)
        
        # 找到车道线起点
        left1_base, left2_base, right1_base, right2_base = self.find_lane_line_base(
            binary_warped, left_lines, right_lines)
            
        # 检查与上一帧的变化并平滑处理
        if self.prev_lane_bases is not None:
            prev_left1, prev_left2, prev_right1, prev_right2 = self.prev_lane_bases
            
            # 对每条线单独检查变化并平滑处理
            if left1_base is not None and prev_left1 is not None:
                if abs(left1_base - prev_left1) > self.change_threshold:
                    left1_base = int(self.smoothing_factor * prev_left1 + (1-self.smoothing_factor) * left1_base)
            
            if left2_base is not None and prev_left2 is not None:
                if abs(left2_base - prev_left2) > self.change_threshold:
                    left2_base = int(self.smoothing_factor * prev_left2 + (1-self.smoothing_factor) * left2_base)
            
            if right1_base is not None and prev_right1 is not None:
                if abs(right1_base - prev_right1) > self.change_threshold:
                    right1_base = int(self.smoothing_factor * prev_right1 + (1-self.smoothing_factor) * right1_base)
            
            if right2_base is not None and prev_right2 is not None:
                if abs(right2_base - prev_right2) > self.change_threshold:
                    right2_base = int(self.smoothing_factor * prev_right2 + (1-self.smoothing_factor) * right2_base)
                
        # 更新上一帧结果
        self.prev_lane_bases = (left1_base, left2_base, right1_base, right2_base)
        
        # 使用滑动窗口搜索车道线并获取窗口位置
        left1_fit, left1_windows = self.sliding_window_search(binary_warped, left1_base)
        left2_fit, left2_windows = self.sliding_window_search(binary_warped, left2_base)
        right1_fit, right1_windows = self.sliding_window_search(binary_warped, right1_base)
        right2_fit, right2_windows = self.sliding_window_search(binary_warped, right2_base)
        
        if ratio != 1.0:
            img = cv2.resize(img, (original_width, original_height))
        
        # 绘制结果(包括滑动窗口)
        result_img, birdseye_view = self.draw_lane_lines(
            img, binary_warped, 
            left1_fit, left2_fit, right1_fit, right2_fit,
            left1_windows, left2_windows, right1_windows, right2_windows
        )
        
        # 创建应急车道区域mask (1表示禁停区)
        if left1_fit is not None and left2_fit is not None:
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            left1_fitx = left1_fit[0]*ploty**2 + left1_fit[1]*ploty + left1_fit[2]
            left2_fitx = left2_fit[0]*ploty**2 + left2_fit[1]*ploty + left2_fit[2]
            pts_left = np.array([np.transpose(np.vstack([left1_fitx, ploty]))])
            pts_left = np.hstack((pts_left, np.array([np.flipud(np.transpose(np.vstack([left2_fitx, ploty])))])))
            cv2.fillPoly(seg_mask, np.int_([pts_left]), 1)
        
        if right1_fit is not None and right2_fit is not None:
            right1_fitx = right1_fit[0]*ploty**2 + right1_fit[1]*ploty + right1_fit[2]
            right2_fitx = right2_fit[0]*ploty**2 + right2_fit[1]*ploty + right2_fit[2]
            pts_right = np.array([np.transpose(np.vstack([right1_fitx, ploty]))])
            pts_right = np.hstack((pts_right, np.array([np.flipud(np.transpose(np.vstack([right2_fitx, ploty])))])))
            cv2.fillPoly(seg_mask, np.int_([pts_right]), 1)
            
        # 对seg_mask执行逆透视变换
        seg_mask = self.inverse_perspective_transform(seg_mask)
        
        return result_img, edges, birdseye_view, seg_mask

    def process_image(self, img_path: str, debug: bool = False):
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法加载图片: {img_path}")
            return
            
        try:
            result, edges, birdseye, seg_mask = self.process_frame(img, debug)
            # 保存seg_mask
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(self.output_dir, f"{img_name}_seg_mask.jpg"), seg_mask*255)
            if debug:
                cv2.imshow("禁停区掩码", seg_mask*255)
            
            # 保存结果图像
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(self.output_dir, f"{img_name}_result.jpg"), result)
            cv2.imwrite(os.path.join(self.output_dir, f"{img_name}_edges.jpg"), edges)
            cv2.imwrite(os.path.join(self.output_dir, f"{img_name}_birdseye.jpg"), birdseye)
            
            cv2.imshow("航拍车道检测结果", result)
            cv2.imshow("边缘检测图", edges)
            cv2.imshow("俯瞰图检测过程", birdseye)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or cv2.getWindowProperty("航拍车道检测结果", cv2.WND_PROP_VISIBLE) < 1:
                    break
            
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"处理图片时发生错误: {e}")
            cv2.destroyAllWindows()
        
    def process_video(self, video_path: str, debug: bool = False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_subdir = os.path.join(self.output_dir, video_name)
        os.makedirs(output_subdir, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(
            os.path.join(output_subdir, f'processed_{video_name}.mp4'),
            fourcc, fps, (width, height))
        gray_video = cv2.VideoWriter(
            os.path.join(output_subdir, f'gray_{video_name}.mp4'),
            fourcc, fps, (width, height), isColor=False)
        birdseye_video = cv2.VideoWriter(
            os.path.join(output_subdir, f'birdseye_{video_name}.mp4'),
            fourcc, fps, (width, height))
        seg_mask_video = cv2.VideoWriter(
            os.path.join(output_subdir, f'seg_mask_{video_name}.mp4'),
            fourcc, fps, (width, height), isColor=False)
        
        try:
            cv2.namedWindow("航拍车道检测结果", cv2.WINDOW_NORMAL)
            cv2.namedWindow("边缘检测图", cv2.WINDOW_NORMAL)
            cv2.namedWindow("俯瞰图检测过程", cv2.WINDOW_NORMAL)
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                result, edges, birdseye, seg_mask = self.process_frame(frame, debug)
                if result is not None:
                    output_video.write(result)
                    cv2.imshow("航拍车道检测结果", result)
                if edges is not None:
                    gray_video.write(edges)
                    cv2.imshow("边缘检测图", edges)
                if birdseye is not None:
                    birdseye_video.write(birdseye)
                    cv2.imshow("俯瞰图检测过程", birdseye)
                if seg_mask is not None:
                    seg_mask_video.write(seg_mask*255)
                    if debug:
                        cv2.imshow("禁停区掩码", seg_mask*255)
                frame_count += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            output_video.release()
            gray_video.release()
            birdseye_video.release()
            seg_mask_video.release()
            
            print(f"视频处理完成，结果已保存到: {output_subdir}")
        except Exception as e:
            print(f"处理视频时发生错误: {e}")
        finally:
            cv2.destroyAllWindows()
        
    def process_realtime(self, cam_id: int = 0, debug: bool = False):
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print("无法打开摄像头")
            return
            
        try:
            cv2.namedWindow("实时车道检测结果", cv2.WINDOW_NORMAL)
            cv2.namedWindow("边缘检测图", cv2.WINDOW_NORMAL)
            cv2.namedWindow("俯瞰图检测过程", cv2.WINDOW_NORMAL)
            if debug:
                cv2.namedWindow("禁停区掩码", cv2.WINDOW_NORMAL)
            
            # 为实时视频创建输出目录
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_subdir = os.path.join(self.output_dir, f"realtime_{timestamp}")
            os.makedirs(output_subdir, exist_ok=True)
            
            # 获取第一帧以确定尺寸
            ret, frame = cap.read()
            if not ret:
                print("无法获取帧")
                return
                
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(
                os.path.join(output_subdir, 'realtime_output.mp4'),
                fourcc, 20.0, (width, height))
            gray_video = cv2.VideoWriter(
                os.path.join(output_subdir, 'realtime_edges.mp4'),
                fourcc, 20.0, (width, height), isColor=False)
            birdseye_video = cv2.VideoWriter(
                os.path.join(output_subdir, 'realtime_birdseye.mp4'),
                fourcc, 20.0, (width, height))
                
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法获取帧")
                    break
                    
                result, edges, birdseye, seg_mask = self.process_frame(frame, debug)
                
                # 保存帧
                output_video.write(result)
                gray_video.write(edges)
                birdseye_video.write(birdseye)
                
                cv2.imshow("实时车道检测结果", result)
                cv2.imshow("边缘检测图", edges)
                cv2.imshow("俯瞰图检测过程", birdseye)
                if debug:
                    cv2.imshow("禁停区掩码", seg_mask*255)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            output_video.release()
            gray_video.release()
            birdseye_video.release()
            
            print(f"实时视频结果已保存到: {output_subdir}")
        except Exception as e:
            print(f"实时处理时发生错误: {e}")
        finally:
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="航拍车道检测系统")
    parser.add_argument("--image", help="图片路径")
    parser.add_argument("--video", help="视频路径") 
    parser.add_argument("--realtime", action="store_true", help="使用实时摄像头")
    parser.add_argument("--cam", type=int, default=0, help="摄像头ID(默认0)")
    parser.add_argument("--debug", action="store_true", help="显示调试信息")
    
    args = parser.parse_args()
    detector = AerialLaneDetector()
    
    if args.image or args.video or args.realtime:
        if args.image:
            detector.process_image(args.image, args.debug)
        elif args.video:
            detector.process_video(args.video, args.debug)
        elif args.realtime:
            detector.process_realtime(args.cam, args.debug)
    else:
        print("\n=== 航拍车道检测系统 ===")
        print("1. 检测图片")
        print("2. 检测视频")
        print("3. 实时检测")
        print("4. 退出")
        
        while True:
            choice = input("\n请选择模式(1-4): ").strip()
            
            if choice == '1':
                img_path = input("请输入图片路径: ").strip()
                if os.path.exists(img_path):
                    detector.process_image(img_path)
                else:
                    print("错误: 文件不存在")
            elif choice == '2':
                video_path = input("请输入视频路径: ").strip()
                if os.path.exists(video_path):
                    detector.process_video(video_path)
                else:
                    print("错误: 文件不存在")
            elif choice == '3':
                cam_id = input("请输入摄像头ID(默认0): ").strip()
                try:
                    cam_id = int(cam_id) if cam_id else 0
                    detector.process_realtime(cam_id)
                except ValueError:
                    print("错误: 请输入数字")
            elif choice == '4':
                print("退出系统")
                break
            else:
                print("无效选择，请重新输入")

if __name__ == "__main__":
    main()
