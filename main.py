import sys
import cv2
import supervision as sv
from rfdetr import RFDETRSegSmall  # Use smaller model for speed
import time
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

# カルマンフィルタベースのトラック管理
@dataclass
class TrackState:
    """カルマンフィルタで拡張したトラック状態"""
    track_id: int
    kf: object  # カルマンフィルタ
    position_history: list  # 位置の履歴
    velocity: np.ndarray  # 速度ベクトル (vx, vy)
    last_visible_pos: tuple = None  # 最後に見えた位置
    last_visible_velocity: np.ndarray = None  # 最後に見えた速度
    age: int = 0  # トラックの年齢
    time_since_update: int = 0  # 最後の検出からのフレーム数
    
class KalmanFilterTracker:
    """2D カルマンフィルタ実装"""
    def __init__(self, dt=1.0, process_noise=0.01, measurement_noise=1.0):
        """
        dt: サンプリング間隔
        process_noise: システムノイズの大きさ
        measurement_noise: 観測ノイズの大きさ
        """
        self.kf = cv2.KalmanFilter(4, 2)  # 状態4 (x,y,vx,vy), 観測2 (x,y)
        
        # 状態遷移行列
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 測定行列
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # プロセスノイズ共分散行列
        self.kf.processNoiseCov = process_noise * np.eye(4, dtype=np.float32)
        
        # 測定ノイズ共分散行列
        self.kf.measurementNoiseCov = measurement_noise * np.eye(2, dtype=np.float32)
        
        # 推定誤差共分散行列
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
    
    def predict(self):
        """次の状態を予測"""
        prediction = self.kf.predict()
        return (float(prediction[0, 0]), float(prediction[1, 0]))  # (x, y) のタプルを返す
    
    def update(self, measurement):
        """観測で更新"""
        measurement = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        self.kf.correct(measurement)
    
    def get_state(self):
        """現在の状態 (x, y, vx, vy) を取得"""
        return self.kf.statePost.flatten()

# Check CUDA availability and use appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = RFDETRSegSmall(device=device)

# Optimize model for faster inference
model.optimize_for_inference()
print("Model optimized for inference")

# Use thicker/faster annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2)
mask_annotator = sv.MaskAnnotator()

# カルマンフィルタベースのトラック管理
track_states = {}  # track_id -> TrackState
next_track_id = 1

# Video capture settings
video_capture = cv2.VideoCapture(r".\Tracking_2026-02-13 114545.mp4")
if not video_capture.isOpened():
    raise RuntimeError("Failed to open video source")

# Get original video properties
original_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = video_capture.get(cv2.CAP_PROP_FPS)

# Initialize tracker with extended ID retention based on video FPS
# lost_track_buffer: Keep tracks for N frames after losing detection
# Setting to 5 seconds worth of frames to maintain ID even when object leaves view
TRACK_RETENTION_SECONDS = 5
lost_buffer = int(original_fps * TRACK_RETENTION_SECONDS)
tracker = sv.ByteTrack(
    track_activation_threshold=0.35,  # Lower threshold for easier track activation
    lost_track_buffer=lost_buffer,  # Keep tracks for specified seconds after losing detection
    minimum_matching_threshold=0.7,  # High matching threshold to avoid ID switches
    frame_rate=int(original_fps)
)
print(f"Tracker configured: retaining IDs for {TRACK_RETENTION_SECONDS}s ({lost_buffer} frames)")

# Processing settings
PROCESS_WIDTH = 800  # Resize frames to this width for processing
FRAME_SKIP = 2  # Process every Nth frame (1=all frames, 2=every other frame)
DISPLAY_WIDTH = 1280  # Display window width

print(f"Original video: {original_width}x{original_height} @ {original_fps:.1f}fps")
print(f"Processing at: {PROCESS_WIDTH}px width, frame skip: {FRAME_SKIP}")

# FPS counter
fps_start_time = time.time()
fps_counter = 0
fps_display = 0
frame_count = 0

print("Displaying video with OpenCV. Press 'q' to quit.")

def update_kalman_tracks(detections, frame_rgb, frame_resized):
    """カルマンフィルタを使用してトラッキング更新"""
    global track_states, next_track_id
    
    if detections is None or len(detections) == 0:
        # 検出なしの場合：既存トラックを予測のみで更新
        for track_id in list(track_states.keys()):
            track = track_states[track_id]
            track.time_since_update += 1
            
            # 予測位置を計算
            pred = track.kf.predict()
            track.position_history.append(pred)
            
            # 古いトラックは削除
            if track.time_since_update > lost_buffer:
                del track_states[track_id]
        return detections
    
    # ByteTrackで元のトラッキング実行
    detections = tracker.update_with_detections(detections)
    
    # カルマンフィルタで各トラックを更新
    current_track_ids = set()
    
    for i, (xyxy, conf, class_id, tracker_id) in enumerate(zip(
        detections.xyxy,
        detections.confidence,
        detections.class_id,
        detections.tracker_id
    )):
        if tracker_id is None:
            continue
        
        # 人のみを対象（RFDETRではクラスID 1 = person）
        # class_idがNoneまたは1の場合のみ処理
        if class_id is not None:
            if isinstance(class_id, (list, np.ndarray)):
                class_id_val = int(class_id[0]) if len(class_id) > 0 else None
            else:
                class_id_val = int(class_id)
            
            # class_id 1 = person のみ処理
            if class_id_val != 1:
                continue
        
        tracker_id = int(tracker_id)
        current_track_ids.add(tracker_id)
        
        # バウンディングボックスの中心を計算
        x_center = (xyxy[0] + xyxy[2]) / 2
        y_center = (xyxy[1] + xyxy[3]) / 2
        
        if tracker_id not in track_states:
            # 新しいトラックの初期化
            # DTはフレームスキップを考慮
            dt = FRAME_SKIP / original_fps
            kf = KalmanFilterTracker(dt=dt)
            kf.kf.statePost = np.array([x_center, y_center, 0, 0], dtype=np.float32).reshape(4, 1)
            
            track_states[tracker_id] = TrackState(
                track_id=tracker_id,
                kf=kf,
                position_history=[(x_center, y_center)],
                velocity=np.array([0.0, 0.0]),
                last_visible_pos=(x_center, y_center),
                last_visible_velocity=np.array([0.0, 0.0]),
                age=0,
                time_since_update=0
            )
        else:
            # 既存トラックの更新
            track = track_states[tracker_id]
            
            # 観測で更新
            track.kf.update((x_center, y_center))
            
            # 速度を計算：位置履歴から直接計算（より安定的）
            if len(track.position_history) >= 2:
                prev_pos = track.position_history[-1]
                curr_pos = (x_center, y_center)
                # フレームスキップを考慮して速度計算
                vx = (curr_pos[0] - prev_pos[0]) / FRAME_SKIP
                vy = (curr_pos[1] - prev_pos[1]) / FRAME_SKIP
                track.velocity = np.array([vx, vy])
            else:
                # カルマンフィルタから取得
                state = track.kf.get_state()
                track.velocity = np.array([state[2], state[3]])
            
            # 位置履歴に追加
            track.position_history.append((x_center, y_center))
            if len(track.position_history) > 50:  # 最大50フレーム保持
                track.position_history.pop(0)
            
            # 最後に見えた位置と速度を保存（視野外予測用）
            track.last_visible_pos = (x_center, y_center)
            track.last_visible_velocity = track.velocity.copy()
            
            track.time_since_update = 0
            track.age += 1
    
    # 検出されなかったトラックを予測のみで更新
    for track_id in list(track_states.keys()):
        if track_id not in current_track_ids:
            track = track_states[track_id]
            track.time_since_update += 1
            
            # デバッグ出力（最初のフレームのみ）
            if track.time_since_update == 1:
                print(f"\n[Occlusion] Track {track_id}")
                print(f"  Last visible pos: {track.last_visible_pos}")
                print(f"  Last visible vel: {track.last_visible_velocity}")
            
            # 最後に見えた位置と速度から線形外挿
            if track.last_visible_pos is not None and track.last_visible_velocity is not None:
                # フレームスキップを考慮して予測
                frames_elapsed = track.time_since_update * FRAME_SKIP
                predicted_pos = (
                    track.last_visible_pos[0] + track.last_visible_velocity[0] * frames_elapsed,
                    track.last_visible_pos[1] + track.last_visible_velocity[1] * frames_elapsed
                )
                
                # デバッグ出力（最初のオクルージョン時）
                if track.time_since_update == 1:
                    print(f"  Predicted pos at frame {track.time_since_update}: {predicted_pos}")
                
                track.position_history.append(predicted_pos)
            
            # 古いトラックは削除
            if track.time_since_update > lost_buffer:
                del track_states[track_id]
    
    return detections

def draw_predictions(frame, scale_back=1.0, frame_width=800, frame_height=600):
    """予測された軌跡と速度ベクトルを描画"""
    for track_id, track in track_states.items():
        if len(track.position_history) < 2:
            continue
        
        # position_history をタプルリストに統一
        hist_clean = []
        for pos in track.position_history:
            if isinstance(pos, (tuple, list)):
                x, y = float(pos[0]), float(pos[1])
            else:
                # numpy配列の場合
                x, y = float(pos[0]), float(pos[1])
            
            # 画面範囲内の値のみ保持（異常値フィルタリング）
            if -500 < x < frame_width + 500 and -500 < y < frame_height + 500:
                hist_clean.append((x, y))
        
        if len(hist_clean) < 2:
            continue
        
        # 軌跡を描画
        hist = np.array(hist_clean) * scale_back
        hist = hist.astype(np.int32)
        
        # グラデーション色で軌跡を描画
        for i in range(1, len(hist)):
            alpha = i / len(hist)
            color = (int(255 * alpha), int(100 * (1 - alpha)), 255)
            cv2.line(frame, tuple(hist[i-1]), tuple(hist[i]), color, 1)
        
        # 最新位置にマーカー
        latest = hist[-1]
        cv2.circle(frame, tuple(latest), 5, (0, 255, 255), -1)
        
        # 速度ベクトル（矢印）を描画
        if track.time_since_update < 10:  # 検出されたトラックのみ
            vel = track.velocity * scale_back * 30  # 可視化用にスケール
            arrow_end = (
                int(latest[0] + vel[0]),
                int(latest[1] + vel[1])
            )
            cv2.arrowedLine(frame, tuple(latest), arrow_end, (0, 255, 0), 2, tipLength=0.3)
            
            # 速度の大きさを表示
            speed = np.linalg.norm(track.velocity)
            cv2.putText(frame, f"ID:{track_id} v:{speed:.1f}", 
                       (latest[0] + 10, latest[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # オクルージョン中のトラック
            cv2.putText(frame, f"ID:{track_id} [Occluded]", 
                       (latest[0] + 10, latest[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

while True:
    success, frame_bgr = video_capture.read()
    if not success:
        break
    
    frame_count += 1
    
    # Resize frame for faster processing
    scale = PROCESS_WIDTH / frame_bgr.shape[1]
    process_height = int(frame_bgr.shape[0] * scale)
    frame_resized = cv2.resize(frame_bgr, (PROCESS_WIDTH, process_height))
    
    # Skip frames for speed
    if frame_count % FRAME_SKIP != 0:
        # Use previous detections for skipped frames
        if 'detections' in locals():
            annotated_frame = mask_annotator.annotate(frame_resized.copy(), detections)
            annotated_frame = box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=detections.tracker_id)
        else:
            annotated_frame = frame_resized
    else:
        # Convert to RGB for model
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Run detection
        detections = model.predict(frame_rgb)
        
        # カルマンフィルタでトラッキング更新
        detections = update_kalman_tracks(detections, frame_rgb, frame_resized)
        
        # Annotate frame
        annotated_frame = mask_annotator.annotate(frame_resized.copy(), detections)
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=detections.tracker_id)
    
    # 予測軌跡を描画（スキップフレームでも実行）
    draw_predictions(annotated_frame, scale_back=1.0, frame_width=annotated_frame.shape[1], frame_height=annotated_frame.shape[0])

    # Calculate FPS
    fps_counter += 1
    if (time.time() - fps_start_time) > 1:
        fps_display = fps_counter
        fps_counter = 0
        fps_start_time = time.time()
    
    # Resize for display to fit screen
    display_scale = DISPLAY_WIDTH / annotated_frame.shape[1]
    display_height = int(annotated_frame.shape[0] * display_scale)
    display_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, display_height))
    
    # Display FPS and frame info
    cv2.putText(display_frame, f"FPS: {fps_display}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Frame: {frame_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display with OpenCV (much faster than matplotlib)
    cv2.imshow("RF-DETR + ByteTrack", display_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print("Video capture stopped.")