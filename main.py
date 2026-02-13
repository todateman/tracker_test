import sys
import cv2
import supervision as sv
from rfdetr import RFDETRSegSmall  # Use smaller model for speed
import time
import torch

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
        
        # Run detection and tracking
        detections = model.predict(frame_rgb)
        detections = tracker.update_with_detections(detections)
        
        # Annotate frame
        annotated_frame = mask_annotator.annotate(frame_resized.copy(), detections)
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=detections.tracker_id)

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