import sys
import cv2
import supervision as sv
from rfdetr import RFDETRSegSmall  # Use smaller model for speed
import time
import torch

# Initialize tracker and model
tracker = sv.ByteTrack()

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
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise RuntimeError("Failed to open video source")

# Set lower resolution for faster processing
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FPS counter
fps_start_time = time.time()
fps_counter = 0
fps_display = 0

print("Displaying video with OpenCV. Press 'q' to quit.")

while True:
    success, frame_bgr = video_capture.read()
    if not success:
        break

    # Convert to RGB for model
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Run detection and tracking
    detections = model.predict(frame_rgb)
    detections = tracker.update_with_detections(detections)

    # Annotate frame
    annotated_frame = mask_annotator.annotate(frame_bgr.copy(), detections)
    annotated_frame = box_annotator.annotate(annotated_frame, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=detections.tracker_id)

    # Calculate FPS
    fps_counter += 1
    if (time.time() - fps_start_time) > 1:
        fps_display = fps_counter
        fps_counter = 0
        fps_start_time = time.time()
    
    # Display FPS
    cv2.putText(annotated_frame, f"FPS: {fps_display}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display with OpenCV (much faster than matplotlib)
    cv2.imshow("RF-DETR + ByteTrack", annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print("Video capture stopped.")