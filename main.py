import cv2
import supervision as sv
from rfdetr import RFDETRSegMedium
import matplotlib
matplotlib.use('Qt5Agg')  # Set interactive backend
import matplotlib.pyplot as plt

tracker = sv.ByteTrack()
model = RFDETRSegMedium()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
mask_annotator = sv.MaskAnnotator()

video_capture = cv2.VideoCapture("/dev/video4")
if not video_capture.isOpened():
    raise RuntimeError("Failed to open video source")

# Setup matplotlib figure
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 8))
img_display = None

print("Displaying video. Close the matplotlib window to stop.")

while True:
    success, frame_bgr = video_capture.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = model.predict(frame_rgb)
    detections = tracker.update_with_detections(detections)

    annotated_frame = mask_annotator.annotate(frame_bgr.copy(), detections)
    annotated_frame = box_annotator.annotate(annotated_frame, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=detections.tracker_id)

    # Convert BGR to RGB for matplotlib
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    # Display with matplotlib
    if img_display is None:
        img_display = ax.imshow(annotated_frame_rgb)
        ax.set_title("RF-DETR + ByteTrack")
        ax.axis('off')
    else:
        img_display.set_data(annotated_frame_rgb)
    
    plt.pause(0.001)
    
    if not plt.fignum_exists(fig.number):
        break

video_capture.release()
print("Video capture stopped.")