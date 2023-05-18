import torch
import cv2
from PIL import Image
from torchvision.transforms import functional as F

# Load the YOLOv5 model with custom weights
model = torch.hub.load('yolov5', 'custom', path='last.pt', source='local')
model.eval()

# Set up the webcam video capture
video = cv2.VideoCapture(0)  # Use 0 for the default webcam, or specify the index of the desired webcam

# Process each frame of the webcam video
while True:
    # Read a frame from the video
    ret, frame = video.read()

    # Convert the frame to PIL Image format
    pil_image = Image.fromarray(frame)

    # Perform object detection
    detections = model(pil_image)

    # Process the detections
    for detection in detections.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection.tolist()

        if conf > 0.5:
            label = model.names[int(cls)]

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f'{label}: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()
