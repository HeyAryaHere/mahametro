import cv2
import numpy as np
import os
import torch
import math
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

# Function to perform fall detection
def fall_detection(poses):
    for pose in poses:
        xmin, ymin = (pose[2] - pose[4] / 2), (pose[3] - pose[5] / 2)
        xmax, ymax = (pose[2] + pose[4] / 2), (pose[3] + pose[5] / 2)
        left_shoulder_y = pose[23]
        left_shoulder_x = pose[22]
        right_shoulder_y = pose[26]
        left_body_y = pose[41]
        left_body_x = pose[40]
        right_body_y = pose[44]
        len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
        left_foot_y = pose[53]
        right_foot_y = pose[56]
        dx = int(xmax) - int(xmin)
        dy = int(ymax) - int(ymin)
        difference = dy - dx
        if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
                len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) or (
                right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (
                len_factor / 2) and right_shoulder_y > right_body_y - (len_factor / 2)) \
                or difference < 0:
            return True, (xmin, ymin, xmax, ymax)
    return False, None

# Function to draw bounding box and alarm text for detected falls
def falling_alarm(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255),
                  thickness=5, lineType=cv2.LINE_AA)
    cv2.putText(image, 'Person Fell down', (11, 100), 0, 1, [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)

# Function to process live video stream
def process_live_stream():
    # Load the pre-trained model for pose estimation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('Fall_detection/yolov7-w6-pose.pt', map_location=device)['model'].float().eval()
    if torch.cuda.is_available():
        model = model.half().to(device)

    # Open a live video stream
    cap = cv2.VideoCapture("Fall_detection/fall_dataset/videos/video_1.mp4")  # Use 0 for default webcam, or provide the path to a video file

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame")
            break

        # Perform pose estimation on the frame
        image = letterbox(frame, 960, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        if torch.cuda.is_available():
            image = image.half().to(device)
        with torch.no_grad():
            output, _ = model(image)
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        with torch.no_grad():
            output = output_to_keypoint(output)

        # Check for falls in the current frame
        is_fall, bbox = fall_detection(output)
        if is_fall:
            falling_alarm(frame, bbox)

        # Display the frame
        cv2.imshow('Live Stream', frame)

        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    process_live_stream()
