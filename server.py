#!/usr/bin/env python3

import cv2
from flask import Flask, Response
from model import CSRNet
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

app = Flask(__name__)

model = CSRNet()
checpoint = torch.load('./weights.pth', map_location='cpu')
model.load_state_dict(checpoint)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def generate_frames():
    video_path = "/mnt/c/Users/prana/Documents/mahametro/Videos/Crowding Near Ticketing line.avi"
    cap = cv2.VideoCapture(video_path)
    while True:
        success, frame = cap.read()
        if not success:
            print("Error reading frame from video")
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error encoding frame as JPEG")
            break
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')


def count():
    video_path = "/mnt/c/Users/prana/Documents/mahametro/Videos/Crowding Near Ticketing line.avi"
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        # Process every 20th frame
        if frame_count % 20 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            img = transform(image).unsqueeze(0)
            output = model(img)
            predicted_count = int(output.detach().cpu().sum().numpy())
            print("Predicted count for frame {}: {}".format(frame_count, predicted_count))
    cap.release()


@app.route('/')
def index():
    return "Live Video Streaming"

@app.route('/video_feed')
def video_feed():
    try:
        count()
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print("Error:", e)
        return "An error occurred while streaming video"

if __name__ == '__main__':
    app.run(debug=True)
