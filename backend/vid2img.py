

import cv2
import os

def video_to_images(video_path, output_folder,frame_interval ,title):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error Could not open video")
        return

    frame_count  = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break


        if frame_count%frame_interval==0:
            image_path = os.path.join(output_folder, f"frame_{title}_{frame_count}.jpg")
            cv2.imwrite(image_path, frame)


        frame_count += 1

    video.release()
    print("Frames Extracted")

video_to_images("/mnt/c/Users/prana/Documents/mahametro/Videos/FalllOnEscalator_1.avi", "/mnt/c/Users/prana/Documents/mahametro/Fall_on_escalator",50, "video1")
video_to_images("/mnt/c/Users/prana/Documents/mahametro/Videos/FalllOnEscalator_2.avi", "/mnt/c/Users/prana/Documents/mahametro/Fall_on_escalator",50, "video2")
video_to_images("/mnt/c/Users/prana/Documents/mahametro/Videos/FalllOnEscalator_False_Scenario.avi", "/mnt/c/Users/prana/Documents/mahametro/Fall_on_escalator",50, "video3")
video_to_images("/mnt/c/Users/prana/Documents/mahametro/Videos/FallOnEscalator.avi", "/mnt/c/Users/prana/Documents/mahametro/Fall_on_escalator",50, "video4")
video_to_images("/mnt/c/Users/prana/Documents/mahametro/Videos/FallOnEscalator_1.avi", "/mnt/c/Users/prana/Documents/mahametro/Fall_on_escalator",50, "video5")
video_to_images("/mnt/c/Users/prana/Documents/mahametro/Videos/FallOnEscalator_2.avi", "/mnt/c/Users/prana/Documents/mahametro/Fall_on_escalator",50, "video6")
video_to_images("/mnt/c/Users/prana/Documents/mahametro/Videos/FallOnEscalator_3.avi", "/mnt/c/Users/prana/Documents/mahametro/Fall_on_escalator",50, "video7")
video_to_images("/mnt/c/Users/prana/Documents/mahametro/Videos/FallOnEscalator_4.avi", "/mnt/c/Users/prana/Documents/mahametro/Fall_on_escalator",50, "video8")
video_to_images("/mnt/c/Users/prana/Documents/mahametro/Videos/FallOnEscalator_5.avi", "/mnt/c/Users/prana/Documents/mahametro/Fall_on_escalator",50, "video9")
video_to_images("/mnt/c/Users/prana/Documents/mahametro/Videos/FallOnEscalator_6.avi", "/mnt/c/Users/prana/Documents/mahametro/Fall_on_escalator",50, "video10")
