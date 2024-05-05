import cv2
import numpy as np

video_path = "/mnt/c/Users/prana/Documents/mahametro/Videos/Platform_Edge_Crossing.avi"
cap = cv2.VideoCapture(video_path)
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD_deploy.prototxt", "./MobileNetSSD_deploy.caffemodel")
point1 = (657, 1074)  # Example
point2 = (1027, 209)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Human detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Draw bounding boxes around detected humans
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Threshold for confidence
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Draw the line on the frame
    cv2.line(frame, point1, point2, (0, 255, 0), 2)

    # Check for line crossings
    # Implement your line crossing logic here

    # Display the result
    cv2.imshow('Human and Line Crossing Detection', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
