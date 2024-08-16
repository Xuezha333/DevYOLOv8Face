from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os

weight = "/home/intern/jingjie/Projects/DevYOLOv8Face/runs/pose/Train1_wiseai_100epochs_defArgs/weights/yolov8n-pose.pt"
image = "/mnt2/shared/jingjie/Unprocessed/fd_data/production_202404_liveness_rejected/face/frames/b35d0791-0bb1-479f-8cc3-013852ce8697/face_3.jpg"
image = '/home/intern/jingjie/projects/DevYOLOv8Face/data/test.jpg'

model = YOLO(weight)  # Load an official model
results = model(image)
print(f"Total Speed: {round(results[0].speed['preprocess'] + results[0].speed['inference']+ results[0].speed['postprocess'],4)}ms")
print(results[0].keypoints)
print(results[0].boxes)