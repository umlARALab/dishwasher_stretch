#!/usr/bin/env python3

# https://moveit.ai/ros/moveit/events/2021/10/29/rosworld-moveit-workshop.html
from ultralytics import YOLO

# loads a small pretrained yolo model
model = YOLO("yolov8n.pt")
imageName = 'cat_dog.jpg'

results = model.predict(imageName)
result = results[0]

print('Objects Detected: ' + str(len(result.boxes)))

for box in result.boxes:
	print("Object type:", box.cls[0].item())
	print("Coordinates:", box.xyxy[0].tolist())
	print("Probability:", box.conf[0].item())
	print("--------------------------------------")

