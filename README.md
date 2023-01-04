# Overview
VSOD's python code uses OpenCV to summarise long, redundant, cctv/surveillance camera or similar footage, and uses YOLOv5 to conduct object detection on the summarised video

## How does it work?
The code can be split into 2 sections
* Video summarisation using summarisevideo(), which mainly utilises the cv2 library
* Object detection via the detect(), which mainly utilises the torch library to access YOLOv5 (yolov5s.yaml)
