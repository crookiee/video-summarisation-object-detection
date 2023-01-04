# Overview
VSOD's python code uses OpenCV to summarise long, redundant, cctv/surveillance camera or similar footage, and uses YOLOv5 to conduct object detection on the summarised video. We will be using cctv.mp4 as an example.

### How does it work?
The code can be split into 2 sections:
* Video summarisation using summarisevideo(), which mainly utilises the cv2 library
* Object detection via the detect(), which mainly utilises the torch library to access YOLOv5 (yolov5s.yaml)

**summarisevideo()**
-This function takes an input .mp4 video, in our case cctv.mp4, and starts reading its frames one by one. It compares the difference between the first frame of the video with the next, and the next frame with its next, and so on. It calculates the difference using image subtraction:

