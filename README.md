# Overview
VSOD's python code uses OpenCV to summarise long, redundant, cctv/surveillance camera or similar footage, and uses YOLOv5 to conduct object detection on the summarised video. We will be using cctv.mp4 as an example.

### How does it work?
The code can be split into 2 sections:
* Video summarisation using summarisevideo(), which mainly utilises the cv2 library
* Object detection via the class detection, which mainly utilises the torch library to access YOLOv5 (yolov5s.yaml)

**summarisevideo() -**
This function takes an input .mp4 video, in our case cctv.mp4, and starts reading its frames one by one. It calculates the difference between the first frame of the video with the next, and the next frame with its next, and so on via image subtraction.

Frames that are similar have a small difference, frames where big changes happen (unique) have a larger difference. If this difference value exceeds a user-set threshold, the frame is saved as a .jpeg and written into the summarised video. This ensures redundant parts of the footage are cut out.

**class detection -**
This class contains the various functions needed to detect objects in the summarised video, that utilises the torch library and the Yolov5s model we will be using. It also draws the rectangles and labels over the detected objects. This section is based on Nicolai Nielsen's code so go check him out for a more in depth explanation: https://github.com/niconielsen32/ComputerVision/blob/master/yoloCustomObjectDetection.py

