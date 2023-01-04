# Overview
VSOD's python code uses OpenCV to summarise long, redundant, cctv/surveillance camera or similar footage, and uses YOLOv5 to conduct object detection on the summarised video. We will be using cctv.mp4 as an example.

### How does it work?
The code can be split into 2 sections:
* Video summarisation using summarisevideo(), which mainly utilises the cv2 library
* Object detection via the detect(), which mainly utilises the torch library to access YOLOv5 (yolov5s.yaml)

**summarisevideo() -**
This function takes an input .mp4 video, in our case cctv.mp4, and starts reading its frames one by one. It compares the difference between the first frame of the video with the next, and the next frame with its next, and so on. 

Frames that are similar have a small difference, frames where big changes happen (unique) have a larger difference. If this difference value exceeds a user-set threshold, the frame is saved as a .jpeg and written into the summarised video. This ensures redundant parts of the footage are cut out.

**class detect -**
This class contains the various sections needed to detect objects in the summarised video

