import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob
import torch
from time import time

def summarisevideo(videofile, duration):
    #accessing the video
    video_v = cv2.VideoCapture(videofile)
    
    #summarised.mp4 only contains frames with big changes
    video_w = int(video_v.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(video_v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    summarised = cv2.VideoWriter('summarised.mp4', fourcc, 24, (video_w, video_h))
    
    u = 0 #unique frame counter
    thres = 20 #threshold for frame similarity
    
    #read and set first frame of video as previous_f
    status_v, frame1_v = video_v.read()
    previous_f = frame1_v
    
    
    while True:
        status_v, current_f = video_v.read()
        
        #if status_v is true, program is reading frames from video
        if status_v is True:
            
            frame_diff = np.sum(np.absolute(current_f - previous_f)) / np.size(current_f) #difference formula
            if(frame_diff > thres):
                cv2.imwrite("D:/Jere Stuff/Code/GitHub/vsod/frameout/frame%d.jpg" %u, current_f) #write jpegs
                summarised.write(current_f) #write into .mp4
                previous_f = current_f
                u = u + 1 #add tally unique frames 
                
            else:
                previous_f = current_f
                
            cv2.imshow('Unique Frame', current_f)
        
        #stop program by pressing x
        if cv2.waitKey(1) & 0xff == ord('x'): #0xff because ord('x') may be different value due to numlock
            break
        
    print("total no. of unique frames: ", u)
    
    video_v.release()
    summarised.release()
    cv2.destroyAllWindows()
    
    #store path of all saved frame jpegs as a string
    frameslist = glob.glob("D:/Jere Stuff/Code/GitHub/vsod/frameout/*.jpg")
    return frameslist

class detection:
    
    #prereqs, loading yolov5 model section
    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' #rearrange
        print("\n\n device used: ", self.device)
        
    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained = True)
        return model
    
    #labelling and object detection section
    def getscore(self, od_frame):
        self.model.to(self.device)
        od_frame = [od_frame]
        od_score = self.model(od_frame)
        labels, coordinates = od_score.xyxyn[0][:, -1], od_score.xyxyn[0][:, :-1]
            
        return labels,coordinates
        
    def classname(self, x):
        return self.classes[int(x)] 
            
    def drawbox(self, od_score, od_frame):
        labels, coordinates = od_score
        n = len(labels)
        od_w, od_h = od_frame.shape[1], od_frame.shape[0]
            
        for i in range(n):
            row = coordinates[i]
                
            if row[4] >= 0.2:
                    
                w1 = int(row[0] * od_w) 
                h1 = int(row[1] * od_h) 
                w2 = int(row[2] * od_w) 
                h2 = int(row[3] * od_h) 
                    
            bgr = (0, 255, 0)
            cv2.rectangle(od_frame, (w1, h1), (w2, h2), bgr, 2)
            cv2.putText(od_frame, self.classname(labels[i]), (w1, h1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                    
        return od_frame
        
    #will get called when instance of object created
    def __call__(self):
        #accessing summarised video
        od_video = cv2.VideoCapture("summarised.mp4")
        
        #writing new video with object detection
        od_w = int(od_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        od_h = int(od_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        od_summarised = cv2.VideoWriter("od_summarised.mp4", fourcc, 24, (od_w, od_h))
        
        while True:
            #start = time() #start time
            status_od, od_frame = od_video.read()
            if not status_od:
                break
            
            od_score = self.getscore(od_frame)
            od_frame = self.drawbox(od_score, od_frame)
            #end = time() end time
            
            #fps = 1/np.round(end - start, 3)
            #print(f"fps: {fps}")
            od_summarised.write(od_frame)
  
#main
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #used in summarisevideo() and detect(), MPEG-4 codec just cause


#passing cctv.mp4 into summarisevideo()
videofile = "D:/Jere Stuff/Code/GitHub/vsod/cctv.mp4"
frameslist = summarisevideo(videofile, 10)

#visualise the first 20 frames
fig = plt.figure()
for i, imf in enumerate(frameslist[:20]):

    #display 20 images. 4 x 5
    fig.add_subplot(4, 5, i+1) #framesList array starts with [0, 1, 2, ...], 0 cant be used as an index for the subplot
    im = cv2.imread(imf)
    plt.imshow(im)
    plt.axis('off')
    
plt.show()

#pass summarised.mp4 into detect()
bruh = detection()
bruh()
