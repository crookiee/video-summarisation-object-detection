import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob
import torch

def summarisevideo(videofile, duration):
    
    #accessing the video
    video_v = cv2.VideoCapture(videofile)
    
    #summarised.mp4 only contains frames with big changes
    video_w = int(video_v.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(video_v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    summarised = cv2.VideoWriter("D:/Jere Stuff/Code/GitHub/vsod/odresult/summarised.mp4", fourcc, duration, (video_w, video_h))
    
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
                
            cv2.imshow("super fast unique frame slideshow", current_f)
        
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
    #initialize class for bruh object
    def __init__(self):
        self.model = self.load_model() #loads model we want to use
        self.classes = self.model.names #get labels for object detection
        
        if torch.cuda.is_available():
            self.device = 'cuda' 
            
        else:
            self.device = 'cpu' 
        
        print("\ndevice used: ", self.device)
        
    #load the model (tried using local path could not access, using torch.hub instead)
    def load_model(self):
        model = torch.hub.load("ultralytics/yolov5", 'yolov5s', pretrained = True)
        return model
    
    #labelling and object detection section
    def getstats(self, od_frame):
        
        self.model.to(self.device) #specify using cpu or gpu for the model
        od_frame = [od_frame] #format to pass model
        od_stats = self.model(od_frame)
        
        #all boundary boxes (as coordinates) and labels stored in od_stats
        labels, coordinates = od_stats.xyxyn[0][:, -1], od_stats.xyxyn[0][:, :-1]
            
        return labels,coordinates
    
    #for a label value, return the label as string
    def classname(self, x):
        return self.classes[int(x)] #label index

    def getclassname(self):
        return self.classes
    
    #take frame and od_stats, and draws the boxes and labels them
    def drawbox(self, od_stats, od_frame):
        
        labels, coordinates = od_stats
        n = len(labels) #n = number of a specific label eg. 10 people
        od_w, od_h = od_frame.shape[1], od_frame.shape[0]
            
        for i in range(n):
            row = coordinates[i]
                
            if row[4] >= 0.2:
                
                #widths and heights of a rectangle, od_w and od_h from our original frame
                w1 = int(row[0] * od_w) 
                h1 = int(row[1] * od_h) 
                w2 = int(row[2] * od_w) 
                h2 = int(row[3] * od_h) 
                    
            bgr = (0, 255, 0)
            cv2.rectangle(od_frame, (w1, h1), (w2, h2), bgr, 2)
            cv2.putText(od_frame, self.classname(labels[i]), (w1, h1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                    
        return od_frame #returns frame with the labels and boxes drawn
        
    #will get called when instance of object created
    def __call__(self):
        
        print("processing...")
        #accessing summarised video
        od_video = cv2.VideoCapture("D:/Jere Stuff/Code/GitHub/vsod/odresult/summarised.mp4")
        
        #writing new video with object detection
        od_w = int(od_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        od_h = int(od_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        od_summarised = cv2.VideoWriter( "D:/Jere Stuff/Code/GitHub/vsod/odresult/od_summarised.mp4", fourcc, duration, (od_w, od_h))
        
        while True:
            status_od, od_frame = od_video.read()
            if not status_od:
                break
            
            od_stats = self.getstats(od_frame)
            od_frame = self.drawbox(od_stats, od_frame)
            
            print("...")
            od_summarised.write(od_frame)
        
        print("processing done")
        
#main
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #used in summarisevideo() and detect(), MPEG-4 codec just cause

#passing cctv.mp4 into summarisevideo()
videofile = "D:/Jere Stuff/Code/GitHub/vsod/cctv.mp4"
duration = 24 #frames per second for result video
frameslist = summarisevideo(videofile, duration)

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

#list of detected objects
detected_objects = bruh.getclassname()
print("\nlist of detected objects: \n")
        
for i in (detected_objects):
    print(detected_objects[i], end = ',')
    

