import cv2
import numpy as np
import os

top_path="/Users/Parth/Downloads/Compressed/Set1"

camera = cv2.VideoCapture(os.path.join(top_path,'0016-0298.avi'))

history=200
i=0
all_frames=[]
while(camera.isOpened()):
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
    all_frames.append(frame)
    i+=1
    if i>history:
        break

all_frames = np.array(all_frames)
background = np.zeros(frame.shape)
background[:,:,0] = np.median(all_frames[:,:,:,0],axis=0)
background[:,:,1] = np.median(all_frames[:,:,:,1],axis=0)
background[:,:,2] = np.median(all_frames[:,:,:,2],axis=0)
cv2.imwrite('background.png',background)
camera.release()