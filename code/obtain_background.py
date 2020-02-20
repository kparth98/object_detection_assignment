import cv2
import numpy as np
import os
import track_utils

args = track_utils.getArguements()
camera = cv2.VideoCapture(args['video'])

all_frames = []
history=200
i=0
fshape = None
while(camera.isOpened()):
    (grabbed, frame) = camera.read()
    if not grabbed:
        break
    all_frames.append(frame)
    i+=1
    if i>history:
        break

all_frames = np.array(all_frames)
background = np.zeros(all_frames.shape[1:])
background[:,:,0] = np.median(all_frames[:,:,:,0],axis=0)
background[:,:,1] = np.median(all_frames[:,:,:,1],axis=0)
background[:,:,2] = np.median(all_frames[:,:,:,2],axis=0)
cv2.imwrite(args['output'],background)
camera.release()
