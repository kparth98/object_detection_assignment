import cv2
import numpy as np
import os
import track_utils

top_path="/Users/Parth/Downloads/Compressed/Set1"

camera = cv2.VideoCapture(os.path.join(top_path,'0016-0298.avi'))

bg = cv2.imread('background.png')
bg_gray = cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)
bg_gray=np.double(bg_gray)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=False)
i=0
while(camera.isOpened()):
    (grabbed, frame) = camera.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_gray=np.double(frame_gray)
    if i%20==0:
        fgbg.apply(bg)
        
    if not grabbed:
        break
    bg_mask = fgbg.apply(frame)
    frame2 = cv2.bitwise_and(frame, frame, mask=bg_mask)
    cv2.imshow('frame2',frame2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    i+=1
camera.release()