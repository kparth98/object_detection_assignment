import cv2
import numpy as np
import os
import track_utils
import kalman_filter

top_path="/Users/Parth/Downloads/Compressed/Set1"

camera = cv2.VideoCapture(os.path.join(top_path,'14549-14778.avi'))

bg = cv2.imread('background.png')
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=False)
flag = False
roi = None
limits = None
measurements = []
#hist = np.load('hist.npy')
i=0
while(camera.isOpened()):
    (grabbed, frame) = camera.read()
    
    if not grabbed:
        break
#    frame = track_utils.resize(frame, width=600)
    orig_frame = frame.copy()
    if i%20==0:
        fgbg.apply(bg)
    bg_mask = fgbg.apply(frame)
    bg_mask = cv2.dilate(bg_mask, np.ones((3, 3)))
    bg_mask = cv2.erode(bg_mask, np.ones((5, 5)))

#    frame2 = track_utils.removeBG(orig_frame.copy(), fgbg)

    frame2 = cv2.bitwise_and(frame, frame, mask=bg_mask)
    cv2.imshow('frame2',frame2)
    if roi is None:
        roi,roi_mask = track_utils.getROIvid(orig_frame, bg_mask,'input ball')
        hist = track_utils.getHist(roi,roi_mask)
#        print(hist)
#        limits = track_utils.getLimits(roi,roi_mask)
    if hist is not None:
        ball_center, cnt = track_utils.detectBallHB(frame2, hist)
#        ball_center, cnt = track_utils.detectBallThresh(frame2, limits)
#        if cnt is not None:
        if i==0:
            cov = 10*np.eye(4)
            kf = kalman_filter.getKF(np.array([ball_center[0],ball_center[1],0,0]),cov)
            filtered_means = np.array([ball_center[0],ball_center[1],0,0])
            f_center = ball_center

        else:
            if cnt is not None:
                new_meas = np.array([ball_center[0],
                            ball_center[1],
                            ball_center[0]-measurements[-1][0],
                            ball_center[1]-measurements[-1][1]])
            else:
                new_meas = None
            if new_meas is None:
                filtered_means,cov = kalman_filter.updateKF(kf,filtered_means,cov)
            else:
                filtered_means,cov = kalman_filter.updateKF(kf,filtered_means,new_meas)
            print(filtered_means)
            f_center = np.uint32(filtered_means[0:2])
#        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if cnt is not None: 
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 255, 0), 2)
        cv2.circle(frame, ball_center, 2, (0, 0, 255), -1)
        if i>0:
            cv2.circle(frame, (f_center[0],f_center[1]), 2, (0, 255,0), -1)
            # trackBall()
    cv2.imshow('frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    measurements.append(f_center)
    i+=1
camera.release()
measurements = np.array(measurements)
with open('measurements.npy', 'wb') as f:
    np.save(f, measurements)