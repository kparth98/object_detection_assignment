import cv2
import numpy as np
import os
import track_utils
import kalman_filter
import time
top_path="/Users/Parth/Downloads/Compressed/Set1"
camera = cv2.VideoCapture(os.path.join(top_path,'14549-14778.avi'))
bg = cv2.imread('background.png')
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=False)

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
    if i==0:
        fgbg.apply(bg)

    bg_mask = fgbg.apply(frame)
    bg_mask = cv2.dilate(bg_mask, np.ones((3, 3)))
    bg_mask = cv2.erode(bg_mask, np.ones((5, 5)))
    frame2 = cv2.bitwise_and(frame, frame, mask=bg_mask)
    
    cv2.imshow('frame2',frame2)
    
    if roi is None:
        roi,roi_mask = track_utils.getROIvid(orig_frame, bg_mask,'input ball')
        b,g,r = cv2.split(roi)
        b_mean = np.median(b[roi_mask>0])
        g_mean = np.median(g[roi_mask>0])
        r_mean = np.median(r[roi_mask>0])
                
        b_std = 1.5*np.std(b[roi_mask>0])
        g_std = 1.5*np.std(g[roi_mask>0])
        r_std = 1.5*np.std(r[roi_mask>0])

#        hist = track_utils.getHist(roi,roi_mask)
#        with open('hist2.npy', 'wb') as f:
#            np.save(f, hist)    
#        print(hist)
        limits = [(min(int(b_mean+b_std),255), min(int(g_mean+g_std),255), min(int(r_mean+r_std),255)), 
              (int(b_mean-b_std), int(g_mean-g_std), int(r_mean-r_std))]
    
    if limits is not None:
#        ball_center, cnt = track_utils.detectBallHB(frame2, hist)
        ball_center, radius = track_utils.detectBallThresh_RGB(frame2, limits)
        radius = int(radius)
        # get initial estimate of position and velocity of ball
        if i==1:
            cov = 10*np.eye(4)
            temp = np.double(ball_center)
            init_mean = np.array([temp[0],temp[1],
                                  temp[0]-measurements[-1][0],
                                  temp[1]-measurements[-1][1]])
            kf = kalman_filter.getKF(init_mean,cov)
            filtered_means = init_mean
            f_center = ball_center

        elif i>1:
            if radius is not None:
                temp = np.double(ball_center)
                new_meas = np.array([temp[0],temp[1],
                                     temp[0]-measurements[-1][0],
                                     temp[1]-measurements[-1][1]])
                filtered_means,cov = kalman_filter.updateKF(kf,filtered_means,cov,new_meas)
            else:
                filtered_means,cov = kalman_filter.updateKF(kf,filtered_means,cov)
                                
            f_center = np.uint32(filtered_means[0:2])
            
        if radius is not None: 
            cv2.circle(frame, ball_center, radius, (255, 255, 0), 2)
            cv2.circle(frame, ball_center, 2, (0, 0, 255), -1)
        if i>0:
            cv2.circle(frame, (f_center[0],f_center[1]), 2, (0, 255,0), -1)

    if i<1:
        measurements.append(np.double(ball_center))
    else:
        measurements.append(np.double(f_center))

    cv2.imshow('frame',frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    i+=1

camera.release()
measurements = np.array(measurements)
with open('measurements.npy', 'wb') as f:
    np.save(f, measurements)