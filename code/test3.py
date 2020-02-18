import cv2
import numpy as np
import os
import track_utils
import kalman_filter

top_path="/Users/Parth/Downloads/Compressed/Set5"
camera = cv2.VideoCapture(os.path.join(top_path,'69797-69913.avi'))
bg = cv2.imread('background.png')
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20, detectShadows=False)

limits = None
measurements_kf = []
measurements = []
limits = np.load('limits.npy')
i=0
while(camera.isOpened()):
    (grabbed, frame) = camera.read()
    
    if not grabbed:
        break
    orig_frame = frame.copy()
    if i==0:
        fgbg.apply(bg)

    frame2, bg_mask = track_utils.removeBG(frame,fgbg)
    
    cv2.imshow('frame2',frame2)
    
    if limits is None:
        roi,roi_mask = track_utils.getROIvid(orig_frame, bg_mask,'input ball')
        limits = track_utils.getLimits_RGB(roi,roi_mask)
        ball_center, radius = track_utils.detectBallThresh_RGB(frame2, limits)
        radius = int(radius)
    else:
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

    measurements.append(np.double(ball_center))
    if i<1:
        measurements_kf.append(np.double(ball_center))    
    else:
        measurements_kf.append(np.double(f_center))

    cv2.imshow('frame',frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    i+=1

camera.release()

measurements = np.array(measurements)
with open('measurements.npy', 'wb') as f:
    np.save(f, measurements)

measurements_kf = np.array(measurements_kf)
with open('measurements_kf.npy', 'wb') as f:
    np.save(f, measurements_kf)

#with open('limits.npy', 'wb') as f:
#    np.save(f, limits)