import cv2
import numpy as np
import os
import track_utils
import kalman_filter


def main():
    ## Initialisation
    args = track_utils.getArguements()    

    if args['background'] is not None:
        bg = cv2.imread(args['background'])
    else:
        bg = cv2.imread('background.png')

    camera = cv2.VideoCapture(args['video'])

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, 
                                              varThreshold=20, 
                                              detectShadows=False)

    limits = None
    filtered_center = None
    measurements_kf = []
    measurements = []
    
    if args['limits'] is not None:
        limits = np.load(args['limits'])
        
    ## loop over every frame
    i=0
    while(True):
        # grab frame
        (grabbed, frame) = camera.read()

        # stop the loop when the video ends
        if not grabbed:
            break
        
        # this is required, otherwise first frame is considered complete background        
        if i==0:        
            fgbg.apply(bg)
        
        # remove static background
        frame_fg, bg_mask = track_utils.removeBG(frame,fgbg)

        # get the RGB thresholding limits for the ball
        if limits is None:
            roi,roi_mask = track_utils.getROIvid(frame, bg_mask,'input ball')
            limits = track_utils.getLimits_RGB(roi,roi_mask)
            ball_center, radius = track_utils.detectBallThresh_RGB(frame_fg, 
                                                                   limits,
                                                                   None)
        else:
            # detect ball center and radius of the ball
            ball_center, radius = track_utils.detectBallThresh_RGB(frame_fg, 
                                                                   limits, 
                                                                   None)

            if i==1:
                # get initial estimate of position and velocity of ball
                # state of KF = [x, y, v_x, v_y]
                cov = 10*np.eye(4)
                temp = np.double(ball_center)
                init_mean = np.array([temp[0],temp[1],
                                      temp[0]-measurements[-1][0],
                                      temp[1]-measurements[-1][1]])

                # initialise Kalman Filter
                kf = track_utils.getKF(init_mean,cov)
                filtered_means = init_mean
                filtered_center = ball_center
    
            elif i>1:
                if radius is not None:
                    # if ball detected, update KF with new measurements
                    temp = np.double(ball_center)
                    new_meas = np.array([temp[0],temp[1],
                                         temp[0]-measurements[-1][0],
                                         temp[1]-measurements[-1][1]])

                    filtered_means,cov = track_utils.updateKF(kf,filtered_means,cov,new_meas)
                    print(filtered_means)
                else:
                    # Else, keep using previous estimates of KF 
                    filtered_means,cov = track_utils.updateKF(kf,filtered_means,cov)
                                    
                filtered_center = np.uint32(filtered_means[0:2])
            
            # mark the ball, ball center and KF filtered center on the frame
            if args['visualise']:
                if radius is not None: 
                    cv2.circle(frame, ball_center, int(radius), (255, 255, 0), 2)
                    cv2.circle(frame, ball_center, 2, (0, 0, 255), -1)
                if i>0:
                    cv2.circle(frame, (filtered_center[0],filtered_center[1]), 2, (0, 255,0), -1)
    
        
        measurements.append(np.double(ball_center))
        if i==0:
            measurements_kf.append(np.double(ball_center))    
        else:
            measurements_kf.append(np.double(filtered_center))
        
        if i<2:
            i+=1
        # show the frame
        if args['visualise']:
            cv2.imshow('frame',frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    camera.release()
    
    if args['save']:
        measurements = np.array(measurements)
        with open('measurements.npy', 'wb') as f:
            np.save(f, measurements)
        
        measurements_kf = np.array(measurements_kf)
        with open('measurements_kf.npy', 'wb') as f:
            np.save(f, measurements_kf)
        with open('limits.npy', 'wb') as f:
            np.save(f, limits)
    

if __name__ == '__main__':
    main()
