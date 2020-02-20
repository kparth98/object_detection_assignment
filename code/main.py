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

    fgbg = cv2.createBackgroundSubtractorMOG2(history=1000,
                                              varThreshold=10,
                                              detectShadows=False)

    limits = None
    filtered_center = None
    measurements = []
    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))

    if args['limits'] is not None:
        limits = np.load(args['limits'])

    if args['save']:
        out = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (frame_width,frame_height))
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
            detected_center, radius = track_utils.detectBallThresh_RGB(frame_fg,
                                                                   limits,
                                                                   None)
        else:
            # detect ball center and radius of the ball
            if i<20: # wait for background subtractor to learn the background
                detected_center, radius = track_utils.detectBallThresh_RGB(frame_fg,
                                                                   limits,
                                                                   None)
            else:
                detected_center, radius = track_utils.detectBallThresh_RGB(frame_fg,
                                                                   limits,
                                                                   ball_center)

            if i==1:
                # get initial estimate of position and velocity of ball
                # state of KF = [x, y, v_x, v_y]
                cov = 10*np.eye(4)
                temp = np.double(detected_center)
                init_mean = np.array([temp[0],temp[1],
                                      temp[0]-measurements[-1][0],
                                      temp[1]-measurements[-1][1]])

                # initialise Kalman Filter
                kf = track_utils.getKF(init_mean,cov)
                filtered_means = init_mean
                filtered_center = temp

            elif i>1:
                if radius is not None:
                    # if ball detected, update KF with new measurements
                    temp = np.double(detected_center)
                    new_meas = np.array([temp[0],temp[1],
                                         temp[0]-measurements[-1][0],
                                         temp[1]-measurements[-1][1]])

                    filtered_means,cov = track_utils.updateKF(kf,filtered_means,cov,new_meas)
                else:
                    # Else, keep using previous estimates of KF
                    filtered_means,cov = track_utils.updateKF(kf,filtered_means,cov)

                filtered_center = filtered_means[0:2]

            # mark the ball, ball center and KF filtered center on the frame


        if detected_center is not None and filtered_center is not None:
            ball_center = np.uint32((np.double(detected_center)+np.double(filtered_center))*0.5)
        elif filtered_center is None:
            ball_center = np.uint32(detected_center)
        elif detected_center is None:
            ball_center = np.uint32(filtered_center)

        if args['visualise'] or args['output'] is not None:
            if track_utils.checkInFrame(ball_center,frame.shape):
                if radius is not None:
                    cv2.circle(frame, (ball_center[0],ball_center[1]), int(radius), (255, 255, 0), 2)
                cv2.circle(frame, (ball_center[0],ball_center[1]), 4, (0, 0, 255), -1)


        print(str(ball_center[0])+","+str(ball_center[1]))
        measurements.append(np.append(np.double(ball_center),radius))

        i+=1
        # show the frame
        if args['visualise']:
            cv2.imshow('frame',cv2.resize(frame,(int(0.7*frame.shape[1]),int(0.7*frame.shape[0]))))
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        if args['save']:
            out.write(frame)

    camera.release()
    cv2.destroyAllWindows()
    if args['save']:
        out.release()
    if args['save']:
        measurements = np.array(measurements)
        with open('measurements.npy', 'wb') as f:
            np.save(f, measurements)

        with open('limits.npy', 'wb') as f:
            np.save(f, limits)


if __name__ == '__main__':
    main()
