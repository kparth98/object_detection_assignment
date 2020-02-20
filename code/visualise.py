import cv2
import matplotlib.pyplot as plt
import track_utils
import numpy as np

def main():
    args = track_utils.getArguements()    

    camera = cv2.VideoCapture(args['video'])
    measurements = np.load('measurements.npy')
    bg = cv2.imread('background.png')
    

    step=20
    agg_frame = None
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, 
                                              varThreshold=20, 
                                              detectShadows=False)
    i=0
    while(True):
        (grabbed, frame) = camera.read()
        
        # stop the loop when the video ends
        if not grabbed:
            break        
        
        if i==0:
            fgbg.apply(bg)
        
        frame_fg, bg_mask = track_utils.removeBG(frame,fgbg)        
        if i%step == 0:
            if agg_frame is None:
                agg_frame = np.uint8(np.double(frame)*0.8)
            else:
                temp = np.uint8(np.double(frame_fg)*0.4)
                agg_frame = cv2.add(temp,agg_frame,agg_frame)
        i+=1
    
    for i in range(measurements.shape[0]):
        ball_center = (int(measurements[i,0]),int(measurements[i,1]))
        if i%step==0:
            agg_frame = cv2.circle(agg_frame, ball_center, int(measurements[i,2]), (255, 255, 0), 2)
        agg_frame = cv2.circle(agg_frame, ball_center, 2, (0, 0, 255), -1)
    
    cv2.imwrite(args['output'],agg_frame)
    cv2.imshow('agg frame',agg_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
            
    
if __name__ == '__main__':
    main()
