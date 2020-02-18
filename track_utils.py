import numpy as np
import argparse
import cv2
import math
from collections import deque
import matplotlib.pyplot as plt
img = None
orig = None
bbox = None
roi2, roi2_init = None,None

# kernel = np.array([[0, 0, 1, 1, 0, 0],
#                    [0, 1, 1, 1, 1, 0],
#                    [1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1],
#                    [0, 1, 1, 1, 1, 0],
#                    [0, 0, 1, 1, 0, 0]],dtype=np.uint8)

kernel = np.ones((5,5))
ix,iy=0,0
draw = False
rad_thresh = 50

def getArguements():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    args = vars(ap.parse_args())
    return args

def resize(img,width=400.0):
    r = float(width) / img.shape[0]
    dim = (int(img.shape[1] * r), int(width))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

def selectROI(event, x, y, flag, param):
    global img, draw, orig, bbox, ix,iy
    if event == cv2.EVENT_LBUTTONDOWN:
        ix = x
        iy = y
        draw = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if draw:
            img = cv2.rectangle(orig.copy(), (ix, iy), (x, y), (255, 0, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        if draw:
            x1 = max(x, ix)
            y1 = max(y, iy)
            ix = min(x, ix)
            iy = min(y, iy)
            bbox = np.array([[ix,iy],[x1,y1]])
        draw = False

def getROIvid(frame,mask, winName = 'input'):
    global img, orig, bbox
    bbox=None
    img = frame.copy()
    orig = frame.copy()
    cv2.namedWindow(winName)
    cv2.setMouseCallback(winName, selectROI)
    while True:
        cv2.imshow(winName, img)
        if bbox is not None:
            cv2.destroyWindow(winName)
            roi = orig[bbox[0,1]:bbox[1,1],bbox[0,0]:bbox[1,0],:]
            roi_mask = mask[bbox[0,1]:bbox[1,1],bbox[0,0]:bbox[1,0]]
            return roi,roi_mask

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cv2.destroyWindow(winName)
            break

    return None, None

def getROIext(image,winName = 'input'):
    global img, orig, roi2, roi2_init
    img = image.copy()
    orig = image.copy()
    cv2.namedWindow(winName)
    cv2.setMouseCallback(winName, selectROI)
    while True:
        cv2.imshow(winName, img)
        if roi is not None:
            cv2.destroyWindow(winName)
            return roi

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cv2.destroyWindow(winName)
            break

    return roi


def getLimits(roi,roi_mask):
    limits = None
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(roi)
    print(np.mean(h[roi_mask>0]))
    print(np.std(h[roi_mask>0]))
    limits = [(int(np.amax(h[roi_mask>0])), int(np.amax(s[roi_mask>0])), int(np.amax(v[roi_mask>0]))), 
              (int(np.amin(h[h>0])), int(np.amin(s[roi_mask>0])), int(np.amin(v[roi_mask>0])))]
    return limits

def applyMorphTransforms(mask):
    global kernel
    lower = 100
    upper = 255
    #mask = cv2.inRange(mask, lower, upper)
    # mask = cv2.GaussianBlur(mask, (11, 11), 5)
    # mask = cv2.inRange(mask, lower, upper)
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, kernel)

    return mask

def applyMorphTransforms2(backProj):
    global kernel
    lower = 50
    upper = 255
    mask = cv2.inRange(backProj, lower, upper)
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, np.ones((3, 3)))
    mask = cv2.GaussianBlur(mask, (11, 11), 5)
    mask = cv2.inRange(mask, lower, upper)
    return mask



def detectBallThresh(frame,limits):
    global rad_thresh
    upper = limits[0]
    lower = limits[1]
    center = None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = applyMorphTransforms(mask)
    cv2.imshow('mask_threh', mask)


    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    flag = False
    i=0
    if len(cnts) > 0:
        for i in range(len(cnts)):
            (center, radius) = cv2.minEnclosingCircle(cnts[i])
            if radius < rad_thresh and radius > 5:
                flag = True
                break
        if not flag:
            return None, None
        center = np.uint32(center)
#        M = cv2.moments(cnts[i])
#        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return (center[0],center[1]), cnts[i]
    else:
        return None, None

def detectBallHB(frame, roiHist):
    global rad_thresh,kernel

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backProj = cv2.calcBackProject([hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)
    mask = cv2.inRange(backProj, 50, 255)
    mask = cv2.dilate(mask,np.ones((3,3)))
    mask = cv2.erode(mask, np.ones((3,3)))

    cv2.imshow('backproj2',mask)
# find the biggest connected contour
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    i = 0
    if len(cnts) > 0:
        for i in range(len(cnts)):
            hull = cv2.convexHull(cnts[i])
            (center, radius) = cv2.minEnclosingCircle(hull)
            if radius < rad_thresh and radius>10:
                break
        M = cv2.moments(hull)
        if M["m00"] > 0:
#            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            center = np.uint32(center)
            return (center[0],center[1]), cnts[i]
        else:
            return None, None
    else:
        return None, None

def kalmanFilter(meas):
    pred = np.array([],dtype=np.int)
    #mp = np.asarray(meas,np.float32).reshape(-1,2,1)  # measurement
    tp = np.zeros((2, 1), np.float32)  # tracked / prediction

    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.00003
    for mp in meas:
        mp = np.asarray(mp,dtype=np.float32).reshape(2,1)
        kalman.correct(mp)
        tp = kalman.predict()
        np.append(pred,[int(tp[0]),int(tp[1])])

    return pred

def removeBG(frame, fgbg):
    bg_mask = fgbg.apply(frame)
    bg_mask = cv2.erode(bg_mask, np.ones((3, 3)))
    bg_mask = cv2.dilate(bg_mask, np.ones((3, 3)))
    frame = cv2.bitwise_and(frame, frame, mask=bg_mask)
    
    return frame

def getHist(roi,mask):
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)    
    roi_hist = cv2.calcHist([roi],[0,1],mask,[180,256],[0,180,0,256])
    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # print(roi_hist)
    return roi_hist

def getMeanROIHist(roi):
    num_roi = roi.shape[-1]

    mean_hist = np.zeros((180,256))
    for i in range(num_roi):
        hist = getHist(roi[:,:,:,i])
