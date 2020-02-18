import numpy as np
import argparse
import cv2
import math
#from collections import deque
#import matplotlib.pyplot as plt
img = None
orig = None
bbox = None
roi2, roi2_init = None,None

kernel = np.ones((3,3))
ix,iy=0,0
draw = False
rad_thresh = 50

def getArguements():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the video file")
    ap.add_argument("-bg", "--background",
                    help="path to the extracted background")
    ap.add_argument("-l", "--limits",
                    help="path to the rgb threshold limits file")
    ap.add_argument("-vis", "--visualise",
                    help="view the frames and the ball detection",action='store_true')
    ap.add_argument("-s", "--save",
                    help="save data",action='store_true')
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


def getLimits_HSV(roi,roi_mask):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(roi)
    limits = [(int(np.amax(h[roi_mask>0])), int(np.amax(s[roi_mask>0])), int(np.amax(v[roi_mask>0]))), 
              (int(np.amin(h[h>0])), int(np.amin(s[roi_mask>0])), int(np.amin(v[roi_mask>0])))]
    return limits

def getLimits_RGB(roi,roi_mask):
    b,g,r = cv2.split(roi)
    b_mean = np.median(b[roi_mask>0])
    g_mean = np.median(g[roi_mask>0])
    r_mean = np.median(r[roi_mask>0])
            
    b_std = 1.5*np.std(b[roi_mask>0])
    g_std = 1.5*np.std(g[roi_mask>0])
    r_std = 1.5*np.std(r[roi_mask>0])

    limits = [(min(int(b_mean+b_std),255), min(int(g_mean+g_std),255), min(int(r_mean+r_std),255)), 
              (int(b_mean-b_std), int(g_mean-g_std), int(r_mean-r_std))]
    return limits

def applyMorphTransforms(mask):
    global kernel
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
    
def detectBallThresh_RGB(frame,limits):
    global rad_thresh
    upper = limits[0]
    lower = limits[1]

    mask = cv2.inRange(frame, lower, upper)
    mask = applyMorphTransforms(mask)
    cv2.imshow('mask_threh', mask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    flag = False
    if len(cnts) > 0:
        for i in range(len(cnts)):
            hull = cv2.convexHull(cnts[i])
            (center, radius) = cv2.minEnclosingCircle(hull)
            if radius < rad_thresh and radius > 10:
                flag = True
                break
            
        if not flag: # No contour found
            return None, None
        else:
            center = np.uint32(center)
            return (center[0],center[1]), radius
    else: # No contour found
        return None, None

def detectBallHB(frame, roiHist):
    global rad_thresh,kernel

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backProj = cv2.calcBackProject([hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)
    mask = cv2.inRange(backProj, 50, 255)
    mask = cv2.dilate(mask,np.ones((3,3)))
    mask = cv2.erode(mask, np.ones((3,3)))

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

def removeBG(frame, fgbg):
    bg_mask = fgbg.apply(frame)
    bg_mask = cv2.dilate(bg_mask, np.ones((3, 3)))
    bg_mask = cv2.erode(bg_mask, np.ones((5, 5)))
    frame_fg = cv2.bitwise_and(frame, frame, mask=bg_mask)
    return frame_fg, bg_mask
