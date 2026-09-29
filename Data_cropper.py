# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:14:36 2026

@author: ARJASK
"""

import numpy as np

from PIL import Image

import cv2

def find_circle_coords(img,rad = 710):
    out = img.copy()
    out = out.convert('L')
    out = np.asarray(out)
    blur = cv2.medianBlur(out,3)
    #return blur
    thr = np.average(out)
    blur[blur > thr] = 255
    blur[blur <= thr] = 0
    
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=2,
        minDist=100,      
        param1=300,         
        param2=40,       
        minRadius=650,       
        maxRadius=750
    )
    
    if circles is None:
        print("No circles found")
    if circles is not None:
        if len(circles) == 1:
            x,y,r = circles[0][0]
            return x,y,r
        else:
            print("{} circles found".format(len(circles)))

    return None

def crop(im, cent, rad):
    x1 = cent[0] - rad
    x2 = cent[0] + rad
    y1 = cent[1] - rad
    y2 = cent[1] + rad

    out = im.copy()
    return out.crop((x1,y1,x2,y2))

def get_img_np(fnam, r = 720):
    img = Image.open(fnam)
    x,y,r2 = find_circle_coords(img)
    cr = np.asarray(crop(img,(x,y),r).convert('L'))
    return cr
    

def crop_rectangle_data(fnam):
    # Finds biggest rectangle in image and crops it
    img = cv2.imread(fnam)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    all_contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    RR = []
    N,M,_ = img.shape
    for c in all_contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*perimeter, True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(approx)
            if w > 0.95*M and h > 0.95*N:
                continue
            RR.append((x,y,w,h,w*h))
    RR = np.array(RR)
    idx = np.argsort(-RR[:,-1])
    RR = RR[idx]
    x,y,w,h,a = RR[0]
    return np.array(gray[y:y+h,x:x+w])