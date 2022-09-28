import cv2
import numpy as np
import glob
import os
import shutil

files = sorted(glob.glob("segment_out/predictions/*.png"))

for file in files:
    img = cv2.imread(file,0)
    name = os.path.basename(file)
    og = cv2.resize(cv2.imread("images/context/"+name),(448,256), interpolation=cv2.INTER_CUBIC)
    overlay = og.copy()
    #overlay = cv2.imread("base_final/636__chevron-left__727__20220207170720_0060__221_inpaint.png")
    kernel = np.ones((8,8),np.uint8)
    image = cv2.erode(img,kernel=kernel,iterations=1)
    ret, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if(len(contours)==0):
        cv2.imwrite("processed_out/"+name,og)
    
    for c in contours:        
        if(cv2.contourArea(c)>100):
            x,y,w,h = cv2.boundingRect(c)
            alpha = 0.5
            cv2.rectangle(overlay, (x,y),(x+w,y+h),(0,255,0),-1)
            img_new = cv2.addWeighted(overlay, alpha, og, 1-alpha, 0)
            cv2.imwrite("processed_out/"+name,img_new)
        else:
            cv2.imwrite("processed_out/"+name,og)