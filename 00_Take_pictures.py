#import sys
#import os

#import matplotlib
#import numpy as np
#import matplotlib.pyplot as plt
#import copy
import cv2

imgloc = '00 Data/pictures'
startnumber = 1
webcamchannel = 1

cap = cv2.VideoCapture(webcamchannel)

while True:
    ret, img = cap.read()
    #img = cv2.flip(img, 1)
        
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    a = cv2.waitKey(10)
    if ret:
        x1, y1, x2, y2 = 100, 100, 300, 300
        img_cropped = img[y1:y2, x1:x2]

        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.imshow("img", img)
        
        # check if enter is pressed
        if cv2.waitKey(1) == 13:
            cv2.imwrite(imgloc + '/picture_'+ str(startnumber) + '.jpg', img_cropped)
            print('image is saved!')
            startnumber += 1
    else:
        break

# Following line should appear but is not working with opencv-python package
# cv2.destroyAllWindows() 
cv2.VideoCapture(webcamchannel).release()