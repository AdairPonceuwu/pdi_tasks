import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while(1):

    # Task each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    (h, s, v) = cv.split(hsv)

    #Blue
    #lower_blue = np.array([110,50,50])
    #upper_blue = np.array([130,255,255])
    
    #Green
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    

    #mask = cv.inRange(hsv, lower_green, upper_green)
    mask = cv.inRange(h, lower_green, upper_green)

    # Bitwise AND mask and original image
    res = cv.bitwise_and(frame, frame, mask = mask)

    cv.imshow('Frame', frame)
    cv.imshow('h', h)
    cv.imshow('mask', mask)
    cv.imshow('res', res)

    k = cv. waitKey(5) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()