import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)

    
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    mask = cv.inRange(hls, lower_green, upper_green)

    
    res = cv.bitwise_and(frame, frame, mask=mask)

    
    cv.imshow('Original Frame', frame)
    cv.imshow('Mask', mask)
    cv.imshow('Result', res)

    if cv.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
