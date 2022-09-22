import cv2
import imutils
import numpy as np

cap = cv2.VideoCapture("20180910_144521.mp4")

#object detection form stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)

while True:
    
    ret, frame = cap.read()
    # resize do tamanho do video original
    frame = imutils.resize(frame, width=350)
    frameClone = frame.copy()

    limiar_frame = frame[80: 550 , 200: 300]
    
    mask = object_detector.apply(frameClone)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(frame, [cnt], -1, (0,255,0),2)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(limiar_frame,(x, y),(x + w, y + h),(0, 255, 0 ), 3)

    cv2.imshow("mask", mask)
    cv2.imshow("frame", frame)
    cv2.imshow("limiar_frame", limiar_frame)

    key = cv2.waitKey(1)
    if key == 113:
        break
    

cap.release()
cv2.destroyAllWindows()
  


