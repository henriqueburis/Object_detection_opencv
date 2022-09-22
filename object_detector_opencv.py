import cv2
import imutils
import numpy as np

cap = cv2.VideoCapture("output1_orig.mp4")

#object detection form stable camera
object_detector = cv2.createBackgroundSubtractorKNN(history=2000,dist2Threshold=15000) #history=100,dist2Threshold=400,detectShadows=False



while True:
    _, frame = cap.read()
    # resize do tamanho do video original
    frame = imutils.resize(frame, width=450)
    
    frameClone = frame.copy()
    
    mask = object_detector.apply(frameClone)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x,y,w,h = cv2.boundingRect(cnt)
            roi = frameClone[y:y+h, x:x+w]
            cv2.rectangle(frameClone,(x, y),(x + w, y + h),(0, 255, 0 ), 3)

    cv2.imshow("mask", mask)
    cv2.imshow("frameClone", frameClone)
    cv2.imshow("roi", roi)
    #cv2.imshow("limiar_frame", limiar_frame)

    key = cv2.waitKey(1)
    if key == 113:
        break
    
cap.release()
cv2.destroyAllWindows()
  


