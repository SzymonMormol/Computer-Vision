import numpy as np
import matplotlib.pyplot as plt
import cv2 

cascade = cv2.CascadeClassifier("haarcascade_frontalcatface_extended.xml")

cap = cv2.VideoCapture(0)


while(1):

    ret, frame = cap.read()

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(frame_gray,1.1,1)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),4)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
