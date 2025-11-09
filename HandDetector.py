import numpy as np
import matplotlib.pyplot as plt
import cv2 
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import math
from ctypes import cast,POINTER
import comtypes 
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
cx8,cy8,cx4,cy4 = 0,0,0,0
devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume.QueryInterface(IAudioEndpointVolume)


while(1):

    ret,frame = cap.read()
    imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
         for i in results.multi_handedness:
            label = MessageToDict(i)[
                    'classification'][0]['label']     
            #if label == 'Left': #If we want differentiate hands:
            for handlms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame,handlms,mpHands.HAND_CONNECTIONS)

                for id,lm in enumerate(handlms.landmark):          
                    h,w,c = frame.shape
                    cx,cy = int(lm.x*w), int(lm.y*h)

                    if id == 8 or id == 4:
                        cv2.circle(frame,(cx,cy),10,(255,0,255),cv2.FILLED)
                        if(id ==8):
                            cx8=cx
                            cy8=cy
                        elif(id==4):
                            cx4=cx
                            cy4=cy
                        cv2.line(frame,(cx8,cy8),(cx4,cy4),(255,0,255),10)
                        midx=(cx4+cx8)//2 
                        midy=(cy4+cy8)//2                    
                        distance = math.sqrt(pow((cx4-cx8),2) + pow((cy4-cy8),2))
                        if distance<22:
                            cv2.circle(frame,(midx,midy),10,(255,0,0),cv2.FILLED)
                        else:
                            cv2.circle(frame,(midx,midy),10,(255,255,0),cv2.FILLED)
                        power = -65.0 + distance/3.68
                        if(power>0):
                            power = 0
                        print(power)
                        volume.SetMasterVolumeLevel(power,None)





    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break