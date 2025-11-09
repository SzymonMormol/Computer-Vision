import numpy as np
import matplotlib.pyplot as plt
import cv2 
import mediapipe as mp
from google.protobuf.json_format import MessageToDict


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
   
rectBotX = 0
rectBotY = 0
rectTopY = 0
rectTopX = 0
temp_height = 0
cursorCx = 0
cursorCy = 0
Color = (255,0,255)

CX,CY,W,H = 100,100,200,200

while(1):

 
    
    ret, frame = cap.read()
    img = cv2.flip(frame,1)

   

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
         for i in results.multi_handedness:
            label = MessageToDict(i)[
                    'classification'][0]['label']     

            if label == 'Right': #If we want differentiate hands:#
                for handlms in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)
                    
                    for id,lm in enumerate(handlms.landmark):
                        h,w,c = frame.shape
                        cx,cy = int(lm.x*w), int(lm.y*h)

                        if id == 10:
                            temp_height = cy-10
                        if id == 0:
                            rectBotY = cy
                        if id == 4:
                            rectBotX = cx
                        if id == 8:
                            rectTopY=cy-40
                            cursorCy =cy
                            cursorCx = cx
                        if id == 20:
                            rectTopX = cx+100

                        if(temp_height<rectTopY):
                            cv2.rectangle(img,(rectBotX,rectBotY),(rectTopX,temp_height),(0,255,0),3)
                        else:
                            cv2.rectangle(img,(rectBotX,rectBotY),(rectTopX,rectTopY),(0,255,0),3)
            elif label == 'Left':
                for handlms in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)
                    
                    for id,lm in enumerate(handlms.landmark):
                        h,w,c = frame.shape
                        cx,cy = int(lm.x*w), int(lm.y*h)

                        if id == 10:
                            temp_height = cy-40
                        if id == 0:
                            rectBotY = cy
                        if id == 4:
                            rectBotX = cx
                        if id == 8:
                            rectTopY=cy-10
                            cursorCy =cy
                            cursorCx = cx

                        if id == 20:
                            rectTopX = cx-100

                        if(temp_height<rectTopY):
                            cv2.rectangle(img,(rectBotX,rectBotY),(rectTopX,temp_height),(0,255,0),3)
                        else:
                            cv2.rectangle(img,(rectBotX,rectBotY),(rectTopX,rectTopY),(0,255,0),3)

            if(CX-W//2<cursorCx<CX+W//2 and CY-H//2 < cursorCy  < CY+H//2):
                Color=(0,0,0)
                CX = cursorCx
                CY = cursorCy
            else:
                Color=Color = (255,0,255)

    cv2.rectangle(img, (CX-W//2,CY-H//2),(CX+W//2,CY+H//2),Color,cv2.FILLED)

       

                    
                    

                
 

    cv2.imshow('frame',img)
    if cv2.waitKey(1) == ord('q'):
        break

