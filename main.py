import cv2
import time
from collections import deque

import pyCamera

import HandTrackingModule as htm
detector = htm.handDetector()

rec = pyCamera.recognizer()
cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)
pTime = time.time()

mode = "MAIN"
que = deque(maxlen=10)

while cap.isOpened():
    ret, frame = cap.read()

    img = detector.findHands(frame, draw=False)
    lmList = detector.findPosition(frame, draw=False)
    if(mode == "MAIN"):
        frame = cv2.putText(frame, 'Menu', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, '1. Gesture Recognition', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, '2. Virtual Control', (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, '3. Virtual Paint', (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, '4. Add Gesture', (50,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, '5. Emotion Detection', (50,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
        if len(lmList) != 0:
            fingers = detector.fingersUp()
            #print(fingers)
            if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                que.append(1) 
                if sum(que)==10:
                    mode = "GESTURE"
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                que.append(2)
                if sum(que)==20:
                    mode = "MOUSE"
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
                que.append(3)
                if sum(que)==30:
                    mode = "PAINT"
            if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                que.append(4)
                if sum(que)==40:
                    mode = "TRAIN"
            if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                que.append(6)
                if sum(que)==60:
                    mode = "EMOTE"

    elif( mode == "GESTURE"):
        frame = rec.get_gesture(frame)
        if len(lmList) != 0:
            fingers = detector.fingersUp()
            if fingers[0] ==1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                que.append(5)
                if sum(que)==50:
                    mode = "MAIN"
                    cv2.destroyAllWindows()

    elif( mode == "EMOTE"):
        frame = rec.get_emote(frame)
        if len(lmList) != 0:
            fingers = detector.fingersUp()
            if fingers[0] ==1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                que.append(5)
                if sum(que)==50:
                    mode = "MAIN"
                    cv2.destroyAllWindows()
    
    elif( mode == "MOUSE"):
        rec.controlMouse(frame)
        if len(lmList) != 0:
            fingers = detector.fingersUp()
            if fingers[0] ==1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                que.append(5)
                if sum(que)==50:
                    mode = "MAIN"
                    cv2.destroyAllWindows()

    elif( mode == "PAINT"):
        rec.virtual_paint(frame)
        if len(lmList) != 0:
            fingers = detector.fingersUp()
            if fingers[0] ==1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                que.append(5)
                if sum(que)==50:
                    mode = "MAIN"
                    cv2.destroyAllWindows()

    elif( mode == "TRAIN"):
        print(mode)
        cv2.destroyAllWindows()
        cap.release()
        pyCamera.data_collection(rec)
        mode = "MAIN"
        cap = cv2.VideoCapture(0)

    #rec.virtual_paint(frame)
    #frame = rec.get_left_or_right(frame)
    #rec.controlMouse(frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("TDP Camera Toolkit", frame)
    k = cv2.waitKey(10)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()