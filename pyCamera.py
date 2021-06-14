import numpy as np
import cv2
import mediapipe as mp
import os

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import autopy
import time
import HandTrackingModule as htm

class recognizer:
    def __init__(self):
        
        ### gesture ###
        self.svm, self.name, self.pca = self.train()

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(min_detection_confidence=0.55, min_tracking_confidence=0.36)

        ### Mouse Control ###
        ##########################
        self.wCam, self.hCam = 640, 480
        self.frameR = 100 # Frame Reduction
        self.smoothening = 7
        #########################

        self.pTime = 0
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0

        '''
        cap = cv2.VideoCapture(0)
        cap.set(3, wCam)
        cap.set(4, hCam)
        '''
        self.detector = htm.handDetector()
        self.wScr, self.hScr = autopy.screen.size()

        ### virtual paint ###
        #######################
        self.brushThickness = 25
        self.eraserThickness = 100
        ########################

        folderPath = "Header"
        myList = os.listdir(folderPath)
        print(myList)
        self.overlayList = []
        for imPath in myList:
            image = cv2.imread(f'{folderPath}/{imPath}')
            self.overlayList.append(image)
        print(len(self.overlayList))
        self.header = self.overlayList[0]
        self.drawColor = (255, 0, 255)
        self.xp, self.yp = 0, 0
        self.imgCanvas = np.zeros((self.hCam, self.wCam, 3), np.uint8)

        #### emotion detection ####
        self.smile = np.load("smile.npy")
        self.sad = np.load("sad.npy")
        self.wow = np.load("wow.npy")
        self.wow = self.wow.reshape(200,50*50*3)
        self.sad = self.sad.reshape(200,50*50*3)
        self.smile = self.smile.reshape(200,50*50*3)
        
        X = np.r_[self.smile,self.sad,self.wow]
        
        labels = np.zeros(X.shape[0])
        labels[0:200]=1.0
        labels[200:400]=2.0
        labels[400:]=3.0
        self.name3 = {1.0:"SMILE",2.0:"SAD",3.0:"WOW"}
        x_train ,x_test, y_train,y_test = train_test_split(X,labels, test_size=0.25)
        self.pca2 = PCA(n_components=3)
        x_train = self.pca2.fit_transform(x_train)
        self.svm2 = SVC()
        self.svm2.fit(x_train,y_train)
        self.mpface = mp.solutions.face_mesh
        self.face_mesh = self.mpface.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)


    def train(self):
        a = []
        r = np.array([])
        n = []
        name = {}
        X = []
        labels = []
        i = 1
        flag = True
        for root, dirs, files in os.walk("Data_collection", topdown=False):
            for x in files:
                if '.npy'in x:
                    p = x.index('.')
                    na = str(x[:p])
                    name[i]=na
                    i = i+1
                    full = 'Data_collection/'+str(x)
                    y = np.load(full)
                    y = y.reshape(200,50*50*3)
                    if flag:
                        r = y
                        flag = False
                    else:
                        r = np.concatenate((r,y)) 
        X = np.asarray(r) 
        labels = np.ones(X.shape[0])
        ii = 2
        for x in range(200,i*200,200):
            if x+200==1600:
                labels[x:] = ii
                ii = ii +1
            elif x+200<1600:
                labels[x:x+200] = ii
                ii = ii+1
            else:
                break
        x_train ,x_test, y_train,y_test = train_test_split(X,labels, test_size=0.25)
        pca = PCA(n_components=3)
        x_train = pca.fit_transform(x_train)
        svm = SVC()
        svm.fit(x_train,y_train)
        x_test = pca.fit_transform(x_test)
        y_pred = svm.predict(x_test)
        print()
        print()
        print(str(accuracy_score(y_test,y_pred))+"-----------------------------------")
        print()
        print()

        return svm, name, pca


    def get_gesture(self, frame):
        #cap = cv2.VideoCapture(0)

        #print("########################## ------ Sign Detection ------ ##########################")
        #_, frame = cap.read()
        
        h, w = self.hCam, self.wCam

            #while cap.isOpened():
                #ret, frame = cap.read()      
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #image = cv2.flip(image, 1)
        image.flags.writeable = False
        # Detections
        results = self.hands.process(image)
        # Set flag to true
        image.flags.writeable = True
        
        hand_landmarks = results.multi_hand_landmarks
        if True:
            if hand_landmarks:
                for handLMs in hand_landmarks:
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                    for lm in handLMs.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    self.mp_drawing.draw_landmarks(frame, handLMs, self.mp_hands.HAND_CONNECTIONS)
                    face = frame[y_min:y_max,x_min:x_max,:]
                    if face.any():
                        #cv2.imshow("Try",face)
                        face = cv2.resize(face,(50,50))
                        face = cv2.resize(face,(50,50))
                        face = face.reshape(1,-1)
                        face = self.pca.transform(face)
                        p = self.svm.predict(face)
                        n = self.name[int(p)]

                        coords = tuple(np.multiply(
                                np.array((handLMs.landmark[self.mp_hands.HandLandmark.WRIST].x, handLMs.landmark[self.mp_hands.HandLandmark.WRIST].y)),
                            [640,480]).astype(int))
                        image = cv2.putText(frame, n , coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                '''cv2.imshow("Frame", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()'''
        return frame

    def get_emote(self, frame):
        h, w = self.hCam, self.wCam
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(framergb)
        hand_landmarks = results.multi_face_landmarks
        if True:
            if hand_landmarks:
                for handLMs in hand_landmarks:
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                    for lm in handLMs.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    self.mp_drawing.draw_landmarks(frame, handLMs, self.mpface.FACE_CONNECTIONS)
                    face = frame[y_min:y_max,x_min:x_max,:]
                    if face.any():
                        cv2.imshow("Try",face)
                        face = cv2.resize(face,(50,50))
                        face = cv2.resize(face,(50,50))
                        face = face.reshape(1,-1)
                        face = self.pca2.transform(face)
                        p =self.svm2.predict(face)
                        n = self.name3[int(p)]
                        image = cv2.putText(frame, n , (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
    
    def get_label(self, index, hand, results):
        output = None
        for idx, classification in enumerate(results.multi_handedness):
            if classification.classification[0].index == index:
                
                # Process results
                label = classification.classification[0].label
                score = classification.classification[0].score
                text = '{} {}'.format(label, round(score, 2))
                
                # Extract Coordinates
                coords = tuple(np.multiply(
                    np.array((hand.landmark[self.mp_hands.HandLandmark.WRIST].x, hand.landmark[self.mp_hands.HandLandmark.WRIST].y)),
                [640,480]).astype(int))
                
                output = text, coords
                
        return output

    def get_left_or_right(self, frame):
        cap = cv2.VideoCapture(0)

        
            #while cap.isOpened():
        #ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        #image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = self.hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detections
        #print(results)
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                
                # Render left or right detection
                if self.get_label(num, hand, results):
                    text, coord = self.get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        # Save our image    
        #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)

        return image

    def controlMouse(self, img):
        img = self.detector.findHands(img)
        lmList = self.detector.findPosition(img)

        # 2. Get the tip of the index and middle fingers
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            # print(x1, y1, x2, y2)
        
            # 3. Check which fingers are up
            fingers = self.detector.fingersUp()
            # print(fingers)
            cv2.rectangle(img, (self.frameR, self.frameR), (self.wCam - self.frameR, self.hCam - self.frameR),
            (255, 0, 255), 2)
            # 4. Only Index Finger : Moving Mode
            if fingers[1] == 1 and fingers[2] == 0:
                # 5. Convert Coordinates
                x3 = np.interp(x1, (self.frameR, self.wCam - self.frameR), (0, self.wScr))
                y3 = np.interp(y1, (self.frameR, self.hCam - self.frameR), (0, self.hScr))
                # 6. Smoothen Values
                clocX = self.plocX + (x3 - self.plocX) / self.smoothening
                clocY = self.plocY + (y3 - self.plocY) / self.smoothening
            
                # 7. Move Mouse
                autopy.mouse.move(self.wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                self.plocX, self.plocY = clocX, clocY
                
            # 8. Both Index and middle fingers are up : Clicking Mode
            if fingers[1] == 1 and fingers[2] == 1:
                # 9. Find distance between fingers
                length, img, lineInfo = self.detector.findDistance(8, 12, img)
                print(length)
                # 10. Click mouse if distance short
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                    15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()
                    
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                # 9. Find distance between fingers
                length, img, lineInfo = self.detector.findDistance(8, 12, img)
                print(length)
                # 10. Click mouse if distance short
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                    15, (0, 255, 0), cv2.FILLED)
                    print("clicked")
                    autopy.mouse.click(autopy.mouse.Button.RIGHT)

    def virtual_paint(self, frame):
        # 1. Import image
        img = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # 2. Find Hand Landmarks
        img = self.detector.findHands(img, draw=False)
        lmList = self.detector.findPosition(img, draw=False)

        if len(lmList) != 0:

            # print(lmList)

            # tip of index and middle fingers
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            #print(x1,y1)

            # 3. Check which fingers are up
            fingers = self.detector.fingersUp()
            # print(fingers)

            # 4. If Selection Mode – Two finger are up
            if fingers[1] and fingers[2]:
            # xp, yp = 0, 0
                print("Selection Mode")
            # # Checking for the click
                if y1 < 125:
                    if 250 < x1 < w*0.75:
                        header = self.overlayList[0]
                        self.drawColor = (255, 0, 255)
                    elif w*0.75 < x1 < w:
                        header = self.overlayList[1]
                        self.drawColor = (0, 0, 0)
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), self.drawColor, cv2.FILLED)

            # 5. If Drawing Mode – Index finger is up
            if fingers[1] and fingers[2] == False:
                cv2.circle(img, (x1, y1), 15, self.drawColor, cv2.FILLED)
                print("Drawing Mode")
                if self.xp == 0 and self.yp == 0:
                    self.xp, self.yp = x1, y1

                cv2.line(img, (self.xp, self.yp), (x1, y1), self.drawColor, self.brushThickness)

                if self.drawColor == (0, 0, 0):
                    cv2.line(img, (self.xp, self.yp), (x1, y1), self.drawColor, self.eraserThickness)
                    cv2.line(self.imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, self.eraserThickness)
            
                else:
                    cv2.line(img, (self.xp, self.yp), (x1, y1), self.drawColor, self.brushThickness)
                    cv2.line(self.imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, self.brushThickness)

                self.xp, self.yp = x1, y1

            # # Clear Canvas when all fingers are up
            if all (x >= 1 for x in fingers):
                self.imgCanvas = np.zeros((h, w, 3), np.uint8)

        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img,imgInv)
        img = cv2.bitwise_or(img,self.imgCanvas)

        # Setting the header image
        header = cv2.resize(self.header, (640,75))
        img[0:75, 0:640] = header
        # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
        cv2.imshow("Image", img)
        #cv2.imshow("Canvas", self.imgCanvas)
        #cv2.imshow("Inv", imgInv)

def collection(name):
    nameee =name+".npy"
    mphands = mp.solutions.hands
    hands = mphands.Hands(False, 1,min_detection_confidence=0.5, min_tracking_confidence=0.35,)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    print("########################## Data Collection for {} Sign ##########################".format(name))
    _, frame = cap.read()
    data = []
    h, w, c = frame.shape

    while True:
        ret, frame = cap.read()
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks
        if ret:
            if hand_landmarks:
                for handLMs in hand_landmarks:
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                    for lm in handLMs.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
                    face = frame[y_min:y_max,x_min:x_max,:]
                if len(data)<200:
                    if face.any():
                        cv2.imshow("Try",face)
                        face = cv2.resize(face,(50,50))
                        print("No of frames collected for "+name+" Sign: "+str(len(data)))
                        data.append(face)
                else:
                    break
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    save_path = 'Data_collection/'
    completeName = os.path.join(save_path, nameee)
    np.save(completeName,data)

def data_collection(rec): 
    while True:
        n = input("Enter Y to add more data : ")
        if n == "Y":
            name = input("Enter the name and gesture of the file to be collected : ")
            collection(name)
        else:
            break
    print("Data collected------------------")

    rec.svm, rec.name, rec.pca = rec.train()