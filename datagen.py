import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


def quit_application():
    cap.release()
    cv2.destroyAllWindows()
    print('Quitting...')
    raise SystemExit


# CONSTANTS
fps = 25
slidingWindowLength = 30
smoothFaceWindowLength = 5
maxVidWidth = 720
stressThresh = 0.5
upperHRLimit = 2
lowerHRLimit = 0.25
faceCascade = cv2.CascadeClassifier('./classifiers/facehaar.xml')
eyeCascade = cv2.CascadeClassifier('./classifiers/eyehaar.xml')
videoPath = './data/' + sys.argv[1] # Code execution: datagen.py "name.mp4"
key = 0
numEyes = 2
frameNo = 0
color = 0, 255, 0
plt.axis([0, 350, -30, 40])
cap = cv2.VideoCapture(videoPath)
faceArray = np.array([])
framesArray = np.array([])
pointDataArray = np.array([])
diff = 0
beatFrame = 0
betweenBeats = 0

# Heartbeat extractor
sumDataPnts = 0
counterDataPnts = 0
while cap.isOpened():
    if key in [27, 1048603]:
        quit_application()
    ret, frame = cap.read()
    if frame is None:
        break
    key = cv2.waitKey(1)
    frameNo += 1
    if np.shape(frame)[1] > maxVidWidth:
        resizeFactor = maxVidWidth / np.shape(frame)[1]
        frame = cv2.resize(frame, (0, 0), fx=resizeFactor, fy=resizeFactor)

    # make frame smaller if above maximum width
    marked = frame.copy()
    green = frame.copy()
    green[:, :, 0] = 0
    green[:, :, 2] = 0
    
    # take channels
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    
    # run haar cascade face detection on greyscale image
    if faces is not ():
        x, y, w, h = faces[0]
        cv2.rectangle(marked, (x, y), (x + w, y + h), color, 3)
        
        # mark face
        roi = green[y:y + h, int(x + (w * 0.2)):int(x + (w * 0.8))]  # face ROI
        face = gray[y:y + h, x:x + w]
        
        # set region of interest
        faceMean = np.mean(roi)
        faceArray = np.append(faceArray, faceMean)
        
        if len(faceArray) > slidingWindowLength:
            pastAvg = np.mean(faceArray[-slidingWindowLength:])
            smoothFaceMean = np.mean(faceArray[-smoothFaceWindowLength:])
            oldDiff = diff
            diff = (smoothFaceMean - pastAvg)
            
            if diff > 0:
                if oldDiff < 0:
                    oldBetweenBeats = betweenBeats
                    betweenBeats = (frameNo - beatFrame) / fps
                    if betweenBeats > upperHRLimit or betweenBeats < lowerHRLimit:
                        betweenBeats = oldBetweenBeats
                    else:
                        print(60 / betweenBeats)  # bpm output
                        sumDataPnts += 60 / betweenBeats # to find average heartrate
                        counterDataPnts += 1
                        if betweenBeats < stressThresh:
                            color = (0, 0, 255)
                        else:
                            color = (0, 255, 0)
                    beatFrame = frameNo
            framesArray = np.append(framesArray, frameNo)
            pointDataArray = np.append(pointDataArray, diff * 100)
        # TODO: Eye detection - pupils size. 
        eyes = eyeCascade.detectMultiScale(face)
        for i in range(numEyes):
            if eyes.size < 4 * numEyes:
                break
            ex, ey, ew, eh = eyes[i]
            marked = cv2.rectangle(marked, (ex + x, eh + y), (ex + ew + x, ey + eh + y), (255, 0, 0), 3)
            eyeRoi = gray[ex + x:ex + ew + x, eh + y:ey + eh + y]
            if eyeRoi.size:
                pupil = cv2.HoughCircles(eyeRoi, cv2.HOUGH_GRADIENT, 1, 200, param1=200, param2=1, minRadius=0,
                                         maxRadius=0)
                if pupil is not None:
                    cv2.circle(marked, (int(pupil[0][0][0]) + ex + x, int(pupil[0][0][1] + eh + y)), 1, (255, 255, 255),
                               3)
    cv2.imshow('heartbeat', marked)
    
print("Average: ", sumDataPnts / counterDataPnts)
plt.plot(framesArray, pointDataArray, '.r-')
cv2.waitKey()
quit_application()
