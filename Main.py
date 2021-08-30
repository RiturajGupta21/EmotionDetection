from time import sleep
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

faceClassifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
emotionClassifier =load_model('./Emotion_Detection.h5')

classLabels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    grab,frame = capture.read()
    labels = []
    convertGrayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    detectFace = faceClassifier.detectMultiScale(convertGrayscale,1.3,5)

    for (xcord,ycord,width,height) in detectFace:
        cv2.rectangle(frame,(xcord,ycord),(xcord+width,ycord+height),(255,255,255),2)
        resizeGray = convertGrayscale[ycord:ycord+height,xcord:xcord+width]
        resizeGray = cv2.resize(resizeGray,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([resizeGray])!=0:
            imgtoarr = resizeGray.astype('float')/255.0
            imgtoarr = img_to_array(imgtoarr)
            imgtoarr = np.expand_dims(imgtoarr,axis=0)
            prediction = emotionClassifier.predict(imgtoarr)[0]
            label=classLabels[prediction.argmax()]
            labelPosition = (xcord,ycord)
            cv2.putText(frame,label,labelPosition,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
capture.release()
cv2.destroyAllWindows()