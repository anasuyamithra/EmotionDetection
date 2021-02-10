from ms_main import *
import os
import cv2 # OpenCV
import qimage2ndarray # for a memory leak,see gist
import sys # for exiting
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

i = 0
class UIFunctions(MainWindow):
    def displayFrame(self):
        #load model
        model = model_from_json(open("fer.json", "r").read())
        #load weights
        model.load_weights('fer.h5')


        face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


        cap=cv2.VideoCapture(0)

        while True:
            ret, frame=cap.read()# captures frame and returns boolean value and captured image
            if not ret:
                continue
            gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


            for (x,y,w,h) in faces_detected:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)
                roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
                roi_gray=cv2.resize(roi_gray,(48,48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)

                #find max indexed array
                max_index = np.argmax(predictions[0])

                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                predicted_emotion = emotions[max_index]

                cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            #resized_img = cv2.resize(test_img, (1000, 700))
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = qimage2ndarray.array2qimage(frame)
            self.ui.label_display.setPixmap(QPixmap.fromImage(image))
            #cv2.imshow('Facial emotion analysis ',resized_img)



            if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
                break
        
        cap.release()

    