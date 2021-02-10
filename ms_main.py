''' importing header files '''

import sys
import platform
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QTimer, pyqtSlot, QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtGui import (QBrush, QIcon, QImage, QPixmap, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *
import cv2 
import os
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
''' Gui Window '''
from MainWindow import Ui_MainWindow

''' import files '''
#from ms_functions import *


#counter = 0

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        ''' Toggle menu '''
        #self.ui.btn_toggle.clicked.connect(lambda: UIFunctions.toggleMenu(self, 150, True))
        
        
        #Pages
        
        # PAGE 1
        self.ui.Btn_about.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.stackedWidgetPage1))
        
        # PAGE 2
        self.ui.Btn_emotion.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.stackedWidgetPage2))

        # PAGE 3
        #self.ui.Btn_capture.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.stackedWidgetPage3))

        self.ui.Btn_capture.clicked.connect(lambda: self.showSave())

        #FOR VIDEO
        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.Btn_emotion.clicked.connect(self.controlTimer)

        #self.ui.Btn_capture.clicked.connect(self.controlTimer)

        '''
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)



        timer = QTimer()
        timer.timeout.connect(UIFunctions.displayFrame)
        timer.start(60)
        

        self.show()
        '''
    
    def showSave(self):
        self.controlTimer()
        inputwin = QInputDialog(self)
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(9)
        inputwin.setFont(font);

        inputwin.setStyleSheet( "QDialog{background-color: #84dcc6;}" "QInputDialog {background-color: #FA7F72;}""QInputDialog QPushButton{\n"
" background-color: #ffffff;\n"
" border: 1px solid #ffffff;\n"
" color:#FF686B; \n"
" border-radius: 5px;\n"
"}\n"
"QInputDialog QPushButton:hover{\n"
" background-color:  #A5FFD6;\n"
" border: 1px solid #A5FFD6;\n"
" color:#3c3c3c;\n"
"}");
        text, ok = inputwin.getText(self, 'Save Image', 'Save Image As:')

        
        
		
        if ok:
            imgname = str(text) + '.png'
            self.qImg.save(imgname)
            inputwin.close()
            
            

    # view camera
    def viewCam(self):

        #load model
        model = model_from_json(open("fer_new.json", "r").read())
        #load weights
        model.load_weights('fer_new.h5')
        face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # read image in BGR format
        ret, image_emotion = self.cap.read()
        # convert image to RGB format
        image_emotion = cv2.cvtColor(image_emotion, cv2.COLOR_BGR2RGB)

        gray_img= cv2.cvtColor(image_emotion, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x,y,w,h) in faces_detected:
            cv2.rectangle(image_emotion,(x,y),(x+w,y+h),(165,255,214),thickness=3)
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255
            #f = ImageFont.truetype("arial.ttf", 18)

            predictions = model.predict(img_pixels)

            #find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(image_emotion, predicted_emotion, (int(x), int(y)-2), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,104,107), 2)

        #resized_img = cv2.resize(test_img, (1000, 700))

        # get image infos
        height, width, channel = image_emotion.shape
        
        h_anasuya = 557
        w_anasuya = 830
        h = 480
        w = 640
        step = channel * width
        
        # create QImage from image
        self.qImg = QImage(image_emotion.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.video_label.setPixmap(QPixmap.fromImage(self.qImg))
        #self.ui.video_label.adjustSize()
        #self.ui.video_label.setScaledContents(True)
        #self.ui.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            # update control_bt text
            #self.ui.control_bt.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())