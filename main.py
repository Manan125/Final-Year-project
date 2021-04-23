# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout,QPushButton,QFileDialog,QLabel,QTextEdit
from PyQt5.QtGui import QPixmap
# from ML import *
import json
import os, sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from imgaug import augmenters as iaa
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.models import Model
from keras.applications import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import Callback
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import tensorflow as tf
import keras
import warnings
warnings.filterwarnings("ignore")

predicted = []
submit=[]
MODEL=None
PATH=""
INPUT_SHAPE = (299,299,3)
BATCH_SIZE = 10
root=""
label=[
'Nucleoplasm',  
'Nuclear membrane',   
'Nucleoli'   ,
'Nucleoli fibrillar center'   ,
'Nuclear speckles'   ,
'Nuclear bodies'   ,
'Endoplasmic reticulum',   
'Golgi apparatus ',  
'Peroxisomes ',  
'Endosomes',   
'Lysosomes  ', 
'Intermediate filaments ',  
'Actin filaments ' , 
'Focal adhesion sites',   
'Microtubules' ,  
'Microtubule ends',   
'Cytokinetic bridge',   
'Mitotic spindle',   
'Microtubule organizing center ',  
'Centrosome ',  
'Lipid droplets',   
'Plasma membrane',   
'Cell junctions',   
'Mitochondria',   
'Aggresome',   
'Cytosol',   
'Cytoplasmic bodies',   
'Rods & rings',  
]

def f1(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

class data_generator:
    
    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset_info[idx]['path'], shape)   
                if augument:
                    image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield batch_images, batch_labels
            
    
    def load_image(path, shape):
        R = np.array(Image.open(path+'_red.png'))
        G = np.array(Image.open(path+'_green.png'))
        B = np.array(Image.open(path+'_blue.png'))
        Y = np.array(Image.open(path+'_yellow.png'))

        image = np.stack((
            R/2 + Y/2, 
            G/2 + Y/2, 
            B),-1)
        
        image = cv2.resize(image, (shape[0], shape[1]))
        image = np.divide(image, 255)
        return image  
                
            
    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug

def load_models():
    MODEL = load_model("InceptionResNetV2.model" , custom_objects={'f1': f1}
    )
    return MODEL

def submit_list(path):
    for i in os.listdir(path):
        test=i.split("_")
        submit.append(test[0])
    return submit
def get_prediction():
    for name in tqdm(submit):
        path = os.path.join('test/', name)
        image = data_generator.load_image(path, INPUT_SHAPE)
        score_predict = model.predict(image[np.newaxis])[0]
        label_predict = np.arange(28)[score_predict>=0.2]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)
    return predicted





class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.setEnabled(True)
        MainWindow.resize(1047, 665)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("\n"
"background-color: rgb(255, 214, 166);")
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.readimage = QtWidgets.QLabel(self.centralwidget)
        self.readimage.setGeometry(QtCore.QRect(100, 60, 241, 181))
        self.readimage.setStyleSheet("background-color: rgb(255, 228, 194);")
        self.readimage.setText("")
        self.readimage.setScaledContents(True)
        self.readimage.setObjectName("readimage")
        self.resultarea = QtWidgets.QLabel(self.centralwidget)
        self.resultarea.setGeometry(QtCore.QRect(670, 60, 311, 141))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.resultarea.setFont(font)
        self.resultarea.setStyleSheet("background-color: rgb(211, 211, 158);")
        self.resultarea.setLineWidth(4)
        self.resultarea.setText("")
        self.resultarea.setAlignment(QtCore.Qt.AlignCenter)
        self.resultarea.setObjectName("resultarea")
        self.Browesredimage = QtWidgets.QPushButton(self.centralwidget)
        self.Browesredimage.setGeometry(QtCore.QRect(10, 440, 201, 61))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Browesredimage.sizePolicy().hasHeightForWidth())
        self.Browesredimage.setSizePolicy(sizePolicy)
        self.Browesredimage.setStyleSheet("background-color: rgb(179, 0, 0);\n"
"color: rgb(255, 255, 255);\n"
"font-size: 18px;\n"
"border-radius: 8px;")
        self.Browesredimage.setObjectName("Browesredimage")
        self.prediction = QtWidgets.QPushButton(self.centralwidget)
        self.prediction.setGeometry(QtCore.QRect(450, 470, 221, 61))
        self.prediction.setStyleSheet("background-color: rgb(170, 170, 127);\n"
"color: rgb(255, 255, 255);\n"
"font-size: 20px;\n"
"border-radius: 16px;")
        self.prediction.setObjectName("prediction")
        self.greenimage = QtWidgets.QLabel(self.centralwidget)
        self.greenimage.setGeometry(QtCore.QRect(100, 250, 241, 181))
        self.greenimage.setStyleSheet("background-color: rgb(208, 255, 252);")
        self.greenimage.setText("")
        self.greenimage.setScaledContents(True)
        self.greenimage.setObjectName("greenimage")
        self.blueimage = QtWidgets.QLabel(self.centralwidget)
        self.blueimage.setGeometry(QtCore.QRect(370, 250, 231, 181))
        self.blueimage.setStyleSheet("background-color: rgb(120, 255, 129);")
        self.blueimage.setText("")
        self.blueimage.setScaledContents(True)
        self.blueimage.setObjectName("blueimage")
        self.yellowimage = QtWidgets.QLabel(self.centralwidget)
        self.yellowimage.setGeometry(QtCore.QRect(370, 60, 231, 181))
        self.yellowimage.setStyleSheet("background-color: rgb(211, 225, 206);")
        self.yellowimage.setText("")
        self.yellowimage.setScaledContents(True)
        self.yellowimage.setObjectName("yellowimage")
        self.Browesblueimage = QtWidgets.QPushButton(self.centralwidget)
        self.Browesblueimage.setGeometry(QtCore.QRect(10, 510, 201, 61))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Browesblueimage.sizePolicy().hasHeightForWidth())
        self.Browesblueimage.setSizePolicy(sizePolicy)
        self.Browesblueimage.setStyleSheet("background-color: rgb(0, 0, 255);\n"
"color: rgb(255, 255, 255);\n"
"font-size: 18px;\n"
"border-radius: 8px;")
        self.Browesblueimage.setObjectName("Browesblueimage")
        self.Browesyellowimage = QtWidgets.QPushButton(self.centralwidget)
        self.Browesyellowimage.setGeometry(QtCore.QRect(220, 440, 201, 61))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Browesyellowimage.sizePolicy().hasHeightForWidth())
        self.Browesyellowimage.setSizePolicy(sizePolicy)
        self.Browesyellowimage.setStyleSheet("background-color: rgb(223, 223, 0);\n"
"color: rgb(0, 0, 0);\n"
"font-size: 18px;\n"
"border-radius: 8px;")
        self.Browesyellowimage.setObjectName("Browesyellowimage")
        self.Browesgreenimage = QtWidgets.QPushButton(self.centralwidget)
        self.Browesgreenimage.setGeometry(QtCore.QRect(220, 510, 201, 61))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Browesgreenimage.sizePolicy().hasHeightForWidth())
        self.Browesgreenimage.setSizePolicy(sizePolicy)
        self.Browesgreenimage.setStyleSheet("background-color: rgb(0, 124, 0);\n"
"color: rgb(255, 255, 255);\n"
"font-size: 18px;\n"
"border-radius: 8px;")
        self.Browesgreenimage.setObjectName("Browesgreenimage")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(2, -1, 1091, 51))
        font = QtGui.QFont()
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit.setFont(font)
        self.lineEdit.setAutoFillBackground(False)
        self.lineEdit.setStyleSheet("background-color: rgb(211, 211, 158);")
        self.lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit.setReadOnly(True)
        self.lineEdit.setObjectName("lineEdit")
        self.disease = QtWidgets.QLabel(self.centralwidget)
        self.disease.setGeometry(QtCore.QRect(670, 240, 321, 201))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.disease.setFont(font)
        self.disease.setStyleSheet("background-color: rgb(211, 211, 158);")
        self.disease.setLineWidth(4)
        self.disease.setText("")
        self.disease.setAlignment(QtCore.Qt.AlignCenter)
        self.disease.setObjectName("disease")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1047, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.Browesgreenimage.clicked.connect(self.get_green_image)
        self.Browesredimage.clicked.connect(self.get_red_image)
        self.Browesblueimage.clicked.connect(self.get_blue_image)
        self.Browesyellowimage.clicked.connect(self.get_yellow_image)
        self.prediction.clicked.connect(self.predictions)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Protein Image Classification and Disease Prediction"))
        self.Browesredimage.setText(_translate("MainWindow", "Red Image"))
        self.prediction.setText(_translate("MainWindow", "Prediction"))
        self.Browesblueimage.setText(_translate("MainWindow", "Blue Image"))
        self.Browesyellowimage.setText(_translate("MainWindow", "Yellow image"))
        self.Browesgreenimage.setText(_translate("MainWindow", "Green Image"))
        self.lineEdit.setText(_translate("MainWindow", "PROTEIN IMAGE CLASSIFICATION AND DISEASE PREDICTION"))
    def get_blue_image(self):
        fileName , _ = QtWidgets.QFileDialog.getOpenFileName(None,'Chose Blue image','','*_blue.png') 
        self.blueimage.setPixmap(QPixmap(fileName))
    def get_red_image(self):
        fileName , _ = QtWidgets.QFileDialog.getOpenFileName(None,'Chose Blue image','','*_red.png') 
        self.readimage.setPixmap(QPixmap(fileName))
    def get_green_image(self):
        fileName , _ = QtWidgets.QFileDialog.getOpenFileName(None,'Chose Blue image','','*_green.png') 
        self.greenimage.setPixmap(QPixmap(fileName))
    def get_yellow_image(self):
        fileName , _ = QtWidgets.QFileDialog.getOpenFileName(None,'Chose Blue image','','*_yellow.png') 
        self.yellowimage.setPixmap(QPixmap(fileName))
        path='C:'
        # print(p.split('/'))
        for i in fileName.split('/')[1:-1]:
            # print(i)
            path=path+"\\"+i
        root=path
        print(root)
        # imgpath=os.listdir(root)
        for i in os.listdir(root):
            test=i.split("_")
            submit.append(test[0])
        # =fileName.split("/")[:-1]
    def predictions(self):
        predicted = []
        for name in tqdm(submit):
            path = os.path.join('test/', name)
            image = data_generator.load_image(path, INPUT_SHAPE)
            score_predict = MODEL.predict(image[np.newaxis])[0]
            label_predict = np.arange(28)[score_predict>=0.2]
            str_predict_label = ' '.join(str(l) for l in label_predict)
            predicted.append(str_predict_label)

        predicted=list(set(map(int,predicted)))

        print(predicted)
        result=""
        for i in predicted:
            result+=label[i] + "\n" 
        print(result)

        self.resultarea.setText(result)
        with open("disease.json") as f:
            data=json.load(f)
        result=""
        for i in predicted:
            result+=data[list(data.keys())[i]] + "\n" 
        print(result)
        self.disease.setText(result)
        # print()
        # print(data[list(data.keys())[0]])


# import back_rc
# import imageresource_rc

if __name__ == "__main__":
    import sys
    print("Model laoding")
    MODEL=load_models()
    print("MOdel loaded successfully")
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

