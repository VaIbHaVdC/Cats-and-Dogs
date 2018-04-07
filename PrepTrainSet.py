# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 18:27:14 2018

@author: Vaibhav
"""

import cv2 as cv
import glob

#Preprocessing cats images

images_cat = [cv.imread(file,0) for file in glob.glob("dataset/training_set/cats/*.jpg")]

modified_cat =[]
for i in range(0,len(images_cat)) :
    blur = cv.GaussianBlur(images_cat[i],(5,5),0)
    th3 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                           cv.THRESH_BINARY,11,2)    
    modified_cat.append(th3)

#Writing preprocessed cats images on disk
    
i=0
for j in range(0,len(modified_cat)):
    cv.imwrite('dataset/training_set/modcat/pic{:>05}.jpg'.format(i), modified_cat[j])
    i += 1
    
#Preprocessing dogs images
    
images_dog = [cv.imread(file,0) for file in glob.glob("dataset/training_set/dogs/*.jpg")]

modified_dog =[]
for i in range(0,len(images_dog)) :
    blur = cv.GaussianBlur(images_dog[i],(5,5),0)
    th3 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                           cv.THRESH_BINARY,11,2)    
    modified_dog.append(th3)

#Writing preprocessed dogs images on disk
    
i=0
for j in range(0,len(modified_dog)):
    cv.imwrite('dataset/training_set/moddog/pic{:>05}.jpg'.format(i), modified_dog[j])
    i += 1