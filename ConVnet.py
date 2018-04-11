# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 18:11:44 2018

@author: Vaibhav
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt

classifier=Sequential()

classifier.add(Convolution2D(16,kernel_size=(3,3),padding='same',activation='relu',input_shape=(128,128,3)))

classifier.add((MaxPooling2D(pool_size=(2,2),strides = 2)))

#adding a second convolution layer

classifier.add(Convolution2D(32,kernel_size=(3,3),padding='same',activation='relu')) #no need of input shape as its not the first layer

classifier.add((MaxPooling2D(pool_size=(2,2),strides = 2)))

classifier.add(Convolution2D(64,kernel_size=(3,3),padding='same',activation='relu'))

classifier.add(MaxPooling2D(pool_size = (2,2), strides = 2))

classifier.add(Convolution2D(128,kernel_size=(3,3),padding ='same',activation ='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2), strides = 2))
classifier.add(Flatten())

classifier.add(Dropout(rate = 0.4))

classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dropout(rate = 0.2))
classifier.add(Dense(units=64,activation='relu'))
classifier.add(Dropout(rate = 0.2))
classifier.add(Dense(units = 16,activation= 'relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

classifier.summary()

classifier.compile(optimizer='adagrad',loss='binary_crossentropy',metrics=['accuracy'])


#Image Preprocessing
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        height_shift_range=0.1,
        width_shift_range=0.1,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set= train_datagen.flow_from_directory(
        'dataset/training_set/mod',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set/mod',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

hist = classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data =test_set,
        validation_steps=2000)

plt.plot(hist.history['loss'],'r')
plt.plot(hist.history['val_loss'],'b')
plt.plot(hist.history['acc'],'r')
plt.plot(hist.hsitory['val_acc'],'r')