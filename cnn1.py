# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:29:38 2020

@author: Dibya
"""

from keras.preprocessing.image  import ImageDataGenerator
from keras.models import Sequential
from keras.layers import convolutional
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten
#import matplotlib.pyplot as plt

#build a classifier
classifier = Sequential()

#create a convolution layer
classifier.add(convolutional.Conv2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))

#add a max pooling layer
classifier.add(MaxPooling2D(pool_size = (2,2)))

#add another convolution layer
classifier.add(convolutional.Conv2D(32,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#flattening
classifier.add(Flatten())

#add hidden layer
classifier.add(Dense(units = 128,activation = 'relu'))
classifier.add(Dense(units = 1,activation = 'sigmoid'))

#compile
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

#fit the images to the classifier
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

history = classifier.fit_generator(
        training_set,
        epochs=25,
        validation_data=test_set)
        

#accuracy: 0.8924 - val_loss: 0.7082 - val_accuracy: 0.7445 - 1 convlutional layer
#accuracy: 0.9107 - val_loss: 0.5103 - val_accuracy: 0.8030 - 2 convlutional layer

# Saving the model
'''model_json = classifier.to_json()
with open("model.json", "w") as json_file :
	json_file.write(model_json)

classifier.save_weights("model.h5")
print("Saved model to disk")

classifier.save('CNN.model')'''

# Printing a graph showing the accuracy changes during the tr'aining 'phase
'''print(history.history.keys())
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')'''



