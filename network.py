import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import RMSprop

"""
cz = 


"""
train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale= 1/255)

train_dataset = train.flow_from_directory("",
                                          target_size=(200,200),
                                          batch_size= 3,
                                          class_mode= "binary")
validation_dataset = validation.flow_from_directory("",
                                          target_size=(200,200),
                                          batch_size= 3,
                                          class_mode= "binary")

model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3), activation = "relu",input_shape =(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3), activation = "relu"),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(64,(3,3), activation = "relu"),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512,activation= "relu"),
                                    ##
                                    tf.keras.layers.Dense(1,activation = "sigmoid")
                                    ])


model.compile(loss= "binary_crossentropy",
              optimizer= RMSprop(lr=0.001),
              metrics = ["accuracy"])
             #made by cyberft-pdf
model_fit = model.fit(train_dataset,
                      steps_per_epoch= 3,
                      epochs= 30,
                      validation_data= validation_dataset)

dir_path = ""


for i in os.listdir(dir_path):
    img = tf.keras.preprocessing.image.load_img(dir_path+ "//"+i, target_size=(200,200))
    plt.imshow(img)
    plt.show()

    X = tf.keras.preprocessing.image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])

    val = model.predict(images)
    if val == 0:
        print("")

    else:
        print("")

