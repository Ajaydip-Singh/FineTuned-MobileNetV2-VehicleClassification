import os
import random

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from ModelDetector import *


TRAIN_DATA_DIRECTORY = './vehicles'
RANDOM_SEED = random.randint(1, 1000)

# create a model
detector = ModelDetector()

# setup data
train_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train_augumented_generator = train_data_generator.flow_from_directory(TRAIN_DATA_DIRECTORY,
                                                                      target_size=(224, 224),
                                                                      batch_size=64,
                                                                      shuffle=True,
                                                                      seed=RANDOM_SEED,
                                                                      class_mode='categorical',
                                                    subset='training')

validation_augumented_generator = train_data_generator.flow_from_directory(TRAIN_DATA_DIRECTORY,
                                                                           target_size=(224, 224),
                                                                           batch_size=64,
                                                                           class_mode='categorical',
                                                                           shuffle=True,
                                                                           seed=RANDOM_SEED,
                                                                           subset='validation')

opt = Adam(lr=0.0001)
detector.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

step_size_train = train_augumented_generator.n / train_augumented_generator.batch_size
step_size_val = validation_augumented_generator.samples // validation_augumented_generator.batch_size

detector.model.fit_generator(generator=train_augumented_generator,
                             steps_per_epoch=step_size_train,
                             validation_data=validation_augumented_generator,
                             validation_steps=step_size_val,
                             epochs=35)

if not os.path.exists('./trained'):
    os.mkdir('./trained')

detector.model.save_weights('./trained/model_detector.h5')