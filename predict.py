import os
import glob
import numpy as np
from PIL import Image
from ModelDetector import *
import random
TEST_DATA_DIRECTORY = './data/train/**/*'

# create a model
detector = ModelDetector()
detector.model.load_weights("./trained/model_detector.h5")

paths = glob.glob('./data/train/**')
paths = sorted(paths)

# setup data
classes = list(map(lambda x: x.split("/")[-1], paths))

data = glob.glob(TEST_DATA_DIRECTORY)
random.shuffle(data)

correct = 0
index = 0
for image_path in glob.glob(TEST_DATA_DIRECTORY):
    answer = image_path.split("_")[-1].split(".")[0]
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img, dtype=np.float64)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    probabilities = detector.model.predict(img)
    prediction_classes = probabilities[0].argsort()[-5:][::-1]

    prediction = prediction_classes[0]
    if answer in classes[prediction]:
        correct += 1
    index += 1
    if index % 500 == 0 and index > 0:
        print("accuray is ", correct / index)

