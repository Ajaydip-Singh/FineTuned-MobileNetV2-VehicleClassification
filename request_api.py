import requests
import json
import cv2
import glob
import numpy as np
import random
from utils import imread

test_data_pathname = './data/train/**/*'
request_url = 'http://localhost:81/v1/models/model_detector:predict'

with open('model.json') as file:
    class_infos = json.load(file)
paths = glob.glob('./data/train/**')
paths = sorted(paths)
classes = list(map(lambda x: x.split("/")[-1].split("\\")[-1], paths))

image_paths = glob.glob(pathname=test_data_pathname)
random.shuffle(image_paths)

for image_path in image_paths:
    # get image
    image = imread(image_path)

    data = {
        'instances': image.tolist()
    }
    data = json.dumps(data)

    # request
    response = requests.post(request_url, data=data)

    # parsing response
    response = response.json()['predictions'][0]
    response = np.asarray(response)
    response = np.argmax(response, axis=-1)
    prediction = classes[response]
    prediction = class_infos[prediction]

    answer = image_path.split("/")[-2]
    answer = class_infos[answer]

    if answer == prediction:
        print("Correct !")
        print(answer, " ==> ", prediction)
    else:
        print("Wrong !")
        print(answer, " ==> ", prediction)

    cv2.imshow("", cv2.imread(image_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
