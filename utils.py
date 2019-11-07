import cv2
import numpy as np


def imread(image_path):
    image = cv2.imread(image_path)
    image = _preprocessing_images([image])
    return image


def _preprocessing_images(images):
    resize_images = []
    for image in images:
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = np.array(image, dtype=np.float64)
        image = cv2.resize(image, (224, 224))
        image /= 255.0
        resize_images = np.asarray([image])
    return resize_images
