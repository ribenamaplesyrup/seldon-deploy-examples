from ObjectDetector import ObjectDetector
from PIL import Image
import numpy as np

image = Image.open("../input/image2.jpg").convert('RGB')
image_array = np.array(image)

X = {
    "data": {
        "ndarray": image_array
    }
}

result = ObjectDetector().predict(X["data"]["ndarray"])
print(result)
