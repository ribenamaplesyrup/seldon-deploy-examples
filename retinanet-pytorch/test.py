from seldon_core.seldon_client import SeldonClient
from PIL import Image
import numpy as np

endpoint = "127.0.0.1:9001"

image = Image.open("input/image2.jpg").convert('RGB')
# a NumPy copy for OpenCV functions
image_array = np.array(image)

X = image_array

sc = SeldonClient(microservice_endpoint=endpoint)
response = sc.microservice(
   json_data = X,
   method='predict'
)

print(response.response)
