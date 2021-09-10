from seldon_core.seldon_client import SeldonClient
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
test_image = x_test[0].reshape(1,28,28,-1).tolist()

endpoint = "127.0.0.1:9001"

sc = SeldonClient(microservice_endpoint=endpoint)
response = sc.microservice(
   json_data = test_image,
   method='predict'
)

print(response.response)
