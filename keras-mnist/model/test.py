from MnistClassifier import MnistClassifier
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
test_image = x_test[0].reshape(1,28,28,-1)

X = {
    "data": {
        "ndarray":
        [test_image]
    }
}

result = MnistClassifier().predict(X['data']['ndarray'])
print(result)
