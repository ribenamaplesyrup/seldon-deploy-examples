from tensorflow import keras
import logging
import numpy as np
from typing import Dict, List, Union, Iterable

logger = logging.getLogger(__name__)

class MnistClassifier(object):
    def __init__(self):
        self.ready = False

    def load(self):
        logger.info("Loading...")
        self._model = keras.models.load_model("MnistClassifier")
        self._model.make_predict_function()
        self.ready = True

    def predict(self, X: np.ndarray, names: Iterable[str] = None, meta: Dict = None) -> Union[np.ndarray, List, str, bytes]:
        try:
            if not self.ready:
                self.load()

            logger.info("Calling predict_proba...")
            return self._model.predict(X)

        except Exception as ex:
            logging.exception("Exception during predict!")
            logging.exception(f"{ex}")
