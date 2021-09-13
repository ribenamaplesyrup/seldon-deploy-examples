import logging
import numpy as np
import torch
import torchvision
import detect_utils
import os
from typing import Dict, List, Union, Iterable

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ObjectDetector(object):
    def __init__(self):
        self.ready = False

    def load(self):
        logger.info("Loading...")
        os.environ['TORCH_HOME'] = 'mnt'
        self._model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
        self._model.eval().to(device)
        self.ready = True

    def predict(self, X: np.ndarray, names: Iterable[str] = None, meta: Dict = None) -> Union[np.ndarray, List, str, bytes]:
        try:
            if not self.ready:
                self.load()

            logger.info("Calling predict")
            X = X.astype(np.uint8)
            boxes, classes = detect_utils.predict(X, self._model, device, 0.7)
            return [boxes.tolist(), classes]

        except Exception as ex:
            logging.exception("Exception during predict!")
            logging.exception(f"{ex}")
