from .parallel import *
from .defaults import *

__all__ = [
    "FormatedPredictor",
]

class FormatedPredictor(object):
    def __init__(self, parallel = False, config = None):
        if parallel:
            self.predictor = AsyncPredictor(config)
        else:
            self.predictor = DefaultPredictor(config)
    
    def __call__(self, image):
        outputs = self.predictor(image)
        bbox_ltrb = outputs['instances'].pred_boxes.tensor.cpu().numpy().tolist()
        cls_ids = outputs['instances'].pred_classes.cpu().numpy().tolist()
        cls_conf = outputs['instances'].scores.cpu().numpy().tolist()
        bbox_xywh = bbox_ltrb.copy()
        bbox_xywh[2:] -= bbox_xywh[:2]
        bbox_xywh[:2] += bbox_xywh[2:] / 2
        return bbox_xywh, cls_conf, cls_ids