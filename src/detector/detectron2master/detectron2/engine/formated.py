from detectron2.config import get_cfg

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
        bbox_ltrb = outputs['instances'].pred_boxes.tensor.cpu().numpy()
        cls_ids = outputs['instances'].pred_classes.cpu().numpy()
        cls_conf = outputs['instances'].scores.cpu().numpy()
        bbox_xywh = bbox_ltrb.copy()
        bbox_xywh[:,2:] -= bbox_xywh[:,:2]
        bbox_xywh[:,:2] += bbox_xywh[:,2:] / 2
        return bbox_xywh, cls_conf, cls_ids

def build_detectron2_detector(cfg_path, parallel=False):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    return FormatedPredictor(parallel=parallel, config = cfg)