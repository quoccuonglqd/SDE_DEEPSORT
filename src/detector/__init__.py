from .YOLOv3 import YOLOv3
from detectron2.engine import FormatedPredictor
from utils.parser import get_config
from detectron2.config import get_cfg

__all__ = ['build_yolov3_detector', 'build_detectron2_detector']

def build_yolov3_detector(cfg_path, use_cuda):
    cfg = get_config()
    cfg.merge_from_file(cfg_path)
    return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES, 
                    score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH, 
                    is_xywh=True, use_cuda=use_cuda)

def build_detectron2_detector(cfg_path, parallel=False):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    return FormatedPredictor(parallel=parallel, config = cfg)
