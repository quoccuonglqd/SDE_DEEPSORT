import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from pathlib import Path
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
# from detectron2_1.adv import DAGAttacker
from detectron2.structures import pairwise_iou, Boxes
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor, AsyncPredictor

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

from utils.InferrenceDataIter import *



class Tracker(object):
    def __init__(self, cfg, detect_cfg, args, path):
        self.cfg = cfg
        self.args = args
        self.video_path = path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.INPUT_TYPE == 'imageset': 
            self.sequence = ImageSetIter(path)
        elif args.INPUT_TYPE == 'webcam':
            self.sequence = WebcamIter()
        else:
            self.sequence = VideoIter(path)
        self.detector = DefaultPredictor(detect_cfg)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

    def __enter__(self):
        self.im_width = self.sequence.im_width
        self.im_height = self.sequence.im_height

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        # while self.vdo.grab():
        for ori_im in self.sequence:
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)


            # do detectron2 detection
            outputs = self.detector(im)
            bbox_ltrb = outputs['instances'].pred_boxes.tensor.cpu().numpy().tolist()
            cls_ids = outputs['instances'].pred_classes.cpu().numpy().tolist()
            cls_conf = outputs['instances'].scores.cpu().numpy().tolist()
            bbox_xywh = bbox_ltrb.copy()
            bbox_xywh[2:] -= bbox_xywh[:2]
            bbox_xywh[:2] += bbox_xywh[2:] / 2

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--INPUT_TYPE",choices= ['imageset','video','webcam'],help = 'Type of inputs')
    parser.add_argument("--input_path", type = str, default = '')
    # parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    # cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    
    detect_cfg = get_cfg()
    detect_cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    detect_cfg.DATALOADER.NUM_WORKERS = 2
    detect_cfg.MODEL.WEIGHTS = '/content/drive/MyDrive/vehicle_speed_estimation/Model weight/Model detection AIC team Khanh/Faster_RCNN/R_101_FPN_augmented_6k_proposal256.pth' 
    detect_cfg.SOLVER.IMS_PER_BATCH = 1                        
    detect_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  
    detect_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    detect_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    

    with Tracker(cfg, detect_cfg, args, path=args.input_path) as vdo_trk:
        vdo_trk.run()
