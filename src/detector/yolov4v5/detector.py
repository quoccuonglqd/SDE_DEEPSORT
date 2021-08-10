import torch

from models.experimental import attempt_load
from utils import torch_utils
from utils.utils import non_max_suppression

class YoloModel(object):
    def __init__(self, use_cuda = False, weight = None, classify = False, *args, **kwargs):
        self.device = 'cuda' if use_cuda else "cpu"
        assert weight != None, "The weight path need to be passed in!!!"
        self.model = attempt_load(weight, map_location=self.device)

        # if classify:
        #     modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
        #     modelc.to(self.device).eval()

    def __call__(self, image):
        img = torch.from_numpy(image).to(self.device)
        img /= 255.0

        pred = self.model(img, augment=None)[0]
        pred = non_max_suppression(pred, 0.8, 0.6)[0]

        bbox_xywh = pred[:,:4]
        bbox_xywh[:,2:] -= bbox_xywh[:,:2]
        bbox_xywh[:,:2] += bbox_xywh[:,2:] / 2

        cls_ids = pred[:,5]
        cls_conf = pred[:,4]

        return bbox_xywh, cls_conf, cls_ids

def build_ultralityc_yolo_detector(weight, use_cuda=False):
    return YoloModel(use_cuda=use_cuda, weight = weight)