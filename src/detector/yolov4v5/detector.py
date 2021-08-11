import torch

from detector.yolov4v5.models.experimental import attempt_load
from detector.yolov4v5.utils import torch_utils
from detector.yolov4v5.utils.utils import non_max_suppression, scale_coords
from detector.yolov4v5.utils.datasets import *

class YoloModel(object):
    def __init__(self, use_cuda = False, weight = None, classify = False, *args, **kwargs):
        self.device = 'cuda' if use_cuda else "cpu"
        assert weight != None, "The weight path need to be passed in!!!"
        self.model = attempt_load(weight, map_location=self.device)
        # self.model.half() 

        # if classify:
        #     modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
        #     modelc.to(self.device).eval()

    def __call__(self, image):
        img = letterbox(image, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=None)[0]

        pred = non_max_suppression(pred, 0.6, 0.6)
        if pred[0] is not None and len(pred[0]):
            pred[0][:,:4] = scale_coords(img.shape[2:], pred[0][:,:4], image.shape).round()
            pred = pred[0].cpu().numpy()

            bbox_xywh = pred[:,:4]
            bbox_xywh[:,2:] -= bbox_xywh[:,:2]
            bbox_xywh[:,:2] += bbox_xywh[:,2:] / 2

            cls_ids = pred[:,5]
            cls_conf = pred[:,4]

            return bbox_xywh, cls_conf, cls_ids
        
        # else:
        #     return np.array([]), np.array([]), np.array([])

def build_ultralityc_yolo_detector(weight, use_cuda=False):
    return YoloModel(use_cuda=use_cuda, weight = weight)