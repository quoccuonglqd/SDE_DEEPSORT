from os import listdir
import os.path as osp

import cv2 as cv

class ImageSetIter:
    def __init__(self, path):
        accepted_extension = ['.PNG','.JPG','.JPEG','.JPE','.TIFF','TIF',
            '.JP2','.SR','.RAS','.PBM','.PGM','.PPM','.BMP']
        self.data = [osp.join(path,filename) for filename in listdir(path) 
            if osp.splitext(filename)[1].upper() in accepted_extension]
        self.data.sort()
        self.im_height, self.im_width = cv.imread(self.data[0]).shape[:2]

    def __getitem__(self, index):
        return cv.imread(self.data[index])

    def __len__(self):
        return len(self.data)

class VideoIter:
    def __init__(self,path):
        self.video = cv.VideoCapture(path)
        self.im_height = self.video.get(cv.CAP_PROP_FRAME_HEIGHT) 
        self.im_width = self.video.get(cv.CAP_PROP_FRAME_WIDTH)
    def __getitem__(self, index):
        _, img = self.video.read()
        return img

class WebcamIter:
    def __init__(self):
        self.cam = cv.VideoCapture(0)
        self.im_height = self.cam.get(cv.CAP_PROP_FRAME_HEIGHT) 
        self.im_width = self.cam.get(cv.CAP_PROP_FRAME_WIDTH)
    def __iter__(self):
        return self
    def __next__(self):
        _, img = self.cam.read()

        cv.imshow('Camera', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows
            raise StopIteration
        
        return img