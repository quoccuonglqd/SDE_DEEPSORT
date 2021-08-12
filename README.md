# Deep Sort with PyTorch

<img src="demo/faster.gif" width="350" height="350" />

**Hình 1** Faster RCNN + DeepSort

<img src="demo/yolov3.gif" width="350" height="350" />

**Hình 2** YoloV3 + DeepSort

<img src="demo/yolov4.gif" width="350" height="350" />

**Hình 3** YoloV4 + DeepSort

## Introduction
This is the implementation of object tracking algorithm DeepSort. We intend to combine some popular object detection frameworks together so that the trained models can easily be adapted to use with DeepSort.

The implementation of detection frame is inherited directly from [DeepSort](https://github.com/ZQPei/deep_sort_pytorch), [Detectron2](https://github.com/facebookresearch/detectron2), [Yolov4 Pytorch](https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/u5), [Official Yolov5](https://github.com/ultralytics/yolov5)

## Installation
0. Clone repository
```bash
git clone https://github.com/quoccuonglqd/SDE_DEEPSORT.git
cd SDE_DEEPSORT
```

1. Install Deepsort dependencies
```bash
pip install -r requirements.txt
```

2. Install Detectron2 dependencies
```bash
pip install -U torch torchvision
pip install git+https://github.com/facebookresearch/fvcore.git
pip install -e detectron2master
```

3. Install Ultralytic-based Yolo dependencies and
```bash
pip install -r src/detector/yolov4v5/requirements.txt
pip install git+https://github.com/thomasbrandon/mish-cuda
```

## Quick Start
1. Download YOLOv3 parameters
```
cd src/detector/YOLOv3/weight/
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
cd ../../../
```

2. Download deepsort parameters ckpt.t7
```
cd src/deep_sort/deep/checkpoint
# download ckpt.t7 from
https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
cd ../../../
```  

3. Compile nms module
```bash
cd src/detector/YOLOv3/nms
sh build.sh
cd ../../..
```

Notice:
If compiling failed, the simplist way is to **Upgrade your pytorch >= 1.1 and torchvision >= 0.3" and you can avoid the troublesome compiling problems which are most likely caused by either `gcc version too low` or `libraries missing`.

4. Run demo
```
usage: python src/deepsort_runner.py --INPUT_TYPE imageset
                                     --input_path VIDEOPATH
                                     --output     OUTPUTPATH
                                     --config_deepsort configs/deep_sort.yaml    
                                     --config_detection configs/yolov3.yaml 
```


## Training the RE-ID model
The original model used in paper is in original_model.py, and its parameter here [original_ckpt.t7](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6).  

To train the model, first you need download [Market1501](http://www.liangzheng.com.cn/Project/project_reid.html) dataset or [Mars](http://www.liangzheng.com.cn/Project/project_mars.html) dataset.  

Then you can try [train.py](deep_sort/deep/train.py) to train your own parameter and evaluate it using [test.py](deep_sort/deep/test.py) and [evaluate.py](deep_sort/deep/evalute.py).
![train.jpg](deep_sort/deep/train.jpg)

