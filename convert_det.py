import torch
import torchvision
#import onnx
#import model_cls
import time
#import onnxruntime as ort
import numpy as np
from torch import nn
#from onnxruntime.datasets import get_example
from backbone import EfficientDetBackbone
import os
import sys

current_path = os.path.dirname(os.path.abspath(sys.argv[0]))

compound_coef = 0
#obj_list = ['thyroid']
obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef]

def load_efficientdet(path='efficientdet.pth'):
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list), onnx_export=True,
                                 ratios=anchor_ratios, scales=anchor_scales)
    model.backbone_net.model.set_swish(memory_efficient=False)
    model.load_state_dict(torch.load(path))
    model.requires_grad_(False)
    return model.eval()

efficient = load_efficientdet(path=os.path.join(current_path, 'weights/efficientdet-d0.pth'))


dummy_input = torch.randn((1, 3, input_size, input_size), dtype=torch.float32)


input_names = ['data']
output_names = ['output1', 'output2', 'output3', 'output4', 'output5', 'output6', 'output7', 'output8']
print('start to convert!!')

torch_out = torch.onnx.export(efficient, dummy_input, 'efficientdet-d0.onnx', export_params=True, verbose=True, input_names=input_names, output_names=output_names, opset_version=11)

#torch_out = torch.onnx.export(efficient, dummy_input, 'efficientdet-d0.onnx', export_params=True, verbose=True, input_names=input_names, opset_version=11)


#model = onnx.load('densenet201.onnx')
#onnx.checker.check_model(model)
#onnx.helper.printable_graph(model.graph)
#print(torch_out)
#torch.onnx.export(efficient, dummy_input, 'efficientdet.onnx', export_params=True, verbose=True, input_names=input_names, output_names=output_names)
