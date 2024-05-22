import os
import sys

sys.path.append("../")
from classify.resnet import ResNet_50
from classify.vgg import vgg16
from deep_learning.segmentation.fcn import create_fcn_model
# model = create_fcn_model(aux = False, pretrained = True, num_classes = 21)
model = vgg16(pretrained = True)
# model = ResNet_50(task_name="seg", pretrained = True, replace_stride_with_dilation=[False, True, True])
print(model)