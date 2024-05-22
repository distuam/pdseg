
# import sys
# sys.path.append("../../")

from collections import OrderedDict
from typing import Dict
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from deep_learning.hander import segmentation_hander
from deep_learning.classify.resnet import ResNet_50, ResNet_101, segmentation_model_urls
from deep_learning.dp_support import IntermediateLayerGetter, download_pretrained_model, load_module
from deep_learning.hander.classification_header import ClassificationHead
from deep_learning.classify.vgg import vgg16,vgg19
from torch import flatten
#https://blog.csdn.net/weixin_43500354/article/details/124867082

class VGGFCN(nn.Module):
    def __init__(self, num_classes: int, backbone, stride : int = 32, classification_task: bool = False):
        super(VGGFCN, self).__init__()
        self.backbone = backbone
        self.stride = stride

        self.classification_task = classification_task
        if classification_task:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classification_header = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        
        self.header = segmentation_hander.FCNVGGHead(in_channel=512,out_channel=4096)

        self.layer3_conv = nn.Conv2d(256,num_classes,kernel_size=1)    #256
        self.layer4_conv = nn.Conv2d(512,num_classes,kernel_size=1)    #512
        self.layer5_conv = nn.Conv2d(4096,num_classes,kernel_size=1)

        self.transpose_conv1 =nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU()
        )
        # 转置卷积层2
        self.transpose_conv2 = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU()
        )
        # 转置卷积层3
        # self.transpose_conv3 = nn.Sequential(
        #     nn.ConvTranspose2d(num_classes, num_classes,kernel_size=16,stride=8,padding=4),
        #     nn.BatchNorm2d(num_classes),
        #     nn.ReLU()
        # )
        self.transpose_conv3 = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=4, padding=2),  # Adjusted kernel size and stride
            nn.BatchNorm2d(num_classes),
            nn.ReLU()
        )


  
    def forward(self, x):
        result = OrderedDict()  
        output = self.backbone(x)
   
        layer5_out, layer4_out, layer3_out = output['stage4'], output['stage3'], output['stage2']

        if self.classification_task:
            cls_x = self.avgpool(layer5_out)
            cls_x = flatten(cls_x, 1)
            result["cls"] = self.classification_header(cls_x)
            # print(result["cls"].size())

        layer5_out = self.header(layer5_out)
        layer5_out = self.layer5_conv(layer5_out)
        layer4_out = self.layer4_conv(layer4_out)
        layer3_out = self.layer3_conv(layer3_out)

 
        x = self.transpose_conv1(layer5_out)
        
        x = self.transpose_conv2(x + layer4_out) # x.shape Size is ([20, 21, 96, 140]) 
        # print(x.size())
        x = self.transpose_conv3(x + layer3_out)
        # print(x.size())
        result["out"] = x
        return result

class FCN(nn.Module):
    __constants__ = ['aux_classifier']
 
    def __init__(self, backbone, classifier, aux_classifier=None, classification_header = None):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.classification_header = classification_header
  

 
    def forward(self, x: Tensor) -> Dict[str, Tensor]:                                          # x为输入的数据
        input_shape = x.shape[-2:]                                                              # tensor的最后两个维度即为图片的宽高
        # contract: features is a dict of tensors
        features = self.backbone(x)                                                             # feature为字典
 
        result = OrderedDict()                                                                  # 创建字典，方便统计最终输出
        x = features["out"]

        if self.classification_header != None:
            result["cls"] = self.classification_header(x)

        x = self.classifier(x)                                                                  # 将out输出提取并送入主分类器中
        # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)            # 通过双线性插值还原到原少shape
        result["out"] = x                                                                       # 结果存到字典中
 
        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x                                                                   # 辅助分类器的结果

        return result
 




def load_pretrained_weights(model, model_url, num_classes):
    """
    Load and modify pretrained weights as needed for different number of classes.
    """
    model_path_name = download_pretrained_model(model_url)
    weights_dict = load_module(model_path_name)
    if num_classes != 21:
        for k in list(weights_dict.keys()):
            if "classifier.4" in k:
                del weights_dict[k]
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("Missing keys: ", missing_keys)
        print("Unexpected keys: ", unexpected_keys)

 
def create_fcn_model(backbone_type, aux, num_classes, pretrained=False, classification_task=False, batch_norm=True):
    """
    Create a FCN model based on a specified backbone.
    """
    print("Backbone_type:",backbone_type)
    if backbone_type == 'resnet50':
        backbone = ResNet_50(pretrained=pretrained, replace_stride_with_dilation=[False, True, True])
    elif backbone_type == 'resnet101':
        backbone = ResNet_101(pretrained=pretrained, replace_stride_with_dilation=[False, True, True])
    elif backbone_type == 'vgg16':
        stage_indices = [5, 12, 22, 32, 42]
        backbone = vgg16(pretrained=pretrained, batch_norm = batch_norm)
    elif backbone_type == 'vgg19':
        stage_indices = [5, 12, 25, 38, 51]
        backbone = vgg19(pretrained=pretrained, batch_norm = batch_norm)
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")

    
    if backbone_type in ['resnet50', 'resnet101']:
        return_layers = {'layer4': 'out'}
        if aux:
            return_layers['layer3'] = 'aux'
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        out_inplanes = 2048
        aux_inplanes = 1024

        classifier = segmentation_hander.FCNHead(out_inplanes, num_classes)
        aux_classifier = segmentation_hander.FCNHead(aux_inplanes, num_classes) if aux else None
        classification_header = ClassificationHead(out_inplanes, num_classes) if classification_task else None #是否添加分类头
        model = FCN(backbone, classifier, aux_classifier, classification_header)

        if pretrained:
            load_pretrained_weights(model, segmentation_model_urls[backbone_type], num_classes)
    elif backbone_type in ["vgg16", "vgg19"]:
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        backbone = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
        model = VGGFCN(num_classes=num_classes,
                    backbone = backbone, 
                    stride  = 8,
                    classification_task = classification_task)

    return model

# if __name__ == "__main__":
#     create_fcn_model(backbone_type="vgg19", num_classes=32, pretrained = True, batch_norm=True , classification_task = True, aux = False)