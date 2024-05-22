
from collections import OrderedDict
from typing import Dict, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchinfo import summary
from deep_learning.hander import segmentation_hander
from deep_learning.classify.resnet import ResNet_50, ResNet_101
from deep_learning.dp_support import IntermediateLayerGetter, download_pretrained_model, load_module
from deep_learning.hander.classification_header import ClassificationHead
segmentation_model_urls = {
    'resnet50': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'resnet101': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}

class DeepLabV3(nn.Module):
    __constants__ = ['aux_classifier']
    def __init__(self, backbone, classifier, aux_classifier=None, classification_header = None):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.classification_header = classification_header

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        #返回值是result["out"]，如果有aux_classifier，再返回 result["aux"]。
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        
        if self.classification_header != None:
            result["cls"] = self.classification_header(x)


        x = self.classifier(x)
        # 使用双线性插值还原回原图尺度
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier != None:
            x = features["aux"]
            x = self.aux_classifier(x)
            # 使用双线性插值还原回原图尺度
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x
      
        return result
        # return result



class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        ]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )



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

 
def create_deeplab_model(backbone_type, aux, num_classes, pretrained=False, classification_task=False):
    """
    Create a FCN model based on a specified backbone.
    """
    print("Backbone_type:",backbone_type)
    if backbone_type == 'resnet50':
        backbone = ResNet_50(pretrained=pretrained, replace_stride_with_dilation=[False, True, True])
    elif backbone_type == 'resnet101':
        backbone = ResNet_101(pretrained=pretrained, replace_stride_with_dilation=[False, True, True])
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    out_inplanes = 2048
    aux_inplanes = 1024

    classifier = DeepLabHead(out_inplanes, num_classes)

    aux_classifier = segmentation_hander.FCNHead(aux_inplanes, num_classes) if aux else None

    classification_header = ClassificationHead(out_inplanes, num_classes) if classification_task else None
    model = DeepLabV3(backbone, classifier, aux_classifier, classification_header)

    if pretrained:
        load_pretrained_weights(model, segmentation_model_urls[backbone_type], num_classes)
    return model

# def deeplabv3_resnet50(aux,  num_classes: int, pretrained: bool):
#     # :param aux: 是否有辅助损失
#     # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
#     # 'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth'
#     backbone = resnet.ResNet_50(pretrained=pretrained, replace_stride_with_dilation=[False, True, True])
#     # print(backbone)
#     out_inplanes = 2048
#     aux_inplanes = 1024

#     return_layers = {'layer4': 'out'}
#     if aux:
#         return_layers['layer3'] = 'aux'
#     backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)# 返回的是一个字典 backbone输出{'out':layer4输出, 'aux':layer3输出}

#     aux_classifier = None
#     # why using aux: https://github.com/pytorch/vision/issues/4292
#     if aux:
#         aux_classifier = FCNHead(aux_inplanes, num_classes)

#     classifier = DeepLabHead(out_inplanes, num_classes)

#     model = DeepLabV3(backbone, classifier, aux_classifier)

#     return model

# def create_deepLeb_resnet50_model(aux, num_classes,pretrained=False):
#     model = deeplabv3_resnet50(aux=aux, num_classes=num_classes, pretrained = pretrained)                          # 创建模型
 
#     if pretrained:
    
#         model_path_name = download_pretrained_model(model_urls["resnet50"])
#         weights_dict = load_module(model_path_name)    # 字典模式导入预训练权重,还未载入模型，方便先判断是否删除某些权重
 
#         """ # 官方提供的预训练权重是21类(包括背景)
#             # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错"""
#         if num_classes != 21:                                                       # 官方提供的预训练权重是21类(包括背景)
#             for k in list(weights_dict.keys()):                                     # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
#                 if "classifier.4" in k:                                             # classifier.4是关于类别先关的权重，删除之
#                     del weights_dict[k]
 
#         missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)   # 载入权重
#         if len(missing_keys) != 0 or len(unexpected_keys) != 0:
#             print("missing_keys: ", missing_keys)
#             print("unexpected_keys: ", unexpected_keys)

#     return model