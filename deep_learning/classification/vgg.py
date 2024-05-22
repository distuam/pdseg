from deep_learning.dp_support import download_pretrained_model, load_module
from torch import flatten, nn
import torch

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


# make layers using Vgg-Net config(cfg)
# 由cfg构建vgg-Net
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, features, ranges ,num_classes=1000, init_weights=True, classification_task = False):
        super(VGG, self).__init__()
        self.classification_task = classification_task
        self.features = features
        self.ranges = ranges

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        output = {}

        for idx, (begin, end) in enumerate(self.ranges):
        #self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)) (vgg16 examples)
            for layer in range(begin, end):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x
        # output 为一个字典键x1d对应第一个maxpooling输出的特征图，x2...x5类推
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def _vgg(arch, pretrained, batch_norm=False, classification_task = False):
    if pretrained:
        init_weights = False
    else:
        init_weights = True
    model = VGG(features=make_layers(cfgs[arch], batch_norm=batch_norm), ranges=ranges[arch], init_weights=init_weights, classification_task=classification_task)
    if pretrained:
        model_path_name = download_pretrained_model(model_urls[arch + "_bn"])
        pretrained_dict  = load_module(save_path=model_path_name)

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        # 3. Load the new state dict

        model.load_state_dict(model_dict)
    return model

def vgg16(pretrained=False, batch_norm = False, classification_task=False):
    return _vgg('vgg16', pretrained, batch_norm, classification_task = classification_task)

def vgg19(pretrained=False, batch_norm = False, classification_task=False):
    return _vgg('vgg19', pretrained, batch_norm, classification_task = classification_task)

# if __name__ == '__main__':
#     import sys
#     sys.path.append("../")
#     from torchsummary import summary
#     # from deep_learning.dp_support import *
#     from dp_support import *
#     device = pytorch_test()
#     model = vgg16().to(device)
#     # print(model)
#     # img = torch.rand(1, 3, 448, 448).to(device)
#     # y = model(img)
#     summary(model, input_size=(3, 224, 224))