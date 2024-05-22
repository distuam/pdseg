
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


from deep_learning.dp_support import download_pretrained_model


classification_model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


segmentation_model_urls = {
    'resnet50': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'resnet101': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth'
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()


        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=planes*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                block,
                layers,
                num_classes=1000,
                groups=1,
                width_per_group=64,
                replace_stride_with_dilation=None,
                norm_layer = None):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks=layers[0])
        self.layer2 = self._make_layer(block, 128, blocks=layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, blocks=layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, blocks=layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, model, pretrained=False, **kwargs):
    if pretrained:
        
        model_path_name = download_pretrained_model(classification_model_urls[arch])
        model.load_state_dict(torch.load(model_path_name, map_location='cpu'))
        # load_office_module_state(model=model, save_path=model_path_name)
        # if kwargs["task_name"] == "cls":
        #     in_features = model.fc.in_features
        #     model.fc = torch.nn.Linear(in_features, kwargs["num_classes"])
        # else:
        #     model = torch.nn.Sequential(*list(model.children())[:-2])

    return model


def ResNet_18(num_classes=1000):
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)


def ResNet_34(num_classes=1000):
    return ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes)


def ResNet_50(pretrained: bool = False , num_classes=1000, replace_stride_with_dilation = [False, True, True]):
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, replace_stride_with_dilation=replace_stride_with_dilation)
    return _resnet(arch="resnet50",
                   model = model,
                   pretrained = pretrained,
                   num_classes = num_classes)


def ResNet_101(pretrained: bool = False , num_classes=1000, replace_stride_with_dilation = [False, True, True]):
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes, replace_stride_with_dilation=replace_stride_with_dilation)
    return _resnet(arch="resnet101",
                model = model,
                pretrained = pretrained,
                num_classes = num_classes)


def ResNet_152(num_classes=1000):
    return ResNet(block=Bottleneck, layers=[3, 8, 36, 3], num_classes=num_classes)



# if __name__ == "__main__":
#     # net = ResNet_18()
#     # net = ResNet_34()
#     # net = ResNet_50()
#     # net = ResNet_101()
#     net = ResNet_50(num_classes=20)
#     # print(net)
#     summary(net, input_size=(1, 3, 480, 480))
