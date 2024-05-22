from torch import nn

class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),  # 展平层
            nn.Linear(in_channels, num_classes)  # 全连接层
        )