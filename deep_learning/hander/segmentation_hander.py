from torch import nn

class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]
        super(FCNHead, self).__init__(*layers)

class FCNVGGHead(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FCNVGGHead, self).__init__()
        self.fc6 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.fc6(x)
        x = self.fc7(x)
        return x



 