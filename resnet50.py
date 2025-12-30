import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Bottleneck Block
# =========================
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        mid = out_ch

        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)

        self.conv2 = nn.Conv2d(
            mid, mid, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid)

        self.conv3 = nn.Conv2d(
            mid, out_ch * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)

        self.downsample = nn.Identity()
        if stride != 1 or in_ch != out_ch * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch * self.expansion, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_ch * self.expansion),
            )

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + self.downsample(x))


# =========================
# ResNet50
# =========================
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = [Bottleneck(in_ch, out_ch, stride)]
        in_ch = out_ch * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(in_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# =========================
# Builder
# =========================
def build_resnet50(num_classes: int):
    return ResNet50(num_classes)
