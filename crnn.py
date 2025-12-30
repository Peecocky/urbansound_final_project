import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# CRNN Model
# =========================

class AudioCRNN(nn.Module):
    """
    Input: (B, 3, 128, T)
    """
    def __init__(self, num_classes: int):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),   # (32, 64, T/2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),   # (64, 32, T/4)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),   # (128, 16, T/4)
        )

        cnn_out_freq = 128 // 8   # =16
        rnn_input = 128 * cnn_out_freq

        self.gru = nn.GRU(
            input_size=rnn_input,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, 3, F, T)
        x = self.cnn(x)                 # (B, C, F', T')
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)       # (B, T, C, F)
        x = x.contiguous().view(B, T, C * F)

        x, _ = self.gru(x)              # (B, T, 512)

        att = self.attention(x)         # (B, T, 1)
        att = torch.softmax(att, dim=1)
        x = (att * x).sum(dim=1)        # (B, 512)

        return self.classifier(x)


# =========================
# Builder (VERY IMPORTANT)
# =========================

def build_crnn(num_classes: int):
    return AudioCRNN(num_classes)
