import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# Emotion mapping: 0: Neutral, 1: Happy, 2: Sad, 3: Angry
_emotion_names = ['neutral', 'happy', 'sad', 'angry']

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class EmotionCNN(nn.Module):
    """Deep ResNet-style Emotion Classifier"""
    def __init__(self, num_classes=4):
        super(EmotionCNN, self).__init__()
        self.in_channels = 32
        
        # Initial Convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # ResNet Blocks
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.layer4 = self._make_layer(256, 2, stride=2)
        
        # Global Average Pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Classifier
        self.fc1 = nn.Linear(256, 128) # 128-dim emotion embedding
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes) # Logits

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, Mel_bins, Time)
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        
        embedding = F.relu(self.fc1(out))
        embedding_drop = self.dropout(embedding)
        logits = self.fc2(embedding_drop)
        
        return embedding, logits
