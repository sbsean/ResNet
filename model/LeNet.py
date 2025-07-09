# model.py

import torch
from torch import nn


class LeNet(nn.Module):
    
    def __init__(self, drop_prob=0.5, num_classes=10, in_channels=3):
        super().__init__()
        
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),  # (32x32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (16x16)

            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # (16x16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (8x8)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (8x8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> (4x4)
        )
        
       
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))  # -> (2x2)

       
        self.classifier = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob * 0.5),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """He 초기화 (ReLU에 적합한 가중치 초기화)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)             
        x = self.adaptive_pool(x)        
        x = x.view(x.size(0), -1)        
        x = self.classifier(x)           
        return x
