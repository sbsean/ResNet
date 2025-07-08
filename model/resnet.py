import torch.nn as nn


class BasicBlock(nn.Module):    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()                
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)        
        self.relu = nn.ReLU(inplace=True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None        
        self.downsample = downsample
        
    def forward(self, x):       
        i = x       
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            i = self.downsample(i)
                        
        x += i
        x = self.relu(x)
        
        return x


class Bottleneck(nn.Module):    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)        
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)        
        self.relu = nn.ReLU(inplace=True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, 
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None            
        self.downsample = downsample
        
    def forward(self, x):        
        i = x        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)        
        x = self.conv3(x)
        x = self.bn3(x)
                
        if self.downsample is not None:
            i = self.downsample(i)
            
        x += i
        x = self.relu(x)
    
        return x


class ResNetConfig:
    def __init__(self, block, n_blocks, channels):
        self.block = block
        self.n_blocks = n_blocks
        self.channels = channels


class ResNet(nn.Module):
    def __init__(self, config, output_dim, zero_init_residual=False):
        super().__init__()
                
        block, n_blocks, channels = config.block, config.n_blocks, config.channels
        self.in_channels = channels[0]            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, output_dim)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
    def get_resnet_layer(self, block, n_blocks, channels, stride=1):   
        layers = []        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels            
        return nn.Sequential(*layers)
        
    def forward(self, x):        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)        
        return x, h


class CIFAR10ResNet(nn.Module):
    """CIFAR-10에 최적화된 ResNet (32x32 입력용)"""
    def __init__(self, config, output_dim=10, zero_init_residual=False):
        super().__init__()
                
        block, n_blocks, channels = config.block, config.n_blocks, config.channels
        self.in_channels = channels[0]            
        assert len(n_blocks) == len(channels) == 4
        
        # CIFAR-10 최적화: 작은 이미지에 맞게 초기 레이어 수정
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # maxpool 제거 - 32x32에서는 불필요한 다운샘플링
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, output_dim)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
    def get_resnet_layer(self, block, n_blocks, channels, stride=1):   
        layers = []        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels            
        return nn.Sequential(*layers)
        
    def forward(self, x):        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # maxpool 없음
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)        
        return x


# ResNet 설정들
resnet50_config = ResNetConfig(
    block=Bottleneck,
    n_blocks=[3, 4, 6, 3],
    channels=[64, 128, 256, 512]
)

resnet18_config = ResNetConfig(
    block=BasicBlock,
    n_blocks=[2, 2, 2, 2],
    channels=[64, 128, 256, 512]
)

resnet34_config = ResNetConfig(
    block=BasicBlock,
    n_blocks=[3, 4, 6, 3],
    channels=[64, 128, 256, 512]
)

resnet101_config = ResNetConfig(
    block=Bottleneck,
    n_blocks=[3, 4, 23, 3],
    channels=[64, 128, 256, 512]
)

resnet152_config = ResNetConfig(
    block=Bottleneck,
    n_blocks=[3, 8, 36, 3],
    channels=[64, 128, 256, 512]
)


# 팩토리 함수들
def ResNet50_model(output_dim=10, cifar10_optimized=True):
    """ResNet50 모델 생성"""
    if cifar10_optimized:
        return CIFAR10ResNet(resnet50_config, output_dim=output_dim)
    else:
        return ResNet(resnet50_config, output_dim=output_dim)

def ResNet18_model(output_dim=10, cifar10_optimized=True):
    """ResNet18 모델 생성"""
    if cifar10_optimized:
        return CIFAR10ResNet(resnet18_config, output_dim=output_dim)
    else:
        return ResNet(resnet18_config, output_dim=output_dim)

def ResNet34_model(output_dim=10, cifar10_optimized=True):
    """ResNet34 모델 생성"""
    if cifar10_optimized:
        return CIFAR10ResNet(resnet34_config, output_dim=output_dim)
    else:
        return ResNet(resnet34_config, output_dim=output_dim)

def ResNet101_model(output_dim=10, cifar10_optimized=True):
    """ResNet101 모델 생성"""
    if cifar10_optimized:
        return CIFAR10ResNet(resnet101_config, output_dim=output_dim)
    else:
        return ResNet(resnet101_config, output_dim=output_dim)

def ResNet152_model(output_dim=10, cifar10_optimized=True):
    """ResNet152 모델 생성"""
    if cifar10_optimized:
        return CIFAR10ResNet(resnet152_config, output_dim=output_dim)
    else:
        return ResNet(resnet152_config, output_dim=output_dim)