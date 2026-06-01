import torch.nn.functional as F
import torch.nn as nn
import torch


class Net(nn.Module):
    """
    RedResidual para ImageNet (224x224).
    
    Arquitectura simplificada inspirada en ResNet pero optimizada para
    entrenamiento distribuido. Soporta 1000 clases de ImageNet.
    """
    
    def __init__(self, num_classes=1000, pretrained=False, freeze_backbone=False):
        super(Net, self).__init__()
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        if pretrained:
            from torchvision.models import resnet18, ResNet18_Weights
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
            if freeze_backbone:
                for name, param in self.backbone.named_parameters():
                    if "fc" not in name:
                        param.requires_grad = False
        else:
            # ─────────────────────────────────────────────────────────
            # BLOQUE INICIAL!
            # ─────────────────────────────────────────────────────────
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            # ─────────────────────────────────────────────────────────
            # BLOQUES RESIDUALES (Simplified ResNet)
            # ─────────────────────────────────────────────────────────
            
            # Layer 1: 64 canales, 56x56 spatial
            self.layer1 = self._make_residual_block(64, 64, num_blocks=3, stride=1)
            
            # Layer 2: 128 canales, 28x28 spatial
            self.layer2 = self._make_residual_block(64, 128, num_blocks=4, stride=2)
            
            # Layer 3: 256 canales, 14x14 spatial
            self.layer3 = self._make_residual_block(128, 256, num_blocks=6, stride=2)
            
            # Layer 4: 512 canales, 7x7 spatial
            self.layer4 = self._make_residual_block(256, 512, num_blocks=3, stride=2)
            
            # ─────────────────────────────────────────────────────────
            # CLASIFICADOR
            # ─────────────────────────────────────────────────────────
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
            
            # Inicialización de pesos
            self._initialize_weights()
    
    def _make_residual_block(self, in_channels, out_channels, num_blocks, stride):
        """Crea un bloque de capas residuales."""
        layers = []
        
        # Primer bloque con stride (puede cambiar dimensiones)
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # Bloques subsecuentes (stride=1)
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, (1.0/512)**0.5)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.pretrained:
            return self.backbone(x)
            
        # Entrada: (N, 3, 224, 224)
        
        # Bloque inicial
        x = self.conv1(x)          # (N, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)        # (N, 64, 56, 56)
        
        # Bloques residuales
        x = self.layer1(x)         # (N, 64, 56, 56)
        x = self.layer2(x)         # (N, 128, 28, 28)
        x = self.layer3(x)         # (N, 256, 14, 14)
        x = self.layer4(x)         # (N, 512, 7, 7)
        
        # Clasificador
        x = self.avgpool(x)        # (N, 512, 1, 1)
        x = x.flatten(1)           # (N, 512)
        x = self.fc(x)             # (N, 1000)
        
        return x


class BasicBlock(nn.Module):
    """
    Bloque residual básico con conexión skip.
    
    Estructura:
        Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → (+ skip connection) → ReLU
    """
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Proyección del skip connection si necesario
        self.skip_projection = None
        if stride != 1 or in_channels != out_channels:
            self.skip_projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        identity = x
        
        # Rama principal
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.skip_projection is not None:
            identity = self.skip_projection(x)
        
        # Sumar y activar
        out += identity
        out = self.relu(out)
        
        return out
