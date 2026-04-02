from torch import nn
from torchvision import models


class ResNet(nn.Module):
    """ResNet model with different architectures"""

    def __init__(self, variant, pretrained=True):
        super(ResNet, self).__init__()

        if variant == 18:
            weights = models.ResNet18_Weights() if pretrained else None
            self.model = models.resnet18(weights=weights)
        elif variant == 34:
            weights = models.ResNet34_Weights() if pretrained else None
            self.model = models.resnet34(weights=weights)
        elif variant == 50:
            weights = models.ResNet50_Weights() if pretrained else None
            self.model = models.resnet50(weights=weights)
        elif variant == 101:
            weights = models.ResNet101_Weights() if pretrained else None
            self.model = models.resnet101(weights=weights)
        elif variant == 152:
            weights = models.ResNet152_Weights() if pretrained else None
            self.model = models.resnet152(weights=weights)
        else:
            raise ValueError("Invalid ResNet variant")
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output tensor after feature extraction.
        """
        return self.model(x)
