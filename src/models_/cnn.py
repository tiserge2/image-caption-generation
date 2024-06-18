import torch
import torch.nn as nn
import torchvision.models as models
import logging
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CNN(nn.Module):
    def __init__(self, architecture="alexnet", freeze_weights=False, pretrained=False):
        """
        Initialize the CNN module.

        Args:
            architecture (str): The architecture of the CNN model. Options are 'resnet18', 'resnet101', 'vgg', 'alexnet', and 'googlelenet'.
            freeze_weights (bool): Whether to freeze the weights of the pretrained model.
            pretrained (bool): Whether to load pretrained weights.
        """
        super().__init__()
        self.architecture = architecture
        self.freeze_weights = freeze_weights
        self.pretrained = "DEFAULT" if pretrained else None
        logging.info(f"==> Loaded Model: {self.architecture}")
        logging.info(f"==> Pretrained weights: {self.pretrained}")

        if self.architecture == 'resnet18':
            self.model = models.resnet18(weights=self.pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            self.dim = 512
        elif self.architecture == 'resnet101':
            self.model = models.resnet101(weights=self.pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            self.dim = 2048
        elif self.architecture == 'vgg':
            self.model = models.vgg19(weights=self.pretrained)
            self.dim = 512
            self.model = nn.Sequential(*list(self.model.features.children()))
        elif self.architecture == 'alexnet':
            self.model = models.alexnet(weights=self.pretrained)
            self.model = nn.Sequential(*list(self.model.features.children()))
            self.dim = 256
        elif self.architecture == 'googlelenet':
            self.model = models.inception_v3(weights=self.pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-3])
            self.dim = 512
        else:
            raise ValueError(f"Unsupported model: {self.architecture}")

        if self.freeze_weights and self.pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
        elif self.freeze_weights and not self.pretrained:
            print("WARNING: Cannot freeze the weights if the pretrained model is not set.")

        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
    
    def print_layers(self):
        """
        Print the layers of the model.
        """
        for idx, layer in enumerate(self.model.children()):
            print(f"Layer {idx}: {layer}")

    def forward(self, x):
        """
        Forward pass for the CNN module.

        Args:
            x (torch.Tensor): Input image tensor. Shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Processed feature tensor. Shape (batch_size, num_features, feature_dim).
        """
        x = self.model(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x

    
if __name__ == "__main__":
    # Example usage to print layers
    cnn_model = CNN(architecture='alexnet', freeze_weights=False, pretrained=False)
    cnn_model.print_layers()