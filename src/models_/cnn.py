import torch
import torch.nn as nn
import torchvision.models as models
import logging
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CNN(nn.Module):
    def __init__(self, architecture="alexnet", freeze_weights=False, pretrained=False):
        super().__init__()
        self.architecture = architecture
        self.freeze_weights = freeze_weights
        self.pretrained = "DEFAULT" if pretrained else None
        logging.info(f"==> Loaded Model: {self.architecture}")
        logging.info(f"==> Pretrained weights: {self.pretrained}")

        if self.architecture == 'resnet':
            self.model = models.resnet50(weights=self.pretrained)
            # Remove the fully connected layers
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            self.dim = 2048
        elif self.architecture == 'vgg':
            self.model = models.vgg19(weights=self.pretrained)
            self.dim = 512
            # Remove the fully connected layers
            self.model = nn.Sequential(*list(self.model.features.children()))
        elif self.architecture == 'alexnet':
            self.model = models.alexnet(weights=self.pretrained)
            # Remove the fully connected layers
            self.model = nn.Sequential(*list(self.model.features.children()))
            self.dim = 256
        elif self.architecture == 'googlelenet':
            self.model = models.inception_v3(weights=self.pretrained)
            # Remove the fully connected layers
            self.model = nn.Sequential(*list(self.model.children())[:-3])
            self.dim = 1024
        else:
            raise ValueError(f"Unsupported model: {self.architecture}")
        
        # Freeze weights if specified
        if self.freeze_weights and self.pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
        elif self.freeze_weights and not self.pretrained:
            print("WARNING: Cannot freeze the weights if the pretrained models is not set.")

    
    def print_layers(self):
        for idx, layer in enumerate(self.model.children()):
            print(f"Layer {idx}: {layer}")

    def forward(self, x):
        x = self.model(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x
    
if __name__ == "__main__":
    # Example usage to print layers
    cnn_model = CNN(architecture='alexnet', freeze_weights=False, pretrained=False)
    cnn_model.print_layers()