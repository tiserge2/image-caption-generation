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

    def get_pretrained_cnn(self):
        if self.architecture == 'resnet':
            self.model = models.resnet50(weights=self.pretrained)
            # Remove the fully connected layers
            self.model = nn.Sequential(*list(self.model.children())[:-2])
        elif self.architecture == 'vgg':
            self.model = models.vgg16(weights=self.pretrained)
            # Remove the fully connected layers
            self.model = nn.Sequential(*list(self.model.features.children()))
        elif self.architecture == 'alexnet':
            self.model = models.alexnet(weights=self.pretrained)
            # Remove the fully connected layers
            self.model = nn.Sequential(*list(self.model.features.children()))
        elif self.architecture == 'googlelenet':
            self.model = models.inception_v3(weights=self.pretrained)
            # Remove the fully connected layers
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        else:
            raise ValueError(f"Unsupported model: {self.architecture}")
        
        # Freeze weights if specified
        if self.freeze_weights and self.pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
        elif self.freeze_weights and not self.pretrained:
            print("WARNING: Cannot freeze the weights if the pretrained models is not set.")

        return self.model
    
    def print_layers(self):
        model = self.get_pretrained_cnn()
        for idx, layer in enumerate(model.children()):
            print(f"Layer {idx}: {layer}")

    def forward(self, x):
        cnn = self.get_pretrained_cnn()
        x = cnn(x)
        return x
    
if __name__ == "__main__":
    # Example usage to print layers
    cnn_model = CNN(architecture='alexnet', freeze_weights=False, pretrained=False)
    cnn_model.print_layers()