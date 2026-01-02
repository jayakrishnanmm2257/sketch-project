import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GenBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, channels_img, image_size=256):
        super(Generator, self).__init__()
        self.init_size = image_size // 64 # 4x4 for 256 size
        self.label_dim = label_dim
        
        # Initial projection
        self.l1 = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 1024 * self.init_size ** 2)
        )

        # 4x4 -> 8x8
        self.block1 = GenBlock(1024, 512)
        # 8x8 -> 16x16
        self.block2 = GenBlock(512, 256)
        # 16x16 -> 32x32
        self.block3 = GenBlock(256, 128)
        
        # SPATIAL INJECTION POINT (32x32)
        # Input to block4 will be 128 (features) + label_dim (attributes)
        
        # 32x32 -> 64x64
        self.block4 = GenBlock(128 + label_dim, 64)
        
        # 64x64 -> 128x128
        self.block5 = GenBlock(64, 32)
        
        # 128x128 -> 256x256
        self.block6 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, channels_img, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # 1. Initial Processing (Latent)
        gen_input = torch.cat((labels, noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 1024, self.init_size, self.init_size)
        
        # 2. Upsampling
        out = self.block1(out) # -> 8x8
        out = self.block2(out) # -> 16x16
        out = self.block3(out) # -> 32x32
        
        # 3. Spatial Injection
        # Resize labels to 32x32 and concatenate
        label_map = labels.view(labels.size(0), labels.size(1), 1, 1)
        label_map = label_map.repeat(1, 1, out.size(2), out.size(3))
        out = torch.cat((out, label_map), 1)
        
        # 4. Final Upsampling
        out = self.block4(out) # -> 64x64
        out = self.block5(out) # -> 128x128
        img = self.block6(out) # -> 256x256
        
        return img


class Discriminator(nn.Module):
    def __init__(self, channels_img, label_dim, image_size=256):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [spectral_norm(nn.Conv2d(in_filters, out_filters, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return nn.Sequential(*block)

        # We define layers explicitly to access intermediate features for Feature Matching Loss
        self.layer1 = discriminator_block(channels_img + label_dim, 32, bn=False) # 128x128
        self.layer2 = discriminator_block(32, 64)       # 64x64
        self.layer3 = discriminator_block(64, 128)      # 32x32
        self.layer4 = discriminator_block(128, 256)     # 16x16
        self.layer5 = discriminator_block(256, 512)     # 8x8
        self.layer6 = discriminator_block(512, 1024)    # 4x4

        # The output layer
        self.adv_layer = nn.Sequential(
            spectral_norm(nn.Conv2d(1024, 1, 4, 2, 0)), # 1x1
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Spatially replicate labels to match image size
        label_map = labels.view(labels.size(0), labels.size(1), 1, 1)
        label_map = label_map.repeat(1, 1, self.image_size, self.image_size)
        
        # Concatenate image and label maps along channel dimension
        d_in = torch.cat((img, label_map), 1)
        
        # Pass through layers and collect features
        features = []
        out = self.layer1(d_in); features.append(out)
        out = self.layer2(out); features.append(out)
        out = self.layer3(out); features.append(out)
        out = self.layer4(out); features.append(out)
        out = self.layer5(out); features.append(out)
        out = self.layer6(out); features.append(out)
        
        validity = self.adv_layer(out)
        
        return validity.view(validity.shape[0], -1), features

# Helper to initialize weights
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

from torchvision.models import resnet34

def get_attribute_classifier(num_attributes=40, weights_path=None, device='cpu'):
    model = resnet34(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_attributes)
    )
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    return model
