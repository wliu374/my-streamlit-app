import torch.nn.functional as F
import torch
import torch.nn as nn
import timm
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
class MobileNetV4Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV4Encoder, self).__init__()
        self.backbone = timm.create_model(
            'mobilenetv4_conv_small.e2400_r224_in1k',
            pretrained=pretrained,
            features_only=True  # ✅ Extracts multi-scale feature maps
        )
    def forward(self, x):
        features = self.backbone(x)  # Returns multiple feature maps
        return features

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = nn.Sequential(
            convBlock(960, 64),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.up2 = nn.Sequential(
            convBlock(64, 64),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.weights_init()

    def forward(self, features, input):
        _, _, f2, _, f4 = features  # Unpack feature maps (deepest to shallowest)
        x = self.up1(f4)
        x = F.interpolate(x, size=f2.shape[2:], mode="bilinear", align_corners=True)
        x = self.up2(f2 + x)
        x = F.interpolate(x, size=input.shape[2:], mode="bilinear", align_corners=True)
        return x

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class convBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(convBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# Full segmentation model
class MobileNetV4Segmentation(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV4Segmentation, self).__init__()
        self.encoder = MobileNetV4Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        features = self.encoder(x)  # Extract feature maps
        output = self.decoder(features, x)  # Upsample to full resolution
        return output

def loadCheckpoint(net, chekcpointPath, start_epoch):
    checkpoint = torch.load(chekcpointPath)
    net.load_state_dict(checkpoint['state_dict'])
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
    if 'optimizer' in checkpoint:
        optimizer = checkpoint['optimizer']
    if 'train_loss' in checkpoint:
        net.train_loss = checkpoint['train_loss']
    if 'val_loss' in checkpoint:
        net.val_loss = checkpoint['val_loss']
    print("==> load checkpoint '{}' (epoch {})"
          .format(chekcpointPath, start_epoch))

    if 'measure' in checkpoint:
        net.measure = checkpoint['measure']

    print("net.measure[accuracy]:",len(net.measure['accuracy']))
    return net

# Function to load the MobileNet segmentation model
def load_mobilenet_seg():
    """Loads the MobileNetV4 segmentation model with pretrained weights."""

    model = MobileNetV4Segmentation()
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_best_r1-score.pth")
    model = loadCheckpoint(model,MODEL_PATH,0)

    model.eval()

    return model

if __name__ == '__main__':
    ratio = 0.125
    image_mean = [0.4663, 0.4657, 0.3188]
    image_std = [1, 1, 1]
    image_mean = np.array(image_mean).reshape((1, 1, 3))
    image_std = np.array(image_std).reshape((1, 1, 3))

    image = np.array(Image.open("./fragm.JPG"))

    h,w = image.shape[:2]
    nh = int(np.ceil(h * ratio))
    nw = int(np.ceil(w * ratio))

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    image = (image/255 - image_mean)/image_std
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).cuda()
    # print(image.size())
    model = load_mobilenet_seg().cuda()
    model.eval()
    with torch.no_grad():
        output = model(image)
        print(output.size())
        output = output.squeeze().cpu().detach().numpy()

    output[output >= 0.5] = 1
    output[output < 0.5] = 0

    plt.imshow(output,cmap='gray')
    plt.show()









