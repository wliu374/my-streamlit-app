import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class SegFormer(nn.Module):
    def __init__(self):
        super(SegFormer, self).__init__()
        self.segformerForSemanticSegmentation = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=1,
            ignore_mismatched_sizes=True)

    def forward(self, x):
        output = self.segformerForSemanticSegmentation(x)
        output = output.logits()
        output = torch.sigmoid(output)
        output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return output

def loadCheckpoint(net, chekcpointPath, start_epoch):
    checkpoint = torch.load(chekcpointPath,map_location="cpu")
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

# Function to load the SegFormer model
def load_segformer_model():
    """Loads the SegFormer model with pretrained weights."""

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels = 1,
    ignore_mismatched_sizes = True)
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_best_r1-score.pth")
    model = loadCheckpoint(model,MODEL_PATH,0)
    model.eval()

    return model

if __name__ == "__main__":
    ratio = 0.25
    image_mean = [0.4663, 0.4657, 0.3188]
    image_std = [1, 1, 1]
    image_mean = np.array(image_mean).reshape((1, 1, 3))
    image_std = np.array(image_std).reshape((1, 1, 3))

    image = np.array(Image.open("./fragm.JPG"))

    h, w = image.shape[:2]
    nh = int(np.ceil(h * ratio))
    nw = int(np.ceil(w * ratio))

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    image = (image / 255 - image_mean) / image_std
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).cuda()
    # print(image.size())
    model = load_segformer_model().cuda()
    # model.eval()
    # with torch.no_grad():
    #     output = model(image)
    #     print(output.size())
    #     output = output.squeeze().cpu().detach().numpy()
    #
    # output[output >= 0.5] = 1
    # output[output < 0.5] = 0
    #
    # plt.imshow(output, cmap='gray')
    # plt.show()










