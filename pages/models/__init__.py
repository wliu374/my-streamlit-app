from .Mobilenetv4.Mobilenet_seg import load_mobilenet_seg
from .SegFormer.SegFormer import load_segformer_model


# Function to load all models
def load_models():
    """Load all available models and return them in a dictionary."""
    return {
        "MobileNetV4 Segmentation": load_mobilenet_seg(),
        "SegFormer": load_segformer_model()
    }
