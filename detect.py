import torch
import numpy as np
from PIL import Image
import tempfile

# Load YOLOv5 model (make sure best.pt exists in weights folder)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt', force_reload=False)

def run(image_file):
    image = Image.open(image_file).convert('RGB')
    image_np = np.array(image)

    # Run model
    results = model(image_np)

    # Save result image temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
        results.save()
        result_img_path = tmpfile.name
        return result_img_path, results.pandas().xyxy[0]
