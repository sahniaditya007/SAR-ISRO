from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests

# Step 1: Load the image from a URL
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Step 2: Initialize the processor and model
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# Step 3: Prepare the image for the model
inputs = processor(images=image, return_tensors="pt")

# Step 4: Perform inference to get the depth prediction
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# Step 5: Interpolate the depth map to the original image size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),   # Add a channel dimension
    size=image.size[::-1],          # Resize to original image dimensions (width, height)
    mode="bicubic",                 # Bicubic interpolation
    align_corners=False,            # Disable aligning corners for interpolation
)

# Step 6: Normalize and convert the depth map to a visualizable format
output = prediction.squeeze().cpu().numpy()   # Remove channel dimension and convert to numpy array
formatted = (output * 255 / np.max(output)).astype("uint8")  # Scale to 8-bit range (0-255)

# Step 7: Convert the depth map to an image for visualization
depth = Image.fromarray(formatted)

# Optional: Display or save the depth image
depth.show()  # Displays the depth map
# depth.save("depth_map.png")  # Save the depth map to a file
