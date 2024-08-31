#https://huggingface.co/Intel/dpt-large
from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import requests
import torch
# Initialize the depth estimation pipeline
depth_estimation_pipeline = pipeline("depth-estimation", model="Intel/dpt-large")
# Load the processor and model directly
processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large")
def estimate_depth(image_url):
    # Load image
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Use the pipeline for quick depth estimation
    depth_map_pipeline = depth_estimation_pipeline(image)

    # Use the processor and model directly for more control
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Normalize the depth map
    predicted_depth = predicted_depth.squeeze().cpu().numpy()
    normalized_depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())

    return depth_map_pipeline, normalized_depth
# Example usage
image_url = "https://example.com/path-to-your-image.jpg"
depth_map_pipeline, normalized_depth = estimate_depth(image_url)

# Display or process the depth maps as needed
# For example, using matplotlib to display the depth maps
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# Display pipeline depth estimation
plt.subplot(1, 2, 1)
plt.imshow(depth_map_pipeline['depth'].squeeze(), cmap='plasma')
plt.title("Pipeline Depth Estimation")

# Display normalized depth map from direct model usage
plt.subplot(1, 2, 2)
plt.imshow(normalized_depth, cmap='plasma')
plt.title("Direct Model Depth Estimation")

plt.show()
