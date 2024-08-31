import torch
import numpy as np
from PIL import Image
import requests
from transformers import DPTImageProcessor, DPTForDepthEstimation

class DepthEstimationModel:
    def __init__(self, model_name="Intel/dpt-hybrid-midas"):
        self.image_processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name, low_cpu_mem_usage=True)

    def estimate_depth(self, image_url: str):
        # Load image
        image = Image.open(requests.get(image_url, stream=True).raw)
        
        # Prepare image for the model
        inputs = self.image_processor(images=image, return_tensors="pt")
        
        # Predict depth
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        # Process the prediction into a visualizable format
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth_image = Image.fromarray(formatted)

        return depth_image

    def save_depth_image(self, depth_image: Image, save_path: str):
        depth_image.save(save_path)

# Example usage:
if __name__ == "__main__":
    # Instantiate the depth estimation model
    depth_model = DepthEstimationModel()
    
    # Provide the URL of the image you want to process
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    # Estimate the depth and obtain the depth image
    depth_image = depth_model.estimate_depth(image_url)
    
    # Optionally, save the depth image to a file
    save_path = "depth_map.png"
    depth_model.save_depth_image(depth_image, save_path)
    print(f"Depth image saved to {save_path}")
