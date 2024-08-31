#https://huggingface.co/Falconsai/nsfw_image_detection
from transformers import pipeline, AutoModel, AutoTokenizer

# Step 1: Create an image classification pipeline
pipe = pipeline("image-classification", model="umm-maybe/AI-image-detector")

# Step 2: Load the model directly (this is useful if you need more control over the model itself)
model = AutoModel.from_pretrained("umm-maybe/AI-image-detector")

# Optionally, load the tokenizer if you're dealing with text or need to preprocess text inputs
tokenizer = AutoTokenizer.from_pretrained("umm-maybe/AI-image-detector")

# Example usage of the pipeline for image classification
def classify_image(image_path):
    # Use the pipeline to classify the image
    results = pipe(image_path)
    return results

# Example image path
image_path = "path_to_your_image.jpg"

# Classify the image
classification_results = classify_image(image_path)

# Display the results
print(classification_results)

# If you need to directly use the model for any custom operations
def custom_inference(image_tensor):
    # Assuming image_tensor is a preprocessed tensor of the image ready for model input
    outputs = model(image_tensor)
    return outputs

# For example, you might preprocess the image and run a custom inference
# image_tensor = your_preprocessing_function(image_path)
# custom_results = custom_inference(image_tensor)
# print(custom_results)
