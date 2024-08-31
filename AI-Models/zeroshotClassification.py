#https://huggingface.co/openai/clip-vit-base-patch32

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from transformers import pipeline, AutoProcessor, AutoModelForZeroShotImageClassification

# Step 1: Create a zero-shot image classification pipeline
pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

# Step 2: Load the model and processor directly (for more control)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")

# Function to create a custom colormap
def create_custom_colormap():
    colors = [(0, 0, 0),        # black
              (0.2, 0.1, 0.6),  # purple
              (0.1, 0.5, 0.8),  # blue
              (0.4, 0.9, 0.6),  # green
              (0.9, 0.9, 0.4),  # yellow
              (1, 0.6, 0.2),    # orange
              (1, 0, 0)]        # red
    
    cmap_name = 'custom_colormap'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# Function to enhance contrast using OpenCV's histogram equalization
def enhance_contrast(image):
    return cv2.equalizeHist(image)

# Function to normalize an image to the range [0, 1]
def normalize_image(image):
    return tf.cast(image, tf.float32) / 255.0

# Function to apply a custom colormap to an image
def apply_custom_colormap(image, cmap):
    colored_image = cmap(image.numpy())
    return (colored_image[:, :, :3] * 255).astype(np.uint8)

# Function to blend two images with given weights using TensorFlow
def blend_images(image1, image2, alpha=0.5, beta=0.5, gamma=0):
    blended_image = tf.add(tf.multiply(image1, alpha), tf.multiply(image2, beta))
    return tf.cast(blended_image + gamma, tf.uint8)

# Main function to process the SAR image using TensorFlow
def process_sar_image(sar_image_path, output_image_path):
    # Load the SAR image in grayscale mode
    sar_image = cv2.imread(sar_image_path, cv2.IMREAD_GRAYSCALE)
    
    if sar_image is None:
        raise FileNotFoundError(f"Error: Could not load image from {sar_image_path}. Please check the file path.")
    
    # Enhance contrast using OpenCV
    sar_image_equalized = enhance_contrast(sar_image)
    
    # Create a custom colormap
    custom_cmap = create_custom_colormap()
    
    # Normalize the equalized image
    normalized_sar_image = normalize_image(sar_image_equalized)
    
    # Apply the custom colormap
    colorful_sar_image_custom = apply_custom_colormap(normalized_sar_image, custom_cmap)
    
    # Apply a predefined colormap (e.g., COLORMAP_JET) for comparison
    colorful_sar_image_jet = cv2.applyColorMap(sar_image_equalized, cv2.COLORMAP_JET)
    
    # Blend the custom colormap image with the predefined colormap image
    combined_image = blend_images(tf.convert_to_tensor(colorful_sar_image_custom, dtype=tf.float32),
                                  tf.convert_to_tensor(colorful_sar_image_jet, dtype=tf.float32))
    
    # Save the resulting image
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, combined_image.numpy())
    
    # Optionally save the individual images
    cv2.imwrite(output_image_path.replace('.png', '_custom.png'), colorful_sar_image_custom)
    cv2.imwrite(output_image_path.replace('.png', '_jet.png'), colorful_sar_image_jet)

    return sar_image, colorful_sar_image_custom, combined_image.numpy()

# Function to classify SAR images using zero-shot image classification
def classify_sar_image(image_path, candidate_labels):
    # Preprocess the image
    image = processor(images=image_path, return_tensors="pt").pixel_values

    # Run the zero-shot image classification
    results = model(image, candidate_labels=candidate_labels)
    
    # Alternatively, use the pipeline for simplicity
    # results = pipe(image_path, candidate_labels=candidate_labels)
    
    return results

# Function to display images
def display_images(sar_image, colorful_sar_image_custom, combined_image):
    plt.figure(figsize=(18, 8))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Grayscale SAR Image')
    plt.imshow(sar_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Colorful SAR Image (Custom Colormap)')
    plt.imshow(colorful_sar_image_custom)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Combined Colorful SAR Image')
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()

# Paths to the input and output images
sar_image_path = r"C:\delete\Onagawa_KV.jpg"
output_image_path = r"C:\delete\colorful_sar_image.png"

# Define candidate labels for zero-shot classification
candidate_labels = ["urban area", "forest", "water", "farmland"]

# Process the SAR image and get the results
try:
    sar_image, colorful_sar_image_custom, combined_image = process_sar_image(sar_image_path, output_image_path)

    # Classify the SAR image using zero-shot classification
    classification_results = classify_sar_image(sar_image_path, candidate_labels)

    # Display the images
    display_images(sar_image, colorful_sar_image_custom, combined_image)
    
    # Print the classification results
    print("Zero-Shot Classification Results:")
    for result in classification_results:
        print(f"Label: {result['label']}, Score: {result['score']:.4f}")
    
except FileNotFoundError as e:
    print(e)
