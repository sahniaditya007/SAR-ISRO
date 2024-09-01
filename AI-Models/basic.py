import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

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

# Function to enhance contrast using histogram equalization
def enhance_contrast(image):
    return cv2.equalizeHist(image)

# Function to normalize an image to the range [0, 1]
def normalize_image(image):
    return image / 255.0

# Function to apply a custom colormap to an image
def apply_custom_colormap(image, cmap):
    colored_image = cmap(image)
    return (colored_image[:, :, :3] * 255).astype(np.uint8)

# Function to blend two images with given weights
def blend_images(image1, image2, alpha=0.5, beta=0.5, gamma=0):
    return cv2.addWeighted(image1, alpha, image2, beta, gamma)

# Function to generate an artificial depth map
def generate_depth_map(image, depth_factor=0.5):
    # Simulate depth by creating a gradient based on intensity
    depth_map = cv2.GaussianBlur(image, (15, 15), 0)
    depth_map = normalize_image(depth_map)  # Normalize to range [0, 1]
    depth_map = depth_map * depth_factor    # Adjust depth factor
    return depth_map

# Function to apply the depth map to the image
def apply_depth_map(image, depth_map):
    # Adjust the brightness based on the depth map
    return cv2.convertScaleAbs(image * (1 + depth_map))

# Main function to process the SAR image
def process_sar_image(sar_image_path, output_image_path):
    # Load the SAR image in grayscale mode
    sar_image = cv2.imread(sar_image_path, cv2.IMREAD_GRAYSCALE)
    
    if sar_image is None:
        raise FileNotFoundError(f"Error: Could not load image from {sar_image_path}. Please check the file path.")
    
    # Enhance contrast
    sar_image_equalized = enhance_contrast(sar_image)
    
    # Create a custom colormap
    custom_cmap = create_custom_colormap()
    
    # Normalize the equalized image
    normalized_sar_image = normalize_image(sar_image_equalized)
    
    # Apply the custom colormap
    colorful_sar_image_custom = apply_custom_colormap(normalized_sar_image, custom_cmap)
    
    # Generate and apply a depth map
    depth_map = generate_depth_map(sar_image_equalized)
    depth_image = apply_depth_map(sar_image_equalized, depth_map)
    
    # Apply a predefined colormap (e.g., COLORMAP_JET) for comparison
    colorful_sar_image_jet = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    
    # Blend the custom colormap image with the predefined colormap image
    combined_image = blend_images(colorful_sar_image_custom, colorful_sar_image_jet)
    
    # Save the resulting image
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, combined_image)
    
    # Optionally save the individual images
    cv2.imwrite(output_image_path.replace('.png', '_custom.png'), colorful_sar_image_custom)
    cv2.imwrite(output_image_path.replace('.png', '_jet.png'), colorful_sar_image_jet)

    return sar_image, colorful_sar_image_custom, combined_image, depth_map

# Function to display images
def display_images(sar_image, colorful_sar_image_custom, combined_image, depth_map):
    plt.figure(figsize=(24, 8))
    
    plt.subplot(1, 4, 1)
    plt.title('Original Grayscale SAR Image')
    plt.imshow(sar_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title('Depth Map')
    plt.imshow(depth_map, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title('Colorful SAR Image (Custom Colormap)')
    plt.imshow(colorful_sar_image_custom)
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title('Combined Colorful SAR Image')
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()

# Paths to the input and output images
sar_image_path = r"C:\Users\adity\Downloads\qse.jpg"
output_image_path = r"C:\delete\colorful_sar_image.png"

# Process the SAR image and get the results
try:
    sar_image, colorful_sar_image_custom, combined_image, depth_map = process_sar_image(sar_image_path, output_image_path)
    display_images(sar_image, colorful_sar_image_custom, combined_image, depth_map)
except FileNotFoundError as e:
    print(e)
