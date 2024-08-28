import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Function to create a custom colormap
def create_custom_colormap():
    # Define custom colors (you can customize these colors)
    colors = [(0, 0, 0),  # black
              (0.2, 0.1, 0.6),  # purple
              (0.1, 0.5, 0.8),  # blue
              (0.4, 0.9, 0.6),  # green
              (0.9, 0.9, 0.4),  # yellow
              (1, 0.6, 0.2),  # orange
              (1, 0, 0)]  # red
              
    n_bins = 256  # Number of bins for interpolation
    cmap_name = 'custom_colormap'
    
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Load the SAR image in grayscale mode
sar_image_path = r"C:\delete\Onagawa_KV.jpg"
output_image_path = r"C:\delete\colorful_sar_image.png"

sar_image = cv2.imread(sar_image_path, cv2.IMREAD_GRAYSCALE)

if sar_image is None:
    print("Error: Could not load image. Please check the file path.")
else:
    # Enhance contrast using Histogram Equalization
    sar_image_equalized = cv2.equalizeHist(sar_image)
    
    # Create a custom colormap
    custom_cmap = create_custom_colormap()
    
    # Normalize the image to the range [0, 1] for applying the custom colormap
    normalized_sar_image = sar_image_equalized / 255.0
    
    # Apply the custom colormap
    colorful_sar_image_custom = custom_cmap(normalized_sar_image)
    
    # Convert the image from [0, 1] to [0, 255] and from float to uint8
    colorful_sar_image_custom = (colorful_sar_image_custom[:, :, :3] * 255).astype(np.uint8)
    
    # Apply a predefined colormap for comparison (e.g., COLORMAP_JET)
    colorful_sar_image_jet = cv2.applyColorMap(sar_image_equalized, cv2.COLORMAP_JET)
    
    # Combine both colormaps (custom and JET) to enrich colors
    combined_image = cv2.addWeighted(colorful_sar_image_custom, 0.5, colorful_sar_image_jet, 0.5, 0)
    
    # Save the colorful SAR image
    cv2.imwrite(output_image_path, combined_image)

    # Display the images
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
