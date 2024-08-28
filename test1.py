import cv2
import matplotlib.pyplot as plt

# Load the SAR image in grayscale mode
sar_image = cv2.imread('C:\delete\Onagawa_KV.jpg', cv2.IMREAD_GRAYSCALE)

# Apply a colormap to the grayscale image (e.g., COLORMAP_JET)
colorful_sar_image = cv2.applyColorMap(sar_image, cv2.COLORMAP_JET)

# Save the colorful SAR image
cv2.imwrite('C:\delete\colorful_sar_image.png', colorful_sar_image)

# Display the images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Grayscale SAR Image')
plt.imshow(sar_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Colorful SAR Image')
plt.imshow(cv2.cvtColor(colorful_sar_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
