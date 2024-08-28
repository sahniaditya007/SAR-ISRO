from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

def create_custom_colormap():
    colors = [(0, 0, 0), (0.2, 0.1, 0.6), (0.1, 0.5, 0.8), 
              (0.4, 0.9, 0.6), (0.9, 0.9, 0.4), (1, 0.6, 0.2), (1, 0, 0)]
    cmap_name = 'custom_colormap'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

def enhance_contrast(image):
    return cv2.equalizeHist(image)

def normalize_image(image):
    return image / 255.0

def apply_custom_colormap(image, cmap):
    colored_image = cmap(image)
    return (colored_image[:, :, :3] * 255).astype(np.uint8)

def blend_images(image1, image2, alpha=0.5, beta=0.5, gamma=0):
    return cv2.addWeighted(image1, alpha, image2, beta, gamma)

def process_sar_image(image):
    sar_image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    sar_image_equalized = enhance_contrast(sar_image)
    custom_cmap = create_custom_colormap()
    normalized_sar_image = normalize_image(sar_image_equalized)
    colorful_sar_image_custom = apply_custom_colormap(normalized_sar_image, custom_cmap)
    colorful_sar_image_jet = cv2.applyColorMap(sar_image_equalized, cv2.COLORMAP_JET)
    combined_image = blend_images(colorful_sar_image_custom, colorful_sar_image_jet)
    return combined_image

@app.route('/process_image', methods=['POST'])
def upload_image():
    file = request.files['image']
    processed_image = process_sar_image(file)
    _, img_encoded = cv2.imencode('.png', processed_image)
    return send_file(BytesIO(img_encoded.tobytes()), mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
