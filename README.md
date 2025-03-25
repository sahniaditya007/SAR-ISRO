# SAR-ISRO

This repository contains various AI models and a web application for processing and colorizing SAR (Synthetic Aperture Radar) images.

## Table of Contents

- [SAR-ISRO](#sar-isro)
  - [Table of Contents](#table-of-contents)
  - [AI Models](#ai-models)
    - [Depth Estimation](#depth-estimation)
    - [Image Classification](#image-classification)
    - [Zero-Shot Image Classification](#zero-shot-image-classification)
    - [Image Segmentation](#image-segmentation)
    - [Basic Image Processing](#basic-image-processing)
  - [Web Application](#web-application)
    - [Setup](#setup)
    - [Running the Application](#running-the-application)
  - [API](#api)
    - [Process Image](#process-image)
  - [License](#license)

## AI Models

### Depth Estimation

The depth estimation model is implemented in [AI-Models/AImodel.py](AI-Models/AImodel.py). It uses PyTorch to estimate depth from images.

### Image Classification

The image classification model is implemented in [AI-Models/classification.py](AI-Models/classification.py). It uses the `umm-maybe/AI-image-detector` model from Hugging Face.

### Zero-Shot Image Classification

The zero-shot image classification model is implemented in [AI-Models/zeroshotClassification.py](AI-Models/zeroshotClassification.py). It uses the `openai/clip-vit-base-patch32` model from Hugging Face.

### Image Segmentation

The image segmentation model is implemented in [AI-Models/segmentation.py](AI-Models/segmentation.py). It uses the `briaai/RMBG-1.4` model from Hugging Face.

### Basic Image Processing

Basic image processing functions are implemented in [AI-Models/basic.py](AI-Models/basic.py). These functions include contrast enhancement, normalization, and custom colormap application.

## Web Application

The web application is a React-based frontend with a Flask backend for processing SAR images.

### Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/SAR-ISRO.git
    cd SAR-ISRO
    ```

2. Install the required dependencies for the Flask backend:
    ```sh
    pip install -r requirements.txt
    ```

3. Navigate to the `sar-image-colorizer` directory and install the required dependencies for the React frontend:
    ```sh
    cd sar-image-colorizer
    npm install
    ```

### Running the Application

1. Start the Flask backend:
    ```sh
    python app.py
    ```

2. Start the React frontend:
    ```sh
    npm start
    ```

3. Open [http://localhost:3000](http://localhost:3000) in your browser to view the application.

## API

### Process Image

- **Endpoint:** `/process_image`
- **Method:** `POST`
- **Description:** Upload an SAR image to be processed and colorized.
- **Request:**
    - `image` (file): The SAR image file to be processed.
- **Response:** Returns the processed image.

## License

This project is licensed under the MIT License.