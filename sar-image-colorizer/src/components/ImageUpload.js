import React, { useState } from 'react';
import axios from 'axios';

const ImageUpload = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [processedImage, setProcessedImage] = useState(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleUpload = () => {
        const formData = new FormData();
        formData.append('image', selectedFile);

        axios.post('http://localhost:5000/process_image', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            responseType: 'blob',
        })
        .then((response) => {
            const imageUrl = URL.createObjectURL(new Blob([response.data]));
            setProcessedImage(imageUrl);
        })
        .catch((error) => {
            console.error('Error uploading the file:', error);
        });
    };

    return (
        <section id="upload" style={styles.uploadSection}>
            <h2 style={styles.uploadTitle}>Upload and Process Your Image</h2>
            <input type="file" onChange={handleFileChange} style={styles.uploadInput} />
            <button onClick={handleUpload} style={styles.uploadButton}>Upload and Process</button>
            {processedImage && (
                <div style={styles.imageContainer}>
                    <h3 style={styles.resultTitle}>Processed Image:</h3>
                    <img src={processedImage} alt="Processed SAR" style={styles.resultImage} />
                </div>
            )}
        </section>
    );
};

const styles = {
    uploadSection: {
        padding: '50px 20px',
        textAlign: 'center',
        backgroundColor: '#f4f4f4',
    },
    uploadTitle: {
        fontSize: '32px',
        marginBottom: '20px',
    },
    uploadInput: {
        margin: '20px 0',
        fontSize: '16px',
    },
    uploadButton: {
        padding: '10px 20px',
        fontSize: '18px',
        color: '#fff',
        backgroundColor: '#333',
        border: 'none',
        borderRadius: '5px',
        cursor: 'pointer',
    },
    imageContainer: {
        marginTop: '30px',
    },
    resultTitle: {
        fontSize: '24px',
    },
    resultImage: {
        maxWidth: '100%',
        height: 'auto',
        borderRadius: '10px',
        boxShadow: '0 0 10px rgba(0, 0, 0, 0.5)',
    }
};

export default ImageUpload;
