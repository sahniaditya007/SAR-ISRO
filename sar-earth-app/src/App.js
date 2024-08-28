import React, { useState } from 'react';
import axios from 'axios';

function App() {
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
            responseType: 'blob' // Important: Set the response type to blob
        })
        .then((response) => {
            // Create a URL for the image blob
            const imageUrl = URL.createObjectURL(new Blob([response.data]));
            setProcessedImage(imageUrl);
        })
        .catch((error) => {
            console.error('Error uploading the file:', error);
        });
    };

    return (
        <div className="App">
            <h1>SAR Image Colorizer</h1>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleUpload}>Upload and Process</button>
            {processedImage && (
                <div>
                    <h2>Processed Image:</h2>
                    <img src={processedImage} alt="Processed SAR" />
                </div>
            )}
        </div>
    );
}

export default App;
