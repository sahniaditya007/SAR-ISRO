import React from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import ImageUpload from './components/ImageUpload';
import Footer from './components/Footer';

function App() {
    return (
        <div>
            <Navbar />
            <Hero />
            <ImageUpload />
            <Footer />
        </div>
    );
}

export default App;
