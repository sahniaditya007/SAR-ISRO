import React from 'react';

const Hero = () => {
    return (
        <section style={styles.hero}>
            <div style={styles.overlay}>
                <h1 style={styles.heroText}>Turn Your Black & White SAR Images into Vibrant Color!</h1>
                <p style={styles.heroSubText}>Upload your SAR images and watch the magic happen.</p>
            </div>
        </section>
    );
};

const styles = {
    hero: {
        height: '100vh',
        background: 'url(https://source.unsplash.com/random/1600x900/?nature) no-repeat center center/cover',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
    },
    overlay: {
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        padding: '50px',
        borderRadius: '10px',
        textAlign: 'center',
    },
    heroText: {
        color: '#fff',
        fontSize: '48px',
        fontWeight: 'bold',
        margin: '0',
    },
    heroSubText: {
        color: '#fff',
        fontSize: '24px',
        marginTop: '20px',
    }
};

export default Hero;
