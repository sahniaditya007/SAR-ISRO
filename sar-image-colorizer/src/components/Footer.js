import React from 'react';

const Footer = () => {
    return (
        <footer style={styles.footer}>
            <p style={styles.footerText}>© 2024 SAR Image Colorizer. All rights reserved.</p>
        </footer>
    );
};

const styles = {
    footer: {
        backgroundColor: '#333',
        padding: '20px',
        textAlign: 'center',
        color: '#fff',
        marginTop: '40px',
    },
    footerText: {
        margin: '0',
        fontSize: '16px',
    }
};

export default Footer;
