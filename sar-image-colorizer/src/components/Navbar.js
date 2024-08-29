import React from 'react';

const Navbar = () => {
    return (
        <nav style={styles.navbar}>
            <h1 style={styles.logo}>SAR Image Colorizer</h1>
            <ul style={styles.navLinks}>
                <li><a href="#home" style={styles.navItem}>Home</a></li>
                <li><a href="#about" style={styles.navItem}>About</a></li>
                <li><a href="#upload" style={styles.navItem}>Upload</a></li>
            </ul>
        </nav>
    );
};

const styles = {
    navbar: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '20px',
        backgroundColor: '#333',
        color: '#fff',
    },
    logo: {
        fontSize: '24px',
        fontWeight: 'bold',
    },
    navLinks: {
        listStyle: 'none',
        display: 'flex',
        gap: '20px',
    },
    navItem: {
        color: '#fff',
        textDecoration: 'none',
        fontSize: '18px',
    }
};

export default Navbar;
