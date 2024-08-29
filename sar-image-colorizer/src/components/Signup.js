import React, { useState } from 'react';
import { createUserWithEmailAndPassword } from 'firebase/auth';
import { auth } from '../firebaseConfig';

const Signup = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');

    const handleSignup = async (event) => {
        event.preventDefault();
        try {
            await createUserWithEmailAndPassword(auth, email, password);
            // Redirect or update the UI after successful signup
        } catch (error) {
            setError('Failed to create an account. Please try again.');
        }
    };

    return (
        <div style={styles.container}>
            <h2>Signup</h2>
            {error && <p style={styles.error}>{error}</p>}
            <form onSubmit={handleSignup} style={styles.form}>
                <input 
                    type="email" 
                    placeholder="Email" 
                    value={email} 
                    onChange={(e) => setEmail(e.target.value)} 
                    style={styles.input}
                    required
                />
                <input 
                    type="password" 
                    placeholder="Password" 
                    value={password} 
                    onChange={(e) => setPassword(e.target.value)} 
                    style={styles.input}
                    required
                />
                <button type="submit" style={styles.button}>Signup</button>
            </form>
        </div>
    );
};

const styles = {
    container: {
        maxWidth: '400px',
        margin: '50px auto',
        padding: '20px',
        textAlign: 'center',
        boxShadow: '0 0 10px rgba(0, 0, 0, 0.1)',
        borderRadius: '5px',
        backgroundColor: '#fff',
    },
    error: {
        color: 'red',
        marginBottom: '15px',
    },
    form: {
        display: 'flex',
        flexDirection: 'column',
    },
    input: {
        margin: '10px 0',
        padding: '10px',
        fontSize: '16px',
    },
    button: {
        padding: '10px',
        fontSize: '18px',
        backgroundColor: '#333',
        color: '#fff',
        border: 'none',
        cursor: 'pointer',
        borderRadius: '5px',
    },
};

export default Signup;
