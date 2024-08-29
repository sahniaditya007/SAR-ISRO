import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getStorage } from 'firebase/storage';

const firebaseConfig = {
    apiKey: "AIzaSyC89vaJYFZU2_hPeGQ1648Aue-18ehkPpQ",
    authDomain: "sih2024-61ef0.firebaseapp.com",
    projectId: "sih2024-61ef0",
    storageBucket: "sih2024-61ef0.appspot.com",
    messagingSenderId: "23176558516",
    appId: "1:23176558516:web:e651f831ce17997349a4e1",
    measurementId: "G-W6BLW16EV9"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const storage = getStorage(app);

export { auth, storage };
