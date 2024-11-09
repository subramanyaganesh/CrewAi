import React from 'react';
import App from './Components/App';
import {createRoot} from "react-dom/client"
import "./styles.css";

const root=createRoot(document.getElementById('root'))
root.render(<App />);