import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'

// Global reset
document.body.style.margin = '0';
document.body.style.padding = '0';
document.body.style.background = '#0f1117';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
