// API Configuration
// Defaults to localhost in development, uses VITE_API_DATE_URL env var in production

const getApiBaseUrl = () => {
    // If explicitly set in environment
    if (import.meta.env.VITE_API_BASE_URL) {
        return import.meta.env.VITE_API_BASE_URL;
    }

    // Default for development (proxy handling or direct)
    // If running in development mode without env var, assume localhost:8000
    // Note: Vite proxy handles relative paths like '/api', but hardcoded http://localhost:8000
    // in the codebase suggests we need a full URL if not using proxy exclusively.

    // However, for Vercel deployment, we want it to point to the backend URL.
    // For local dev, sticking to localhost:8000 is fine.

    return 'http://localhost:8000';
};

export const API_BASE_URL = getApiBaseUrl();
