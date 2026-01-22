import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Interceptor for responses to handle errors globally if needed
apiClient.interceptors.response.use(
    (response: any) => response,
    (error: any) => {
        // We can dispatch global toasts here later
        return Promise.reject(error);
    }
);
