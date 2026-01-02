import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const api = axios.create({
    baseURL: API_BASE_URL,
});

export const getAttributes = async () => {
    try {
        const response = await api.get('/attributes');
        return response.data;
    } catch (error) {
        console.error("Error fetching attributes:", error);
        return [];
    }
};

export const generateSketch = async (selectedAttributes) => {
    try {
        const response = await api.post('/generate', { attributes: selectedAttributes });
        return response.data.image;
    } catch (error) {
        console.error("Error generating sketch:", error);
        throw error;
    }
};

export const searchDatabase = async (selectedAttributes) => {
    try {
        const response = await api.post('/search', { attributes: selectedAttributes });
        return response.data;
    } catch (error) {
        console.error("Error searching database:", error);
        throw error;
    }
};
