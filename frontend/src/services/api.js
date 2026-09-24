import axios from 'axios';

const API_BASE_URL = 'http://localhost:8080';

const api = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000,
});

export const loginUser = async (username, password) => {
  const response = await api.post('/api/auth/login', { username, password });
  return response.data;
};

export const registerUser = async (username, email, password, name) => {
  const response = await api.post('/api/auth/register', { username, email, password, name });
  return response.data;
};

export const logoutUser = async () => {
  const response = await api.post('/api/auth/logout');
  return response.data;
};

export const fetchUserHistory = async (userId) => {
  const response = await api.get(`/api/user/${userId}/history`);
  return response.data;
};

export const predictImage = async (imageBase64, userId = null) => {
  const response = await api.post('/predict', {
    image: imageBase64,
    user_id: userId,
  });
  return response.data;
};

export default api;

export const deleteScan = async (scanId) => {
  const response = await api.delete(`/api/scan/${scanId}`);
  return response.data;
};

export const triggerGeminiRecommendation = async (userId) => {
  const response = await api.post(`/api/user/${userId}/recommendations/gemini`);
  return response.data;
};
