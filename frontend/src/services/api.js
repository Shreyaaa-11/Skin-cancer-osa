import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await axios.post(`${API_BASE}/predict`, formData, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return response.data;
}
