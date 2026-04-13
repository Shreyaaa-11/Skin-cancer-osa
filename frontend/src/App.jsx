import { useMemo, useState } from "react";
import ImageUploader from "./components/ImageUploader";
import Loader from "./components/Loader";
import PredictionResult from "./components/PredictionResult";
import { predictImage } from "./services/api";

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  const handlePredict = async () => {
    if (!file) {
      setError("Please upload an image.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (e) {
      setError(e?.response?.data?.detail || "Prediction request failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{ minHeight: "100vh", background: "#f6f7fb", padding: "32px 16px" }}>
      <div style={{ maxWidth: 920, margin: "0 auto", fontFamily: "Inter, Arial, sans-serif" }}>
        <div style={{ padding: 18, borderRadius: 14, background: "#fff", border: "1px solid #eaeaea" }}>
          <h1 style={{ marginTop: 0 }}>Skin Cancer Classification</h1>
          <p style={{ color: "#555" }}>
            Upload a dermoscopic image. You’ll get the model’s most likely class and confidence scores.
          </p>
  
          <ImageUploader onChange={setFile} />
  
          {previewUrl && (
            <div style={{ marginTop: 12 }}>
              <img src={previewUrl} alt="preview" style={{ width: 320, maxWidth: "100%", borderRadius: 12 }} />
            </div>
          )}
  
          <button
            style={{
              marginTop: 12,
              padding: "10px 14px",
              borderRadius: 10,
              border: "1px solid #ddd",
              background: loading ? "#f1f1f1" : "#111827",
              color: loading ? "#555" : "#fff",
              cursor: loading ? "not-allowed" : "pointer",
              fontWeight: 700,
            }}
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? "Predicting..." : "Predict"}
          </button>
  
          {loading && <Loader />}
          {error && <p style={{ color: "crimson" }}>{error}</p>}
        </div>
  
        <PredictionResult result={result} />
      </div>
    </main>
  );
}
