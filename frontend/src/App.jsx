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
    <main style={{ maxWidth: 860, margin: "2rem auto", fontFamily: "Arial, sans-serif" }}>
      <h1>Skin Cancer Classification</h1>
      <p>Upload a dermoscopic image to get a 7-class prediction.</p>
      <ImageUploader onChange={setFile} />
      {previewUrl && (
        <div style={{ marginTop: 12 }}>
          <img src={previewUrl} alt="preview" style={{ width: 280, borderRadius: 8 }} />
        </div>
      )}
      <button style={{ marginTop: 12 }} onClick={handlePredict} disabled={loading}>
        Predict
      </button>
      {loading && <Loader />}
      {error && <p style={{ color: "crimson" }}>{error}</p>}
      <PredictionResult result={result} />
    </main>
  );
}
