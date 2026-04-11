export default function PredictionResult({ result }) {
  if (!result) return null;
  return (
    <div>
      <h3>Prediction</h3>
      <p>Label: {result.label}</p>
      <p>Class index: {result.class_index}</p>
      <p>Latency: {result.latency_ms.toFixed(2)} ms</p>
      <h4>Probabilities</h4>
      <ol>
        {result.probabilities.map((p, idx) => (
          <li key={idx}>{p.toFixed(4)}</li>
        ))}
      </ol>
    </div>
  );
}
