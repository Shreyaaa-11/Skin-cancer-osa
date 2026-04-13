const CLASS_INFO = [
  { key: "akiec", name: "Actinic keratoses / intraepithelial carcinoma", note: "Sun-damage related lesion" },
  { key: "bcc", name: "Basal cell carcinoma", note: "Common skin cancer type" },
  { key: "bkl", name: "Benign keratosis-like lesions", note: "Usually non-cancerous" },
  { key: "df", name: "Dermatofibroma", note: "Usually benign fibrous nodule" },
  { key: "mel", name: "Melanoma", note: "Serious skin cancer type" },
  { key: "nv", name: "Melanocytic nevi (moles)", note: "Usually benign mole" },
  { key: "vasc", name: "Vascular lesions", note: "Blood-vessel related lesion" },
];

function clamp01(x) {
  if (typeof x !== "number") return 0;
  return Math.max(0, Math.min(1, x));
}

function percent(x) {
  return `${(clamp01(x) * 100).toFixed(1)}%`;
}

export default function PredictionResult({ result }) {
  if (!result) return null;

  const probs = Array.isArray(result.probabilities) ? result.probabilities : [];
  const items = CLASS_INFO.map((c, i) => ({
    ...c,
    index: i,
    p: probs[i] ?? 0,
  }))
    .sort((a, b) => b.p - a.p);

  const top = items[0];
  const confidence = top ? clamp01(top.p) : 0;

  return (
    <section
      style={{
        marginTop: 16,
        padding: 16,
        borderRadius: 12,
        border: "1px solid #e6e6e6",
        background: "#fff",
      }}
    >
      <h2 style={{ margin: 0, marginBottom: 8 }}>Result</h2>

      <div style={{ display: "grid", gap: 10 }}>
        <div style={{ padding: 12, borderRadius: 10, background: "#f7f7fb" }}>
          <div style={{ fontSize: 14, color: "#444" }}>Top prediction</div>
          <div style={{ fontSize: 18, fontWeight: 700 }}>
            {top?.name ?? result.label ?? "Unknown"}
          </div>
          <div style={{ color: "#555", marginTop: 4 }}>
            Confidence (model): <b>{percent(confidence)}</b>
          </div>
          <div style={{ color: "#666", fontSize: 13, marginTop: 6 }}>
            This is the class with the highest score from the model.
          </div>
        </div>

        <div>
          <h3 style={{ marginBottom: 8 }}>All class probabilities</h3>
          <div style={{ display: "grid", gap: 8 }}>
            {items.map((it) => {
              const isTop = it.index === result.class_index || it.key === result.label;
              return (
                <div
                  key={it.key}
                  style={{
                    padding: 10,
                    borderRadius: 10,
                    border: "1px solid #eee",
                    background: isTop ? "#f0fbf4" : "#fff",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                    <div>
                      <div style={{ fontWeight: 700 }}>
                        {it.name} <span style={{ fontWeight: 500, color: "#666" }}>({it.key})</span>
                      </div>
                      <div style={{ color: "#666", fontSize: 13 }}>{it.note}</div>
                    </div>
                    <div style={{ fontWeight: 800 }}>{percent(it.p)}</div>
                  </div>

                  <div style={{ marginTop: 8, height: 10, background: "#f1f1f1", borderRadius: 999 }}>
                    <div
                      style={{
                        width: `${clamp01(it.p) * 100}%`,
                        height: "100%",
                        borderRadius: 999,
                        background: isTop ? "#2e7d32" : "#4f46e5",
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div style={{ fontSize: 15, color: "#444" }}>
          <b>Debug values:</b> label=<code>{String(result.label)}</code>, class_index=<code>{String(result.class_index)}</code>, latency=<code>{Number(result.latency_ms).toFixed(2)} ms</code>
        </div>

        <details style={{ padding: 12, borderRadius: 10, background: "#fafafa", border: "1px solid #eee" }}>
          <summary style={{ cursor: "pointer", fontWeight: 600, fontSize: 13 }}>
            What do these fields mean?
          </summary>

          <div style={{ marginTop: 10, color: "#444", fontSize: 14, lineHeight: 1.5 }}>
            <p style={{ marginTop: 0 }}>
              <b>Label</b> is the short code for the predicted class (example: <code>mel</code>).
            </p>
            <p>
              <b>Class index</b> is just the position of that class in the model’s internal list (0–6). It’s mainly for
              debugging; the class name is what matters.
            </p>
            <p>
              <b>Latency</b> is how long the prediction took on your computer/server (in milliseconds). Lower is faster.
            </p>
            <p style={{ marginBottom: 0 }}>
              <b>Probabilities</b> are the model’s confidence scores for each class. Higher means “the model thinks this
              class is more likely”. They should sum close to 1.
            </p>
          </div>
        </details>

        <div style={{ fontSize: 12.5, color: "#666" }}>
          <b>Note:</b> This is an ML model output, not a medical diagnosis. If you’re concerned, consult a dermatologist.
        </div>
      </div>
    </section>
  );
}