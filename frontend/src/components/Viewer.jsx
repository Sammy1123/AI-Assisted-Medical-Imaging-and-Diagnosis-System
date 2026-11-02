import React, { useState } from "react";
import axios from "axios";

const API = process.env.REACT_APP_API_URL || "http://localhost:8000/api";

export default function Viewer() {
  const [file, setFile]         = useState(null);
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [radId, setRadId]       = useState("");
  const [notes, setNotes]       = useState("");
  const [reviewed, setReviewed] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setReviewed(false);
    const form = new FormData();
    form.append("file", file);
    try {
      const { data } = await axios.post(`${API}/predict`, form);
      setResult(data);
    } catch (e) {
      alert("Prediction failed: " + (e.response?.data?.detail || e.message));
    } finally {
      setLoading(false);
    }
  };

  const handleReview = async (decision) => {
    if (!radId.trim()) { alert("Enter radiologist ID first."); return; }
    const form = new FormData();
    form.append("prediction_id", result.prediction_id);
    form.append("decision", decision);
    form.append("radiologist_id", radId);
    form.append("notes", notes);
    await axios.post(`${API}/review`, form);
    setReviewed(true);
  };

  return (
    <div style={{ fontFamily: "sans-serif", maxWidth: 800, margin: "0 auto", padding: 24 }}>
      <h1>🩺 Medical Imaging AI</h1>
      <p style={{ color: "red", fontWeight: "bold" }}>
        ⚠️ FOR EDUCATION / RESEARCH ONLY — NOT FOR CLINICAL USE
      </p>

      {/* Upload */}
      <div style={{ marginBottom: 16 }}>
        <input type="file" accept=".dcm,.png,.jpg,.jpeg"
          onChange={e => setFile(e.target.files[0])} />
        <button onClick={handleUpload} disabled={!file || loading}
          style={{ marginLeft: 8 }}>
          {loading ? "Analysing…" : "Analyse"}
        </button>
      </div>

      {/* Result */}
      {result && (
        <div style={{ border: "1px solid #ccc", borderRadius: 8, padding: 16 }}>
          <h2>Result</h2>
          <p><strong>Label:</strong> {result.label}</p>
          <p><strong>Probability:</strong> {(result.probability * 100).toFixed(1)}%</p>
          <p><strong>Confidence:</strong> {result.confidence}%</p>

          <h3>Grad-CAM Overlay</h3>
          <img src={`data:image/png;base64,${result.overlay_b64}`}
            alt="Grad-CAM" style={{ maxWidth: "100%", borderRadius: 4 }} />

          {/* Radiologist review */}
          {!reviewed ? (
            <div style={{ marginTop: 16 }}>
              <h3>Radiologist Review</h3>
              <input placeholder="Radiologist ID" value={radId}
                onChange={e => setRadId(e.target.value)}
                style={{ marginRight: 8 }} />
              <input placeholder="Notes (optional)" value={notes}
                onChange={e => setNotes(e.target.value)}
                style={{ width: 300, marginRight: 8 }} />
              <button onClick={() => handleReview("approve")}
                style={{ background: "#4caf50", color: "#fff", marginRight: 8 }}>
                ✅ Approve
              </button>
              <button onClick={() => handleReview("reject")}
                style={{ background: "#f44336", color: "#fff" }}>
                ❌ Reject
              </button>
            </div>
          ) : (
            <p style={{ color: "green", marginTop: 16 }}>✔ Decision recorded in audit log.</p>
          )}
        </div>
      )}
    </div>
  );
}
