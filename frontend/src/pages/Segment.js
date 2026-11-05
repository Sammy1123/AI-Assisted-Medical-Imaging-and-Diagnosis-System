import React, { useState } from "react";
import axios from "axios";

const API = process.env.REACT_APP_API_URL || "http://localhost:8080/api";

export default function Segment() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sliceIdx, setSliceIdx] = useState(0);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const { data } = await axios.post(`${API}/segment`, form, {
        timeout: 120000,
      });
      setResult(data);
      setSliceIdx(0);
    } catch (e) {
      alert("Segmentation failed: " + (e.response?.data?.detail || e.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        fontFamily: "sans-serif",
        maxWidth: 900,
        margin: "0 auto",
        padding: 24,
      }}
    >
      <h1>🧠 Spleen Segmentation (3D U-Net)</h1>
      <p style={{ color: "red", fontWeight: "bold" }}>
        ⚠️ FOR EDUCATION / RESEARCH ONLY — NOT FOR CLINICAL USE
      </p>

      <div style={{ marginBottom: 16 }}>
        <input
          type="file"
          accept=".nii,.nii.gz"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          style={{ marginLeft: 8 }}
        >
          {loading ? "Segmenting…" : "Segment"}
        </button>
      </div>

      {result && result.slices_b64?.length > 0 && (
        <div style={{ border: "1px solid #ccc", borderRadius: 8, padding: 16 }}>
          <p>
            <strong>Masked axial slices:</strong> {result.n_slices_with_mask}
          </p>
          <p>
            <strong>Segmented voxels:</strong>{" "}
            {result.voxels_segmented.toLocaleString()}
          </p>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              marginBottom: 8,
            }}
          >
            <button onClick={() => setSliceIdx((i) => Math.max(0, i - 1))}>
              ◀ Prev
            </button>
            <span>
              Slice {result.slice_indices[sliceIdx]} ({sliceIdx + 1}/
              {result.slices_b64.length})
            </span>
            <button
              onClick={() =>
                setSliceIdx((i) =>
                  Math.min(result.slices_b64.length - 1, i + 1),
                )
              }
            >
              Next ▶
            </button>
          </div>

          <img
            src={`data:image/png;base64,${result.slices_b64[sliceIdx]}`}
            alt="Segmentation overlay"
            style={{ maxWidth: "100%", borderRadius: 4 }}
          />
        </div>
      )}

      {result && result.slices_b64?.length === 0 && (
        <p>
          No spleen voxels detected in this volume (model undertrained — see
          PROGRESS.md).
        </p>
      )}
    </div>
  );
}
