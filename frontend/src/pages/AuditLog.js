import React, { useEffect, useState } from "react";
import axios from "axios";

const API = process.env.REACT_APP_API_URL || "/api";

export default function AuditLog() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get(`${API}/audit?limit=50`)
      .then(r => setEntries(r.data))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p>Loading audit log…</p>;
  if (!entries.length) return <p>No audit entries yet.</p>;

  return (
    <div>
      <h2>Audit Log</h2>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ background: "#e8eaf6" }}>
            {["Timestamp", "Event", "Prediction ID", "Label / Decision", "Radiologist", "Notes"].map(h => (
              <th key={h} style={{ padding: "8px 12px", textAlign: "left", border: "1px solid #ccc" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {entries.map((e, i) => (
            <tr key={i} style={{ background: i % 2 ? "#f5f5f5" : "#fff" }}>
              <td style={{ padding: "6px 12px", border: "1px solid #eee" }}>{e.timestamp}</td>
              <td style={{ padding: "6px 12px", border: "1px solid #eee" }}>{e.event}</td>
              <td style={{ padding: "6px 12px", border: "1px solid #eee", fontFamily: "monospace" }}>{e.prediction_id}</td>
              <td style={{ padding: "6px 12px", border: "1px solid #eee" }}>{e.label || e.decision}</td>
              <td style={{ padding: "6px 12px", border: "1px solid #eee" }}>{e.radiologist_id || "—"}</td>
              <td style={{ padding: "6px 12px", border: "1px solid #eee" }}>{e.notes || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
