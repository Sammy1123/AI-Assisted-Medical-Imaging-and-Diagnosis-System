import React from "react";
import Viewer from "./components/Viewer";
import AuditLog from "./pages/AuditLog";
import Segment from "./pages/Segment";

export default function App() {
  const [page, setPage] = React.useState("viewer");

  return (
    <div style={{ fontFamily: "sans-serif" }}>
      <nav style={{ background: "#1a237e", padding: "12px 24px", display: "flex", gap: 24 }}>
        <span style={{ color: "#fff", fontWeight: "bold", fontSize: 18 }}>🩺 Medical Imaging AI</span>
        <button onClick={() => setPage("viewer")}
          style={{ background: "none", border: "none", color: page === "viewer" ? "#ffeb3b" : "#fff", cursor: "pointer", fontSize: 15 }}>
          Viewer
        </button>
        <button onClick={() => setPage("audit")}
          style={{ background: "none", border: "none", color: page === "audit" ? "#ffeb3b" : "#fff", cursor: "pointer", fontSize: 15 }}>
          Audit Log
        </button>
        <button onClick={() => setPage("segment")}
          style={{ background: "none", border: "none", color: page === "segment" ? "#ffeb3b" : "#fff", cursor: "pointer", fontSize: 15 }}>
          Segmentation
        </button>
      </nav>
      <div style={{ padding: 24 }}>
        {page === "viewer" ? <Viewer /> : page === "segment" ? <Segment /> : <AuditLog />}
      </div>
    </div>
  );
}
