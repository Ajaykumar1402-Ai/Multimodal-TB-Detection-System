import React, { useState } from "react"

export function ConsentModal({ onAccept }) {
  const [checked, setChecked] = useState(false)

  return (
    <div style={{
      position: "fixed", inset: 0,
      background: "rgba(0,0,0,0.85)",
      zIndex: 9999,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: "20px"
    }}>
      <div style={{
        background: "white",
        padding: "40px",
        borderRadius: "12px",
        maxWidth: "560px",
        width: "100%"
      }}>
        <h2 style={{ color: "#C0392B", marginBottom: "16px" }}>
          ⚕️ Patient Data & Privacy Notice
        </h2>
        <p><strong>What we collect:</strong> Chest X-ray image and clinical form data.</p>
        <p><strong>Where processed:</strong> Secure cloud server.</p>
        <p><strong>Retention:</strong> Images deleted immediately after analysis.</p>
        <p><strong>Your rights:</strong> Withdraw at any time by closing this session.</p>
        <p><strong>Legal basis:</strong> DPDP Act 2023 · HIPAA · GDPR Article 9.</p>
        <hr style={{ margin: "20px 0" }} />
        <label style={{
          display: "flex", gap: "12px",
          alignItems: "flex-start", cursor: "pointer"
        }}>
          <input
            type="checkbox"
            checked={checked}
            onChange={e => setChecked(e.target.checked)}
            style={{ width: 20, height: 20, marginTop: 2 }}
          />
          <span>
            I confirm I have obtained the patient's consent
            to upload and process their radiograph for
            AI-assisted TB screening.
          </span>
        </label>
        <div style={{ display: "flex", gap: "12px", marginTop: "24px" }}>
          <button
            onClick={() => {
              if (checked) onAccept()
            }}
            disabled={!checked}
            style={{
              flex: 1, padding: "12px",
              background: checked ? "#1E8449" : "#cccccc",
              color: "white", border: "none",
              borderRadius: "8px",
              cursor: checked ? "pointer" : "not-allowed",
              fontWeight: 700, fontSize: "16px"
            }}
          >
            I Acknowledge & Proceed
          </button>
          <button
            onClick={() => window.location.href = "/"}
            style={{
              padding: "12px 20px",
              background: "#eeeeee",
              border: "none", borderRadius: "8px",
              cursor: "pointer"
            }}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}
