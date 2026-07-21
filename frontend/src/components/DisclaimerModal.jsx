import React from 'react';

const DisclaimerModal = ({ onAccept }) => {
  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 99999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1rem',
        backgroundColor: 'rgba(2, 6, 23, 0.85)',
        backdropFilter: 'blur(24px)',
        WebkitBackdropFilter: 'blur(24px)',
      }}
    >
      <div
        style={{
          width: '100%',
          maxWidth: '540px',
          backgroundColor: '#ffffff',
          borderRadius: '2rem',
          boxShadow: '0 40px 100px rgba(0,0,0,0.5)',
          overflow: 'hidden',
          animation: 'disclaimerFadeIn 0.3s ease-out',
        }}
      >
        <div style={{ padding: '3rem 2.5rem', textAlign: 'center' }}>
          {/* Warning icon */}
          <div
            style={{
              width: '80px',
              height: '80px',
              borderRadius: '50%',
              backgroundColor: '#fef2f2',
              border: '2px solid #fecaca',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 2rem',
              fontSize: '2.5rem',
            }}
          >
            ⚠️
          </div>

          {/* Title */}
          <h2
            style={{
              fontSize: '1.5rem',
              fontWeight: 900,
              color: '#0f172a',
              letterSpacing: '-0.02em',
              marginBottom: '1.5rem',
              textTransform: 'uppercase',
              fontFamily: "'Poppins', 'Inter', sans-serif",
            }}
          >
            Medical Disclaimer
          </h2>

          {/* Disclaimer text */}
          <div
            style={{
              backgroundColor: '#f8fafc',
              padding: '1.75rem',
              borderRadius: '1.5rem',
              border: '1px solid #e2e8f0',
              marginBottom: '2rem',
            }}
          >
            <p
              style={{
                color: '#334155',
                fontWeight: 600,
                lineHeight: 1.7,
                fontSize: '0.95rem',
                margin: 0,
              }}
            >
              ⚠️ Medical Disclaimer — This AI output is a screening aid only.
              It does not constitute a clinical diagnosis. All results must be
              reviewed by a licensed medical professional before any action is
              taken.
            </p>
          </div>

          {/* Accept button */}
          <button
            onClick={onAccept}
            style={{
              width: '100%',
              backgroundColor: '#0f172a',
              color: '#ffffff',
              padding: '1.25rem 1.5rem',
              borderRadius: '1rem',
              fontWeight: 900,
              fontSize: '0.875rem',
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              border: 'none',
              cursor: 'pointer',
              boxShadow: '0 10px 30px rgba(15, 23, 42, 0.3)',
              transition: 'all 0.2s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.75rem',
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.backgroundColor = '#1e293b';
              e.currentTarget.style.transform = 'translateY(-1px)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.backgroundColor = '#0f172a';
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            <span style={{ fontSize: '1.1rem' }}>✓</span>
            I Understand &amp; Agree
          </button>
        </div>

        {/* Bottom colour indicator */}
        <div
          style={{
            height: '4px',
            background: 'linear-gradient(to right, #ef4444, #f59e0b, #3b82f6)',
          }}
        />
      </div>

      {/* Keyframe animation — injected inline for zero-dependency */}
      <style>{`
        @keyframes disclaimerFadeIn {
          from { opacity: 0; transform: scale(0.95) translateY(10px); }
          to   { opacity: 1; transform: scale(1) translateY(0); }
        }
      `}</style>
    </div>
  );
};

export default DisclaimerModal;
