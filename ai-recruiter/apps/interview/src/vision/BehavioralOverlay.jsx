// BehavioralOverlay — small floating panel rendered on the interview page.
// Shows the candidate's webcam preview + live 4-D dimension scores from the
// vision pipeline. Calibration progress shown until is_calibrated is true.
import React from 'react';

const DIM_LABELS = {
  cognitive_load: 'Cognitive Load',
  emotional_arousal: 'Emotional Arousal',
  engagement_level: 'Engagement',
  confidence_level: 'Confidence',
};

export default function BehavioralOverlay({
  videoRef, canvasRef, connected, isCalibrated,
  calibrationProgress, dimensions, latencyMs, error,
}) {
  const dims = dimensions || {};
  return (
    <div style={styles.wrap}>
      <div style={styles.header}>
        <strong>Behavioral signal</strong>
        <span style={{ ...styles.dot, background: connected ? '#22c55e' : '#94a3b8' }} />
      </div>

      <div style={styles.videoBox}>
        <video ref={videoRef} muted playsInline style={styles.video} />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
        {!isCalibrated && connected && (
          <div style={styles.calOverlay}>
            <div>Calibrating… {Math.round((calibrationProgress || 0) * 100)}%</div>
          </div>
        )}
      </div>

      <div style={styles.dims}>
        {Object.keys(DIM_LABELS).map((k) => (
          <Bar key={k} label={DIM_LABELS[k]} value={dims[k]} disabled={!isCalibrated} />
        ))}
      </div>

      <div style={styles.footer}>
        {error
          ? <span style={{ color: '#f87171' }}>{error}</span>
          : <span>{latencyMs ? `${latencyMs.toFixed(0)} ms / frame` : '—'}</span>}
      </div>
    </div>
  );
}

function Bar({ label, value, disabled }) {
  const v = typeof value === 'number' ? value : 0;
  const pct = Math.max(0, Math.min(1, v)) * 100;
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
        <span style={{ opacity: disabled ? 0.5 : 1 }}>{label}</span>
        <span style={{ opacity: disabled ? 0.5 : 1 }}>
          {disabled ? '—' : v.toFixed(2)}
        </span>
      </div>
      <div style={{ height: 6, background: '#1e293b', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%',
          background: disabled ? '#475569' : 'linear-gradient(90deg,#22c55e,#eab308,#ef4444)' }} />
      </div>
    </div>
  );
}

const styles = {
  wrap: {
    position: 'fixed', right: 16, top: 16, width: 280, padding: 12,
    borderRadius: 10, background: 'rgba(15,23,42,.85)',
    color: '#e2e8f0', boxShadow: '0 8px 24px rgba(0,0,0,.4)',
    fontFamily: '-apple-system, sans-serif', fontSize: 12, zIndex: 50,
    backdropFilter: 'blur(8px)',
  },
  header: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    marginBottom: 8,
  },
  dot: { width: 8, height: 8, borderRadius: 999 },
  videoBox: {
    position: 'relative', width: '100%', aspectRatio: '4/3',
    background: '#000', borderRadius: 6, overflow: 'hidden', marginBottom: 8,
  },
  video: { width: '100%', height: '100%', objectFit: 'cover' },
  calOverlay: {
    position: 'absolute', inset: 0, background: 'rgba(0,0,0,.55)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 13,
  },
  dims: { padding: '4px 0' },
  footer: {
    fontSize: 10, color: '#94a3b8', marginTop: 4, textAlign: 'right',
  },
};
