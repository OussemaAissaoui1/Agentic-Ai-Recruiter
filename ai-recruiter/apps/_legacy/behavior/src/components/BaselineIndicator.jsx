import { useState, useEffect } from 'react';

export default function BaselineIndicator({
  calibrationProgress = 0,
  isCalibrated = false,
  calibrationDuration = 45,
}) {
  const [showCheck, setShowCheck] = useState(false);
  const [visible, setVisible] = useState(true);
  const [fading, setFading] = useState(false);

  useEffect(() => {
    if (isCalibrated) {
      setShowCheck(true);
      // Start fade-out immediately
      requestAnimationFrame(() => setFading(true));
      const hideTimer = setTimeout(() => setVisible(false), 800);
      return () => clearTimeout(hideTimer);
    }
  }, [isCalibrated]);

  if (!visible) return null;

  const remaining = Math.ceil((1 - calibrationProgress) * calibrationDuration);
  const pct = Math.min(calibrationProgress * 100, 100);
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - calibrationProgress);

  return (
    <div style={{
      ...styles.overlay,
      opacity: fading ? 0 : 1,
      transition: 'opacity 0.8s ease',
    }}>
      <div style={styles.content}>
        {showCheck ? (
          <div style={styles.checkContainer}>
            <svg width="120" height="120" viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="54" fill="none" stroke="#3fb950" strokeWidth="5" />
              <polyline
                points="35,62 52,78 85,42"
                fill="none"
                stroke="#3fb950"
                strokeWidth="5"
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{
                  strokeDasharray: 80,
                  strokeDashoffset: 0,
                  animation: 'checkDraw 0.4s ease forwards',
                }}
              />
            </svg>
            <p style={styles.checkText}>Baseline Ready</p>
          </div>
        ) : (
          <>
            <svg width="120" height="120" viewBox="0 0 120 120" style={{ transform: 'rotate(-90deg)' }}>
              {/* Background ring */}
              <circle
                cx="60" cy="60" r={radius}
                fill="none"
                stroke="#30363d"
                strokeWidth="5"
              />
              {/* Progress ring */}
              <circle
                cx="60" cy="60" r={radius}
                fill="none"
                stroke="#58a6ff"
                strokeWidth="5"
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={dashOffset}
                style={{ transition: 'stroke-dashoffset 0.5s ease' }}
              />
            </svg>
            <div style={styles.progressText}>{pct.toFixed(0)}%</div>
            <h2 style={styles.title}>Calibrating baseline...</h2>
            <p style={styles.subtitle}>Please speak naturally and look at the camera</p>
            <p style={styles.countdown}>{remaining}s remaining</p>
          </>
        )}
      </div>
      <style>{`
        @keyframes checkDraw {
          from { stroke-dashoffset: 80; }
          to { stroke-dashoffset: 0; }
        }
      `}</style>
    </div>
  );
}

const styles = {
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    background: 'rgba(0, 0, 0, 0.75)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 10,
    borderRadius: 12,
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 8,
    position: 'relative',
  },
  progressText: {
    position: 'absolute',
    top: 42,
    fontSize: 24,
    fontWeight: 700,
    color: '#58a6ff',
  },
  title: {
    fontSize: 18,
    fontWeight: 600,
    color: '#f0f0f0',
    margin: '8px 0 0 0',
  },
  subtitle: {
    fontSize: 13,
    color: '#8b949e',
    margin: 0,
  },
  countdown: {
    fontSize: 14,
    fontWeight: 600,
    color: '#d29922',
    margin: '4px 0 0 0',
  },
  checkContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 8,
  },
  checkText: {
    fontSize: 18,
    fontWeight: 600,
    color: '#3fb950',
    margin: 0,
  },
};
