const BAR_COLORS = {
  low: '#3fb950',
  medium: '#d29922',
  high: '#f85149',
};

function getColor(prob) {
  if (prob < 0.35) return BAR_COLORS.low;
  if (prob < 0.65) return BAR_COLORS.medium;
  return BAR_COLORS.high;
}

export default function StressGauge({ label, probability, confidence, title }) {
  const pct = ((probability ?? 0) * 100).toFixed(1);
  const color = getColor(probability ?? 0);

  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 13, color: '#8b949e' }}>{title}</span>
        <span
          style={{
            fontSize: 13,
            fontWeight: 700,
            color: label === 'stressed' ? '#f85149' : label === 'not_stressed' ? '#3fb950' : '#8b949e',
          }}
        >
          {label || '—'}
          {confidence != null && ` (${(confidence * 100).toFixed(0)}%)`}
        </span>
      </div>
      <div
        style={{
          width: '100%',
          height: 18,
          background: '#21262d',
          borderRadius: 9,
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: '100%',
            background: color,
            borderRadius: 9,
            transition: 'width 0.2s, background 0.2s',
          }}
        />
        <span
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            fontSize: 11,
            fontWeight: 700,
            color: '#fff',
            textShadow: '0 1px 2px rgba(0,0,0,0.7)',
          }}
        >
          {pct}%
        </span>
      </div>
    </div>
  );
}
