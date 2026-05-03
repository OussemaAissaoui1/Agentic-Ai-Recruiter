export default function StatsPanel({ stats, isCalibrated }) {
  const items = [
    { label: 'Frames Sent', value: stats.framesSent, color: '#58a6ff' },
    { label: 'Analyzed', value: stats.framesAnalyzed, color: '#3fb950' },
    { label: 'Stressed', value: stats.stressedCount, color: '#f85149' },
    { label: 'Latency', value: `${stats.latencyMs}ms`, color: '#d29922' },
  ];

  const stressRate =
    stats.framesAnalyzed > 0
      ? ((stats.stressedCount / stats.framesAnalyzed) * 100).toFixed(1)
      : '0.0';

  return (
    <div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 8,
          marginBottom: 12,
        }}
      >
        {items.map((item) => (
          <div
            key={item.label}
            style={{
              background: '#0d1117',
              borderRadius: 8,
              padding: 10,
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: 22, fontWeight: 700, color: item.color }}>
              {item.value}
            </div>
            <div style={{ fontSize: 11, color: '#8b949e', marginTop: 2 }}>
              {item.label}
            </div>
          </div>
        ))}
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          padding: '8px 4px',
          fontSize: 12,
          color: '#8b949e',
        }}
      >
        <span>Stress Rate</span>
        <span style={{ fontWeight: 700, color: '#f0f0f0' }}>{stressRate}%</span>
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          padding: '4px 4px',
          fontSize: 12,
          color: '#8b949e',
        }}
      >
        <span>Avg Stress Prob</span>
        <span style={{ fontWeight: 700, color: '#f0f0f0' }}>
          {(stats.avgStressProb * 100).toFixed(1)}%
        </span>
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          padding: '4px 4px',
          fontSize: 12,
          color: '#8b949e',
        }}
      >
        <span>Calibration</span>
        <span style={{ fontWeight: 700, color: isCalibrated ? '#3fb950' : '#d29922' }}>
          {isCalibrated ? 'Ready' : 'Pending'}
        </span>
      </div>
    </div>
  );
}
