const DIMENSIONS = [
  {
    key: 'cognitive_load',
    label: 'Cognitive Load',
    color: '#8B5CF6',
    tooltip: 'Mental effort and processing load — elevated during complex questions',
  },
  {
    key: 'emotional_arousal',
    label: 'Emotional Arousal',
    color: '#F97316',
    tooltip: 'Physiological activation level — reflects emotional intensity',
  },
  {
    key: 'engagement_level',
    label: 'Engagement Level',
    color: '#14B8A6',
    tooltip: 'Active participation signals — eye contact, nodding, forward posture',
  },
  {
    key: 'confidence_level',
    label: 'Confidence Level',
    color: '#3B82F6',
    tooltip: 'Composure and self-assurance indicators — steady posture, open body language',
  },
];

function DeviationArrow({ value }) {
  if (value == null) return <span style={{ color: '#555' }}>—</span>;
  if (value > 0.65) return <span style={{ color: '#3fb950' }}>▲</span>;
  if (value < 0.35) return <span style={{ color: '#f85149' }}>▼</span>;
  return <span style={{ color: '#8b949e' }}>—</span>;
}

export default function DimensionGauges({ scores }) {
  return (
    <div style={styles.grid}>
      {DIMENSIONS.map((dim) => {
        const value = scores?.[dim.key];
        const pct = value != null ? (value * 100).toFixed(1) : '--';
        const fillWidth = value != null ? `${value * 100}%` : '0%';

        return (
          <div key={dim.key} style={styles.gauge}>
            <div style={styles.header}>
              <span style={styles.label}>{dim.label}</span>
              <span style={styles.infoIcon} title={dim.tooltip}>ⓘ</span>
            </div>
            <div style={styles.barBg}>
              <div
                style={{
                  ...styles.barFill,
                  width: fillWidth,
                  background: dim.color,
                }}
              />
            </div>
            <div style={styles.footer}>
              <span style={{ color: dim.color, fontWeight: 700, fontSize: 14 }}>
                {pct}{value != null ? '%' : ''}
              </span>
              <DeviationArrow value={value} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

const styles = {
  grid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 12,
  },
  gauge: {
    background: '#0d1117',
    borderRadius: 10,
    padding: 12,
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  label: {
    fontSize: 12,
    fontWeight: 600,
    color: '#c9d1d9',
  },
  infoIcon: {
    fontSize: 13,
    color: '#8b949e',
    cursor: 'help',
  },
  barBg: {
    width: '100%',
    height: 10,
    background: '#21262d',
    borderRadius: 5,
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: 5,
    transition: 'width 300ms ease',
  },
  footer: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 4,
  },
};
