import { useState } from 'react';
import { useSessionReport } from '../hooks/useSessionReport';

const ARC_COLORS = {
  improving: '#3fb950',
  declining: '#f85149',
  stable: '#58a6ff',
  variable: '#d29922',
};

const DIMENSION_LABELS = {
  cognitive_load: 'Cognitive Load',
  emotional_arousal: 'Emotional Arousal',
  engagement_level: 'Engagement Level',
  confidence_level: 'Confidence Level',
};

function TrendArrow({ trend }) {
  if (trend == null) return '—';
  if (trend > 0.001) return <span style={{ color: '#3fb950' }}>↑</span>;
  if (trend < -0.001) return <span style={{ color: '#f85149' }}>↓</span>;
  return <span style={{ color: '#8b949e' }}>→</span>;
}

export default function ReportPanel({
  sessionId,
  sessionStats,
  isCalibrated,
  markQuestion: markQuestionWs,
}) {
  const [expanded, setExpanded] = useState(false);
  const [questionInput, setQuestionInput] = useState('');
  const { isGenerating, downloadJSON, downloadPDF } = useSessionReport(sessionId);

  const handleMarkQuestion = () => {
    if (questionInput.trim()) {
      markQuestionWs(questionInput.trim());
      setQuestionInput('');
    }
  };

  const perDim = sessionStats?.per_dimension || {};
  const arc = sessionStats?.overall_arc || 'stable';
  const notableCount = sessionStats?.notable_moments?.length || 0;

  return (
    <div style={styles.panel}>
      <div style={styles.headerRow} onClick={() => setExpanded(!expanded)}>
        <h2 style={styles.title}>
          {expanded ? '▾' : '▸'} Session Report
        </h2>
        <span style={styles.badge}>
          {sessionStats?.frames_analyzed || 0} frames
        </span>
      </div>

      {expanded && (
        <div style={styles.body}>
          {/* Summary stats table */}
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Dimension</th>
                <th style={styles.th}>Mean</th>
                <th style={styles.th}>Trend</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(DIMENSION_LABELS).map(([key, label]) => {
                const dim = perDim[key];
                return (
                  <tr key={key}>
                    <td style={styles.td}>{label}</td>
                    <td style={styles.td}>
                      {dim ? `${(dim.mean * 100).toFixed(1)}%` : '--'}
                    </td>
                    <td style={styles.td}>
                      <TrendArrow trend={dim?.trend} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {/* Arc + notable */}
          <div style={styles.arcRow}>
            <span style={{ fontSize: 12, color: '#8b949e' }}>Overall Arc:</span>
            <span style={{
              ...styles.arcBadge,
              background: ARC_COLORS[arc] || '#555',
            }}>
              {arc.toUpperCase()}
            </span>
            <span style={{ fontSize: 12, color: '#8b949e', marginLeft: 12 }}>
              Notable moments: <b style={{ color: '#f0f0f0' }}>{notableCount}</b>
            </span>
          </div>

          {/* Controls */}
          <div style={styles.controls}>
            <div style={styles.questionRow}>
              <input
                type="text"
                value={questionInput}
                onChange={(e) => setQuestionInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleMarkQuestion()}
                placeholder="e.g. Tell me about yourself"
                style={styles.input}
                disabled={!isCalibrated}
              />
              <button
                style={styles.btnSmall}
                onClick={handleMarkQuestion}
                disabled={!isCalibrated || !questionInput.trim()}
              >
                Mark Question
              </button>
            </div>

            <div style={styles.downloadRow}>
              <button
                style={styles.btnDownload}
                onClick={downloadJSON}
                disabled={!isCalibrated || isGenerating}
              >
                Download JSON Report
              </button>
              <button
                style={{ ...styles.btnDownload, background: '#8B5CF6' }}
                onClick={downloadPDF}
                disabled={!isCalibrated || isGenerating}
              >
                {isGenerating ? '⟳ Generating...' : 'Download PDF Report'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  panel: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: 12,
    overflow: 'hidden',
  },
  headerRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px 16px',
    cursor: 'pointer',
    userSelect: 'none',
  },
  title: {
    fontSize: 14,
    fontWeight: 600,
    color: '#f0f0f0',
    margin: 0,
  },
  badge: {
    fontSize: 11,
    color: '#8b949e',
    background: '#21262d',
    padding: '2px 8px',
    borderRadius: 8,
  },
  body: {
    padding: '0 16px 16px',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    marginBottom: 12,
  },
  th: {
    textAlign: 'left',
    fontSize: 11,
    color: '#8b949e',
    paddingBottom: 6,
    borderBottom: '1px solid #21262d',
  },
  td: {
    fontSize: 12,
    color: '#c9d1d9',
    padding: '6px 0',
    borderBottom: '1px solid #21262d',
  },
  arcRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
  },
  arcBadge: {
    fontSize: 11,
    fontWeight: 700,
    color: '#fff',
    padding: '2px 10px',
    borderRadius: 6,
  },
  controls: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  questionRow: {
    display: 'flex',
    gap: 8,
  },
  input: {
    flex: 1,
    background: '#0d1117',
    color: '#e0e0e0',
    border: '1px solid #30363d',
    borderRadius: 6,
    padding: '6px 10px',
    fontSize: 12,
    outline: 'none',
  },
  btnSmall: {
    background: '#1f6feb',
    color: '#fff',
    border: 'none',
    borderRadius: 6,
    padding: '6px 12px',
    fontSize: 12,
    fontWeight: 600,
    cursor: 'pointer',
    whiteSpace: 'nowrap',
  },
  downloadRow: {
    display: 'flex',
    gap: 8,
  },
  btnDownload: {
    flex: 1,
    background: '#238636',
    color: '#fff',
    border: 'none',
    borderRadius: 6,
    padding: '8px 12px',
    fontSize: 12,
    fontWeight: 600,
    cursor: 'pointer',
  },
};
