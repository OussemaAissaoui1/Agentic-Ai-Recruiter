import { useState } from 'react';
import { useStressDetection } from './hooks/useStressDetection';
import VideoOverlay from './components/VideoOverlay';
import StressGauge from './components/StressGauge';
import DetectionClasses from './components/DetectionClasses';
import StatsPanel from './components/StatsPanel';
import DimensionGauges from './components/DimensionGauges';
import SessionTimeline from './components/SessionTimeline';
import ReportPanel from './components/ReportPanel';

// Behavior dashboard connects to the unified app's vision WebSocket.
const WS_URL =
  (window.location.protocol === 'https:' ? 'wss' : 'ws') +
  '://' +
  window.location.host +
  '/ws/vision';

function App() {
  const [enableAudio, setEnableAudio] = useState(true);
  const [questionInput, setQuestionInput] = useState('');
  const {
    videoRef,
    canvasRef,
    connected,
    streaming,
    latestResult,
    stats,
    fps,
    setFps,
    start,
    stop,
    // New
    calibrationProgress,
    isCalibrated,
    dimensionScores,
    notableMoments,
    sessionId,
    sessionStats,
    markQuestion,
  } = useStressDetection(WS_URL);

  const handleStart = () => start(enableAudio);

  const handleMarkQuestion = () => {
    if (questionInput.trim()) {
      markQuestion(questionInput.trim());
      setQuestionInput('');
    }
  };

  // Build timeline data from session stats
  const timelineData = sessionStats?.per_dimension
    ? [] // SessionTimeline receives data via sessionStats updates
    : [];

  return (
    <div style={styles.root}>
      {/* Header */}
      <header style={styles.header}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 24 }}>🧠</span>
          <h1 style={{ fontSize: 18, fontWeight: 700, color: '#f0f0f0', margin: 0 }}>
            Stress Detection — Real-Time Analysis
          </h1>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span
            style={{
              ...styles.badge,
              background: connected ? '#1a7f37' : streaming ? '#9e6a03' : '#da3633',
            }}
          >
            {connected ? 'Connected' : streaming ? 'Connecting...' : 'Disconnected'}
          </span>
          {streaming && (
            <span style={{ ...styles.badge, background: '#1f6feb' }}>
              {fps} FPS
            </span>
          )}
          {isCalibrated && (
            <span style={{ ...styles.badge, background: '#238636' }}>
              Calibrated
            </span>
          )}
        </div>
      </header>

      <div style={styles.main}>
        {/* Left: Video + Overlay */}
        <div style={styles.videoPanel}>
          <div style={styles.videoContainer}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={styles.video}
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            <VideoOverlay videoRef={videoRef} result={latestResult} />

            {!streaming && (
              <div style={styles.videoPlaceholder}>
                <span style={{ fontSize: 48 }}>📷</span>
                <p style={{ color: '#8b949e', marginTop: 12 }}>
                  Click <b>Start Detection</b> to begin real-time analysis
                </p>
              </div>
            )}
          </div>

          {/* Controls */}
          <div style={styles.controls}>
            {!streaming ? (
              <button style={styles.btnPrimary} onClick={handleStart}>
                ▶ Start Detection
              </button>
            ) : (
              <button style={styles.btnDanger} onClick={stop}>
                ■ Stop
              </button>
            )}

            <div style={styles.controlGroup}>
              <label style={styles.controlLabel}>FPS</label>
              <select
                value={fps}
                onChange={(e) => setFps(Number(e.target.value))}
                style={styles.select}
              >
                {[5, 10, 15, 20, 30].map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </div>

            <div style={styles.controlGroup}>
              <label style={styles.controlLabel}>
                <input
                  type="checkbox"
                  checked={enableAudio}
                  onChange={(e) => setEnableAudio(e.target.checked)}
                  disabled={streaming}
                  style={{ marginRight: 6 }}
                />
                Audio
              </label>
            </div>
          </div>

          {/* Mark Question control */}
          {streaming && (
            <div style={{ ...styles.controls, marginTop: 8 }}>
              <input
                type="text"
                value={questionInput}
                onChange={(e) => setQuestionInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleMarkQuestion()}
                placeholder="e.g. Tell me about yourself"
                style={styles.questionInput}
              />
              <button
                style={styles.btnQuestion}
                onClick={handleMarkQuestion}
                disabled={!questionInput.trim()}
              >
                Mark Question
              </button>
            </div>
          )}

          {/* Detection Classes */}
          <div style={styles.panel}>
            <h2 style={styles.panelTitle}>Active Detections & Classes</h2>
            <DetectionClasses result={latestResult} />
          </div>
        </div>

        {/* Right: Dashboard */}
        <div style={styles.sidebar}>
          {/* Dimension Gauges */}
          <div style={styles.panel}>
            <h2 style={styles.panelTitle}>Behavioral Dimensions</h2>
            <DimensionGauges scores={dimensionScores} />
          </div>

          {/* Per-Modality (keep existing) */}
          <div style={styles.panel}>
            <h2 style={styles.panelTitle}>Per-Modality Breakdown</h2>
            <StressGauge
              title="👁 Visual (Face/Body)"
              label={latestResult?.visual?.label}
              probability={latestResult?.visual?.probabilities?.stressed}
              confidence={latestResult?.visual?.confidence}
            />
            <StressGauge
              title="🎤 Audio (Voice)"
              label={latestResult?.audio?.label}
              probability={latestResult?.audio?.probabilities?.stressed}
              confidence={latestResult?.audio?.confidence}
            />
          </div>

          {/* Emotion Breakdown (HSEmotion) */}
          {latestResult?.visual?.emotions && (
            <div style={styles.panel}>
              <h2 style={styles.panelTitle}>Emotion Analysis (HSEmotion)</h2>
              <div style={{ fontSize: 13, color: '#8b949e', marginBottom: 8 }}>
                Valence:{' '}
                <b
                  style={{
                    color:
                      latestResult.visual.valence > 0 ? '#3fb950' : '#f85149',
                  }}
                >
                  {latestResult.visual.valence?.toFixed(2)}
                </b>
                {' | '}
                Arousal:{' '}
                <b
                  style={{
                    color:
                      latestResult.visual.arousal > 0.3 ? '#f85149' : '#3fb950',
                  }}
                >
                  {latestResult.visual.arousal?.toFixed(2)}
                </b>
              </div>
              {Object.entries(latestResult.visual.emotions)
                .sort((a, b) => b[1] - a[1])
                .map(([emo, prob]) => (
                  <div key={emo} style={{ marginBottom: 6 }}>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: 12,
                      }}
                    >
                      <span style={{ color: '#c9d1d9' }}>{emo}</span>
                      <span style={{ color: '#8b949e' }}>
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div
                      style={{
                        height: 8,
                        background: '#21262d',
                        borderRadius: 4,
                        overflow: 'hidden',
                      }}
                    >
                      <div
                        style={{
                          width: `${prob * 100}%`,
                          height: '100%',
                          background:
                            emo === 'Happiness'
                              ? '#3fb950'
                              : ['Anger', 'Fear', 'Sadness'].includes(emo)
                              ? '#f85149'
                              : '#58a6ff',
                          borderRadius: 4,
                          transition: 'width 0.2s',
                        }}
                      />
                    </div>
                  </div>
                ))}
            </div>
          )}

          {/* Session Timeline (replaces StressTimeline) */}
          <div style={styles.panel}>
            <h2 style={styles.panelTitle}>Session Timeline</h2>
            <SessionTimeline
              timelineData={
                dimensionScores
                  ? stats.history.map((h, i) => ({
                      timestamp: (h.t - (stats.history[0]?.t || 0)) / 1000,
                      cognitive_load: dimensionScores?.cognitive_load || 0,
                      emotional_arousal: dimensionScores?.emotional_arousal || 0,
                      engagement_level: dimensionScores?.engagement_level || 0,
                      confidence_level: dimensionScores?.confidence_level || 0,
                    }))
                  : []
              }
              notableMoments={notableMoments}
              height={140}
            />
          </div>

          {/* Report Panel */}
          <ReportPanel
            sessionId={sessionId}
            sessionStats={sessionStats}
            isCalibrated={isCalibrated}
            markQuestion={markQuestion}
          />

          {/* Stats */}
          <div style={{ ...styles.panel, marginTop: 16 }}>
            <h2 style={styles.panelTitle}>Session Statistics</h2>
            <StatsPanel stats={stats} isCalibrated={isCalibrated} />
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  root: {
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    background: '#0f1117',
    color: '#e0e0e0',
    minHeight: '100vh',
    margin: 0,
  },
  header: {
    background: '#161b22',
    borderBottom: '1px solid #30363d',
    padding: '12px 24px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  badge: {
    padding: '4px 12px',
    borderRadius: 12,
    fontSize: 12,
    fontWeight: 600,
    color: '#fff',
  },
  main: {
    display: 'flex',
    gap: 20,
    padding: 20,
    maxWidth: 1400,
    margin: '0 auto',
  },
  videoPanel: { flex: 1, minWidth: 0 },
  videoContainer: {
    position: 'relative',
    background: '#000',
    borderRadius: 12,
    overflow: 'hidden',
    border: '1px solid #30363d',
    aspectRatio: '640/480',
  },
  video: { width: '100%', height: '100%', objectFit: 'cover', display: 'block' },
  videoPlaceholder: {
    position: 'absolute',
    top: 0, left: 0, width: '100%', height: '100%',
    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    background: 'rgba(0,0,0,0.8)',
  },
  controls: { marginTop: 12, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' },
  btnPrimary: {
    padding: '10px 24px', border: 'none', borderRadius: 8,
    fontSize: 14, fontWeight: 600, cursor: 'pointer', background: '#238636', color: '#fff',
  },
  btnDanger: {
    padding: '10px 24px', border: 'none', borderRadius: 8,
    fontSize: 14, fontWeight: 600, cursor: 'pointer', background: '#da3633', color: '#fff',
  },
  controlGroup: { display: 'flex', alignItems: 'center', gap: 6 },
  controlLabel: { fontSize: 13, color: '#8b949e', display: 'flex', alignItems: 'center' },
  select: {
    background: '#0d1117', color: '#e0e0e0', border: '1px solid #30363d',
    borderRadius: 6, padding: '4px 8px', fontSize: 13,
  },
  questionInput: {
    flex: 1, background: '#0d1117', color: '#e0e0e0', border: '1px solid #30363d',
    borderRadius: 6, padding: '8px 10px', fontSize: 13, outline: 'none',
  },
  btnQuestion: {
    padding: '8px 16px', border: 'none', borderRadius: 8,
    fontSize: 13, fontWeight: 600, cursor: 'pointer', background: '#1f6feb', color: '#fff',
    whiteSpace: 'nowrap',
  },
  sidebar: { width: 380, flexShrink: 0 },
  panel: {
    background: '#161b22', border: '1px solid #30363d',
    borderRadius: 12, padding: 16, marginBottom: 16,
  },
  panelTitle: { fontSize: 14, fontWeight: 600, color: '#f0f0f0', margin: '0 0 12px 0' },
};

export default App;
