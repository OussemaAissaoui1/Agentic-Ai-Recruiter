export default function DetectionClasses({ result }) {
  if (!result) return null;

  const classes = [];

  // Face detection + Emotion
  if (result.detections?.face?.face_bbox) {
    const emotion = result.visual?.emotion || 'Processing...';
    const stressLabel = result.visual?.label;
    const conf = result.visual?.confidence
      ? `${(result.visual.confidence * 100).toFixed(0)}%`
      : '';
    classes.push({
      id: 'face',
      label: 'Face Detected',
      icon: '👁',
      color: '#00ff88',
      details: [`${stressLabel} (${conf})`, `Emotion: ${emotion}`],
      active: true,
    });
  } else {
    classes.push({
      id: 'face',
      label: 'Face',
      icon: '👁',
      color: '#555',
      details: ['Not detected'],
      active: false,
    });
  }

  // Emotion breakdown (HSEmotion)
  if (result.visual?.emotions) {
    const emotions = result.visual.emotions;
    const sorted = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
    const top3 = sorted.slice(0, 3).map(([e, p]) => `${e}: ${(p * 100).toFixed(1)}%`);
    const valence = result.visual?.valence;
    const arousal = result.visual?.arousal;
    const details = [...top3];
    if (valence != null)
      details.push(
        `Valence: ${valence.toFixed(2)} | Arousal: ${arousal?.toFixed(2) ?? '?'}`
      );

    classes.push({
      id: 'emotion',
      label: `Emotion: ${result.visual.emotion}`,
      icon: getEmotionIcon(result.visual.emotion),
      color: getEmotionColor(result.visual.emotion),
      details,
      active: true,
    });
  }

  // Body detection
  if (result.detections?.body?.body_bbox) {
    classes.push({
      id: 'body',
      label: 'Body Pose',
      icon: '🏃',
      color: '#00aaff',
      details: [
        `${result.detections.body.body_landmarks?.length || 0} landmarks`,
      ],
      active: true,
    });
  } else {
    classes.push({
      id: 'body',
      label: 'Body Pose',
      icon: '🏃',
      color: '#555',
      details: ['Not detected'],
      active: false,
    });
  }

  // Audio
  if (result.audio) {
    classes.push({
      id: 'audio',
      label: 'Voice Stress',
      icon: '🎤',
      color: result.audio.label === 'stressed' ? '#f85149' : '#3fb950',
      details: [
        `${result.audio.label} (${(result.audio.confidence * 100).toFixed(0)}%)`,
      ],
      active: true,
    });
  } else {
    classes.push({
      id: 'audio',
      label: 'Voice Stress',
      icon: '🎤',
      color: '#555',
      details: ['Waiting for audio...'],
      active: false,
    });
  }
  if (result.fused && result.fused.label !== 'unknown') {
    const isStressed = result.fused.label === 'stressed';
    classes.push({
      id: 'fused',
      label: 'Fused Result',
      icon: '🧠',
      color: isStressed ? '#f85149' : '#3fb950',
      details: [
        `${result.fused.label} (${(result.fused.confidence * 100).toFixed(0)}%)`,
        `Stress: ${(result.fused.probabilities.stressed * 100).toFixed(1)}%`,
        `Sources: ${(result.fused.modalities_used || []).join(' + ')}`,
      ],
      active: true,
    });
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {classes.map((cls) => (
        <div
          key={cls.id}
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 10,
            padding: '10px 12px',
            background: cls.active ? '#161b22' : '#0d1117',
            borderRadius: 8,
            borderLeft: `3px solid ${cls.color}`,
            opacity: cls.active ? 1 : 0.5,
            transition: 'all 0.2s',
          }}
        >
          <span style={{ fontSize: 20 }}>{cls.icon}</span>
          <div style={{ flex: 1 }}>
            <div
              style={{
                fontSize: 13,
                fontWeight: 600,
                color: cls.active ? '#f0f0f0' : '#8b949e',
              }}
            >
              {cls.label}
              {cls.active && (
                <span
                  style={{
                    marginLeft: 8,
                    fontSize: 10,
                    padding: '2px 6px',
                    borderRadius: 4,
                    background: cls.color,
                    color: '#000',
                    fontWeight: 700,
                  }}
                >
                  LIVE
                </span>
              )}
            </div>
            {cls.details.map((d, i) => (
              <div key={i} style={{ fontSize: 12, color: '#8b949e', marginTop: 2 }}>
                {d}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function getEmotionIcon(emotion) {
  const map = {
    Anger: '😠',
    Contempt: '😏',
    Disgust: '🤢',
    Fear: '😨',
    Happiness: '😊',
    Neutral: '😐',
    Sadness: '😢',
    Surprise: '😲',
  };
  return map[emotion] || '🎭';
}

function getEmotionColor(emotion) {
  const map = {
    Anger: '#f85149',
    Contempt: '#d29922',
    Disgust: '#a371f7',
    Fear: '#f85149',
    Happiness: '#3fb950',
    Neutral: '#8b949e',
    Sadness: '#58a6ff',
    Surprise: '#d29922',
  };
  return map[emotion] || '#8b949e';
}
