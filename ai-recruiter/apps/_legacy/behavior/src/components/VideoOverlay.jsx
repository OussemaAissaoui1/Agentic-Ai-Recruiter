import { useRef, useEffect } from 'react';

const FACE_COLOR = '#00ff88';
const BODY_COLOR = '#00aaff';
const STRESSED_COLOR = '#ff4444';
const LANDMARK_RADIUS = 3;

export default function VideoOverlay({ videoRef, result, width = 640, height = 480 }) {
  const overlayRef = useRef(null);

  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    if (!result) return;

    const detections = result.detections;
    const isStressed = result.fused?.label === 'stressed';
    const stressProb = result.fused?.probabilities?.stressed ?? 0;

    // ── Draw face bounding box ──
    if (detections?.face?.face_bbox) {
      const bb = detections.face.face_bbox;
      const x = bb.x * width;
      const y = bb.y * height;
      const w = bb.w * width;
      const h = bb.h * height;

      const color = isStressed ? STRESSED_COLOR : FACE_COLOR;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);

      // Label — show emotion + stress
      const emotion = result.visual?.emotion || '';
      const stressLabel = result.visual?.label || 'detecting...';
      const conf = result.visual?.confidence
        ? `${(result.visual.confidence * 100).toFixed(0)}%`
        : '';
      const text = emotion
        ? `${emotion} | ${stressLabel} ${conf}`
        : `FACE: ${stressLabel} ${conf}`;

      ctx.font = 'bold 13px monospace';
      const tm = ctx.measureText(text);
      const labelH = 20;

      ctx.fillStyle = color;
      ctx.fillRect(x, y - labelH, tm.width + 10, labelH);
      ctx.fillStyle = '#000';
      ctx.fillText(text, x + 5, y - 5);
    }

    // ── Draw face landmarks ──
    if (detections?.face?.face_landmarks) {
      ctx.fillStyle = isStressed ? STRESSED_COLOR : FACE_COLOR;
      for (const lm of detections.face.face_landmarks) {
        ctx.beginPath();
        ctx.arc(lm.x * width, lm.y * height, LANDMARK_RADIUS, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // ── Draw body bounding box ──
    if (detections?.body?.body_bbox) {
      const bb = detections.body.body_bbox;
      const x = bb.x * width;
      const y = bb.y * height;
      const w = bb.w * width;
      const h = bb.h * height;

      ctx.strokeStyle = BODY_COLOR;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);

      // Label
      ctx.font = 'bold 12px monospace';
      const text = 'BODY POSE';
      const tm = ctx.measureText(text);
      ctx.fillStyle = BODY_COLOR;
      ctx.fillRect(x, y + h, tm.width + 10, 18);
      ctx.fillStyle = '#000';
      ctx.fillText(text, x + 5, y + h + 14);
    }

    // ── Draw body landmarks + skeleton ──
    if (detections?.body?.body_landmarks) {
      const lms = detections.body.body_landmarks;
      ctx.fillStyle = BODY_COLOR;
      for (const lm of lms) {
        ctx.beginPath();
        ctx.arc(lm.x * width, lm.y * height, 4, 0, Math.PI * 2);
        ctx.fill();
      }

      // Connect skeleton: shoulders(0,1), elbows(2,3), wrists(4,5), hips(6,7)
      const connections = [
        [0, 1], // shoulders
        [0, 2], [2, 4], // left arm
        [1, 3], [3, 5], // right arm
        [0, 6], [1, 7], // torso
        [6, 7], // hips
      ];
      ctx.strokeStyle = BODY_COLOR;
      ctx.lineWidth = 2;
      for (const [a, b] of connections) {
        if (lms[a] && lms[b]) {
          ctx.beginPath();
          ctx.moveTo(lms[a].x * width, lms[a].y * height);
          ctx.lineTo(lms[b].x * width, lms[b].y * height);
          ctx.stroke();
        }
      }
    }

    // ── Fused stress bar at top ──
    if (result.fused && result.fused.label !== 'unknown') {
      const barW = 200;
      const barH = 24;
      const barX = width - barW - 10;
      const barY = 10;

      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(barX - 5, barY - 2, barW + 10, barH + 18);

      // Bar background
      ctx.fillStyle = '#333';
      ctx.fillRect(barX, barY, barW, barH);

      // Bar fill
      const fillW = stressProb * barW;
      const gradient = ctx.createLinearGradient(barX, 0, barX + barW, 0);
      gradient.addColorStop(0, '#3fb950');
      gradient.addColorStop(0.5, '#d29922');
      gradient.addColorStop(1, '#f85149');
      ctx.fillStyle = gradient;
      ctx.fillRect(barX, barY, fillW, barH);

      // Text
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 12px monospace';
      ctx.fillText(
        `STRESS: ${(stressProb * 100).toFixed(1)}%`,
        barX + 5,
        barY + 16
      );

      // Modalities
      ctx.font = '10px monospace';
      ctx.fillStyle = '#aaa';
      ctx.fillText(
        (result.fused.modalities_used || []).join(' + '),
        barX,
        barY + barH + 12
      );
    }
  }, [result, width, height]);

  return (
    <canvas
      ref={overlayRef}
      width={width}
      height={height}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    />
  );
}
