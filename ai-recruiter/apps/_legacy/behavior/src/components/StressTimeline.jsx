import { useRef, useEffect } from 'react';

export default function StressTimeline({ history, width = 320, height = 100 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length < 2) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = 1;
    for (let y = 0; y <= 1; y += 0.25) {
      const py = height - y * height;
      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(width, py);
      ctx.stroke();
    }

    // Threshold line at 50%
    ctx.strokeStyle = '#d29922';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, height * 0.5);
    ctx.lineTo(width, height * 0.5);
    ctx.stroke();
    ctx.setLineDash([]);

    // Stress curve
    const step = width / Math.max(history.length - 1, 1);
    ctx.beginPath();
    ctx.moveTo(0, height - history[0].stress * height);
    for (let i = 1; i < history.length; i++) {
      ctx.lineTo(i * step, height - history[i].stress * height);
    }
    ctx.strokeStyle = '#f85149';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Fill under curve
    ctx.lineTo((history.length - 1) * step, height);
    ctx.lineTo(0, height);
    ctx.closePath();
    ctx.fillStyle = 'rgba(248, 81, 73, 0.1)';
    ctx.fill();

    // Labels
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px monospace';
    ctx.fillText('100%', 2, 12);
    ctx.fillText('50%', 2, height * 0.5 - 3);
    ctx.fillText('0%', 2, height - 3);
  }, [history, width, height]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ width: '100%', height, borderRadius: 8, display: 'block' }}
    />
  );
}
