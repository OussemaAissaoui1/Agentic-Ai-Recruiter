import { useRef, useEffect, useCallback } from 'react';

const LINE_COLORS = {
  cognitive_load: '#8B5CF6',
  emotional_arousal: '#F97316',
  engagement_level: '#14B8A6',
  confidence_level: '#3B82F6',
};

const DIMENSION_LABELS = {
  cognitive_load: 'Cognitive Load',
  emotional_arousal: 'Emotional Arousal',
  engagement_level: 'Engagement',
  confidence_level: 'Confidence',
};

const DIMS = ['cognitive_load', 'emotional_arousal', 'engagement_level', 'confidence_level'];

export default function SessionTimeline({
  timelineData = [],
  questionMarkers = [],
  notableMoments = [],
  height = 180,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const hoverRef = useRef(null);
  const rafRef = useRef(null);
  const dataRef = useRef(timelineData);
  const markersRef = useRef(questionMarkers);
  const notableRef = useRef(notableMoments);

  dataRef.current = timelineData;
  markersRef.current = questionMarkers;
  notableRef.current = notableMoments;

  const draw = useCallback((hoverX) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const data = dataRef.current;
    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 20, right: 15, bottom: 25, left: 40 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, w, h);

    if (data.length < 2) {
      ctx.fillStyle = '#8b949e';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for data...', w / 2, h / 2);
      return;
    }

    const maxTime = data[data.length - 1].timestamp;
    const minTime = data[0].timestamp;
    const timeRange = maxTime - minTime || 1;

    const toX = (t) => pad.left + ((t - minTime) / timeRange) * plotW;
    const toY = (v) => pad.top + (1 - v) * plotH;

    // Grid lines
    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = 1;
    for (const frac of [0.25, 0.5, 0.75]) {
      const y = toY(frac);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();
    }

    // Y-axis labels
    ctx.fillStyle = '#8b949e';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    for (const [frac, label] of [[1, '100%'], [0.75, '75%'], [0.5, '50%'], [0.25, '25%'], [0, '0%']]) {
      ctx.fillText(label, pad.left - 4, toY(frac) + 4);
    }

    // X-axis labels
    ctx.textAlign = 'center';
    const xTicks = 5;
    for (let i = 0; i <= xTicks; i++) {
      const t = minTime + (timeRange * i) / xTicks;
      const x = toX(t);
      ctx.fillText(`${t.toFixed(0)}s`, x, h - 5);
    }

    // Question markers
    const markers = markersRef.current;
    if (markers.length) {
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#d29922';
      ctx.lineWidth = 1;
      for (const [ts, label] of markers) {
        const relTs = ts - (data[0]?.timestamp ?? 0);
        if (relTs < minTime || relTs > maxTime) continue;
        const x = toX(relTs);
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, h - pad.bottom);
        ctx.stroke();

        ctx.save();
        ctx.translate(x + 3, pad.top + 2);
        ctx.rotate(-Math.PI / 4);
        ctx.fillStyle = '#d29922';
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(label, 0, 0);
        ctx.restore();
      }
      ctx.setLineDash([]);
    }

    // Dimension lines
    for (const dim of DIMS) {
      ctx.strokeStyle = LINE_COLORS[dim];
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const x = toX(data[i].timestamp);
        const y = toY(data[i][dim] ?? 0);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Notable moment dots
    const notable = notableRef.current;
    for (const nm of notable) {
      // Find closest data point
      let closest = data[0];
      let closestDist = Infinity;
      for (const d of data) {
        const dist = Math.abs(d.timestamp - nm.timestamp);
        if (dist < closestDist) {
          closestDist = dist;
          closest = d;
        }
      }
      const dim = nm.dimension;
      if (closest && dim in closest) {
        const x = toX(closest.timestamp);
        const y = toY(closest[dim]);
        ctx.fillStyle = LINE_COLORS[dim] || '#fff';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Legend (top-right)
    const legendX = w - pad.right - 130;
    const legendY = pad.top + 4;
    ctx.fillStyle = 'rgba(13,17,23,0.8)';
    ctx.fillRect(legendX - 6, legendY - 4, 135, DIMS.length * 16 + 8);
    for (let i = 0; i < DIMS.length; i++) {
      const dim = DIMS[i];
      const y = legendY + i * 16 + 10;
      ctx.fillStyle = LINE_COLORS[dim];
      ctx.fillRect(legendX, y - 6, 10, 10);
      ctx.fillStyle = '#c9d1d9';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(DIMENSION_LABELS[dim], legendX + 14, y + 3);
    }

    // Hover crosshair + tooltip
    if (hoverX != null && hoverX >= pad.left && hoverX <= w - pad.right) {
      const hoverTime = minTime + ((hoverX - pad.left) / plotW) * timeRange;
      // Find closest data point
      let closestIdx = 0;
      let closestDist = Infinity;
      for (let i = 0; i < data.length; i++) {
        const dist = Math.abs(data[i].timestamp - hoverTime);
        if (dist < closestDist) {
          closestDist = dist;
          closestIdx = i;
        }
      }
      const pt = data[closestIdx];
      const xLine = toX(pt.timestamp);

      // Vertical line
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(xLine, pad.top);
      ctx.lineTo(xLine, h - pad.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      // Tooltip
      const tooltipX = xLine + 10 > w - 140 ? xLine - 140 : xLine + 10;
      ctx.fillStyle = 'rgba(22,27,34,0.95)';
      ctx.fillRect(tooltipX, pad.top, 130, 80);
      ctx.strokeStyle = '#30363d';
      ctx.strokeRect(tooltipX, pad.top, 130, 80);
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      ctx.fillStyle = '#8b949e';
      ctx.fillText(`t=${pt.timestamp.toFixed(1)}s`, tooltipX + 6, pad.top + 14);
      for (let i = 0; i < DIMS.length; i++) {
        const dim = DIMS[i];
        ctx.fillStyle = LINE_COLORS[dim];
        ctx.fillText(
          `${DIMENSION_LABELS[dim].slice(0, 10)}: ${((pt[dim] ?? 0) * 100).toFixed(1)}%`,
          tooltipX + 6,
          pad.top + 28 + i * 14
        );
      }
    }
  }, []);

  // Handle mouse move for hover
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      hoverRef.current = (e.clientX - rect.left) * (canvas.width / rect.width);
    };
    const onLeave = () => {
      hoverRef.current = null;
    };
    canvas.addEventListener('mousemove', onMove);
    canvas.addEventListener('mouseleave', onLeave);
    return () => {
      canvas.removeEventListener('mousemove', onMove);
      canvas.removeEventListener('mouseleave', onLeave);
    };
  }, []);

  // Animation loop
  useEffect(() => {
    let running = true;
    const loop = () => {
      if (!running) return;
      draw(hoverRef.current);
      rafRef.current = requestAnimationFrame(loop);
    };
    loop();
    return () => {
      running = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [draw]);

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        canvas.width = w * 2; // retina
        canvas.height = height * 2;
      }
    });
    observer.observe(container);
    // Init
    canvas.width = container.clientWidth * 2;
    canvas.height = height * 2;
    return () => observer.disconnect();
  }, [height]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height,
          borderRadius: 8,
          display: 'block',
          cursor: 'crosshair',
        }}
      />
    </div>
  );
}
