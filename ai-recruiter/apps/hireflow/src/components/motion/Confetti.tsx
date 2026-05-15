import { motion, useReducedMotion } from "framer-motion";
import { useMemo } from "react";

const PALETTE = [
  "oklch(0.7 0.2 295)",   // accent violet
  "oklch(0.72 0.16 155)", // success green
  "oklch(0.8 0.16 80)",   // warning amber
  "oklch(0.65 0.2 230)",  // chart blue
];

/**
 * Lightweight celebratory burst — no extra packages, just framer-motion.
 * Fires once on mount; consumer keys this with the success state to
 * replay. Position absolutely inside a relatively-positioned parent.
 */
export function Confetti({
  count = 22,
  spread = 220,
  duration = 1.3,
}: {
  count?: number;
  spread?: number;
  duration?: number;
}) {
  const reduced = useReducedMotion();
  const pieces = useMemo(() => {
    return Array.from({ length: count }).map((_, i) => {
      const angle = (i / count) * Math.PI * 2 + (Math.random() - 0.5) * 0.3;
      const radius = spread * (0.55 + Math.random() * 0.45);
      return {
        i,
        dx: Math.cos(angle) * radius,
        dy: Math.sin(angle) * radius - spread * 0.15, // bias upward
        rot: 360 * (Math.random() > 0.5 ? 1 : -1),
        color: PALETTE[i % PALETTE.length],
        delay: Math.random() * 0.08,
        size: 6 + Math.random() * 6,
      };
    });
  }, [count, spread]);

  if (reduced) return null;

  return (
    <div className="pointer-events-none absolute inset-0 overflow-visible">
      {pieces.map((p) => (
        <motion.span
          key={p.i}
          initial={{ opacity: 1, x: 0, y: 0, rotate: 0, scale: 0.6 }}
          animate={{
            opacity: [1, 1, 0],
            x: p.dx,
            y: p.dy,
            rotate: p.rot,
            scale: [0.6, 1, 0.85],
          }}
          transition={{
            duration,
            delay: p.delay,
            ease: [0.16, 0.84, 0.32, 1],
          }}
          style={{
            position: "absolute",
            left: "50%",
            top: "50%",
            width: p.size,
            height: p.size * 0.4,
            borderRadius: 2,
            background: p.color,
            boxShadow: `0 0 8px ${p.color}55`,
          }}
        />
      ))}
    </div>
  );
}
