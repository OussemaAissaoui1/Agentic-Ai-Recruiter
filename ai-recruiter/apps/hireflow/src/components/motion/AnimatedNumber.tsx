import { useEffect, useRef, useState } from "react";

/**
 * Counts smoothly from `from` to `value` over `duration` ms with an
 * ease-out curve. Re-animates when `value` changes. Respects
 * prefers-reduced-motion.
 */
export function AnimatedNumber({
  value,
  from = 0,
  duration = 900,
  format = (n) => Math.round(n).toString(),
  className,
}: {
  value: number;
  from?: number;
  duration?: number;
  format?: (n: number) => string;
  className?: string;
}) {
  const [display, setDisplay] = useState<number>(from);
  const startedAt = useRef<number | null>(null);
  const startFrom = useRef<number>(from);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    if (reduced) {
      setDisplay(value);
      return;
    }
    let raf = 0;
    startedAt.current = null;
    startFrom.current = display;
    const target = value;
    const tick = (now: number) => {
      if (startedAt.current === null) startedAt.current = now;
      const t = Math.min(1, (now - startedAt.current) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const current = startFrom.current + (target - startFrom.current) * eased;
      setDisplay(current);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, duration]);

  return <span className={className}>{format(display)}</span>;
}
