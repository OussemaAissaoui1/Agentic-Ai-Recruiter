import { useEffect, useState } from "react";

export function useStreamingText(full: string, speed = 18, deps: any[] = []) {
  const [text, setText] = useState("");
  const [done, setDone] = useState(false);
  useEffect(() => {
    setText("");
    setDone(false);
    let i = 0;
    const id = setInterval(() => {
      i += Math.max(1, Math.round(full.length / 240));
      if (i >= full.length) {
        setText(full);
        setDone(true);
        clearInterval(id);
      } else {
        setText(full.slice(0, i));
      }
    }, speed);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [full, ...deps]);
  return { text, done };
}

export function useTicker(intervalMs = 1500) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return tick;
}
