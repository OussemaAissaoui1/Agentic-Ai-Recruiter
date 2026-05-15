import { useEffect, useRef } from "react";

// Returns a ref to attach to the modal container. When `active` is true, Tab
// and Shift-Tab cycle through focusable descendants inside the container
// instead of leaking to elements behind the dialog. The previously focused
// element is restored when the trap deactivates.
//
// Focusable selector mirrors the WICG `:focusable` pseudo-class — see
// https://wicg.github.io/web-share-target/#focusable for the canonical set.
const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "textarea:not([disabled])",
  "input:not([disabled]):not([type='hidden'])",
  "select:not([disabled])",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

export function useFocusTrap<T extends HTMLElement>(active: boolean) {
  const ref = useRef<T | null>(null);

  useEffect(() => {
    if (!active) return;
    const container = ref.current;
    if (!container) return;

    const previouslyFocused = document.activeElement as HTMLElement | null;

    const focusables = (): HTMLElement[] =>
      Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR))
        .filter((el) => !el.hasAttribute("inert") && el.offsetParent !== null);

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;
      const items = focusables();
      if (items.length === 0) {
        e.preventDefault();
        return;
      }
      const first = items[0];
      const last = items[items.length - 1];
      const current = document.activeElement as HTMLElement | null;
      if (e.shiftKey && current === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && current === last) {
        e.preventDefault();
        first.focus();
      }
    };

    container.addEventListener("keydown", onKeyDown);
    return () => {
      container.removeEventListener("keydown", onKeyDown);
      // Restore focus to whatever opened the modal — keyboard users land back
      // where they expect instead of at <body>.
      previouslyFocused?.focus?.();
    };
  }, [active]);

  return ref;
}
