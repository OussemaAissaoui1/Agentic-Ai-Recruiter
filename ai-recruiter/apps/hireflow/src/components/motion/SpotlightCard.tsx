import { useRef } from "react";
import { cn } from "@/lib/utils";

/**
 * A wrapper that paints a soft cursor-following violet spotlight on
 * hover. Composes with any styled card; just supply a className.
 *
 *   <SpotlightCard className="rounded-2xl border bg-card p-5 lift">
 *     ...
 *   </SpotlightCard>
 */
export function SpotlightCard({
  as: Tag = "div",
  className,
  children,
  ...rest
}: {
  as?: keyof React.JSX.IntrinsicElements;
  className?: string;
  children: React.ReactNode;
} & React.HTMLAttributes<HTMLElement>) {
  const ref = useRef<HTMLElement | null>(null);

  const onMove = (e: React.MouseEvent<HTMLElement>) => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    el.style.setProperty("--mx", `${e.clientX - r.left}px`);
    el.style.setProperty("--my", `${e.clientY - r.top}px`);
  };

  // Cast through `any` because polymorphic `as` + ref is awkward in TS.
  const Comp = Tag as unknown as React.ElementType;
  return (
    <Comp
      ref={ref as never}
      onMouseMove={onMove}
      className={cn("spotlight", className)}
      {...rest}
    >
      {children}
    </Comp>
  );
}
