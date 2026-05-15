import { cn } from "@/lib/utils";

function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "shimmer rounded-md bg-muted/40 motion-reduce:bg-primary/10 motion-reduce:animate-pulse",
        className,
      )}
      {...props}
    />
  );
}

export { Skeleton };
