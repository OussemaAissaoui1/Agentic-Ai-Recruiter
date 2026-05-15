import * as DialogPrimitive from "@radix-ui/react-dialog";
import {
  AlertTriangle,
  Download,
  ExternalLink,
  FileText,
  Loader2,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";

const CV_HEIGHT = "min(82vh, 900px)";

type ProbeState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ok" }
  | { status: "missing" }
  | { status: "error"; code: number; message: string };

/**
 * Centered modal that previews the candidate's original uploaded CV
 * (PDF, DOCX, image, or txt) — the source file the candidate sent, not
 * the OCR'd text. Backed by `GET /api/recruit/applications/{id}/cv`,
 * which serves the file with `Content-Disposition: inline` so the
 * browser's native PDF/image viewer renders it directly.
 *
 * Probes the endpoint with a HEAD request before mounting the iframe so
 * the user gets a friendly empty state instead of "{detail: 'Not Found'}"
 * for legacy applications that pre-date the cv_path column.
 */
export function CvViewer({
  open,
  onOpenChange,
  applicationId,
  filename,
}: {
  open: boolean;
  onOpenChange: (next: boolean) => void;
  applicationId: string;
  filename?: string | null;
}) {
  const cvUrl = `/api/recruit/applications/${encodeURIComponent(applicationId)}/cv`;
  const ext = (filename ?? "").split(".").pop()?.toLowerCase() ?? "";
  // PDFs + plain text + common images all preview natively in an iframe.
  // DOCX usually triggers a download — we surface that explicitly below.
  const previewable =
    ext === "pdf" ||
    ext === "txt" ||
    ext === "png" ||
    ext === "jpg" ||
    ext === "jpeg" ||
    ext === "" /* unknown — try iframe anyway */;

  const [probe, setProbe] = useState<ProbeState>({ status: "idle" });

  useEffect(() => {
    if (!open) {
      setProbe({ status: "idle" });
      return;
    }
    let cancelled = false;
    setProbe({ status: "loading" });
    // HEAD avoids streaming the whole file just to check existence.
    fetch(cvUrl, { method: "HEAD" })
      .then(async (res) => {
        if (cancelled) return;
        if (res.ok) {
          setProbe({ status: "ok" });
        } else if (res.status === 404) {
          setProbe({ status: "missing" });
        } else {
          setProbe({
            status: "error",
            code: res.status,
            message: res.statusText || `HTTP ${res.status}`,
          });
        }
      })
      .catch((e) => {
        if (cancelled) return;
        setProbe({
          status: "error",
          code: 0,
          message: e instanceof Error ? e.message : String(e),
        });
      });
    return () => {
      cancelled = true;
    };
  }, [open, cvUrl]);

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay
          className="fixed inset-0 z-50 bg-foreground/40 backdrop-blur-sm data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0"
        />
        <DialogPrimitive.Content
          className="fixed left-1/2 top-1/2 z-50 flex w-[min(95vw,1100px)] max-w-[95vw] -translate-x-1/2 -translate-y-1/2 flex-col gap-3 rounded-2xl border border-border bg-card p-4 shadow-card-soft duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95"
          aria-describedby={undefined}
        >
          <header className="flex items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-2">
              <FileText className="h-4 w-4 shrink-0 text-accent" />
              <DialogPrimitive.Title className="truncate text-sm font-semibold">
                {filename || "Candidate CV"}
              </DialogPrimitive.Title>
            </div>
            <div className="flex shrink-0 items-center gap-1">
              {probe.status === "ok" && (
                <>
                  <a
                    href={cvUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 rounded-full border border-border bg-background px-2.5 py-1 text-xs text-muted-foreground hover:bg-muted"
                    title="Open in a new tab"
                  >
                    <ExternalLink className="h-3.5 w-3.5" />
                    Open
                  </a>
                  <a
                    href={cvUrl}
                    download={filename ?? "cv"}
                    className="inline-flex items-center gap-1 rounded-full border border-border bg-background px-2.5 py-1 text-xs text-muted-foreground hover:bg-muted"
                    title="Download original file"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Download
                  </a>
                </>
              )}
              <DialogPrimitive.Close
                className="inline-flex h-7 w-7 items-center justify-center rounded-full text-muted-foreground hover:bg-muted hover:text-foreground"
                aria-label="Close CV preview"
              >
                <X className="h-4 w-4" />
              </DialogPrimitive.Close>
            </div>
          </header>

          {probe.status === "loading" && <BodyLoading />}

          {probe.status === "missing" && (
            <BodyEmpty>
              <FileText className="h-10 w-10 text-muted-foreground" />
              <div className="text-sm font-medium">
                No CV file on record for this application
              </div>
              <p className="max-w-md text-xs text-muted-foreground">
                This usually means the application was submitted before CV
                preview was wired into HireFlow. Once a candidate applies
                with a CV after the latest deploy, that file will preview
                here.
              </p>
            </BodyEmpty>
          )}

          {probe.status === "error" && (
            <BodyEmpty>
              <AlertTriangle className="h-10 w-10 text-destructive" />
              <div className="text-sm font-medium">
                Couldn't load this CV
              </div>
              <p className="max-w-md text-xs text-muted-foreground">
                The server returned an error fetching the file
                {probe.code ? ` (${probe.code})` : ""}: {probe.message}.
                If you just deployed the CV preview feature, restart the
                FastAPI server so the new route + schema migration are
                picked up.
              </p>
            </BodyEmpty>
          )}

          {probe.status === "ok" &&
            (previewable ? (
              <iframe
                key={cvUrl}
                src={cvUrl}
                title={filename ? `CV: ${filename}` : "Candidate CV"}
                className="w-full rounded-xl border border-border bg-background"
                style={{ height: CV_HEIGHT }}
              />
            ) : (
              <BodyEmpty>
                <FileText className="h-10 w-10 text-muted-foreground" />
                <div className="text-sm font-medium">
                  {ext ? `.${ext.toUpperCase()} files` : "This file"} can't
                  be previewed inline.
                </div>
                <p className="max-w-md text-xs text-muted-foreground">
                  Most browsers don't render this format directly. Use the
                  button below to download the original file.
                </p>
                <a
                  href={cvUrl}
                  download={filename ?? "cv"}
                  className="press-tight inline-flex items-center gap-1.5 rounded-full bg-violet-grad px-4 py-2 text-sm font-semibold text-accent-foreground shadow-glow transition-transform hover:scale-[1.02]"
                >
                  <Download className="h-4 w-4" />
                  Download CV
                </a>
              </BodyEmpty>
            ))}
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

function BodyLoading() {
  return (
    <div
      className="flex w-full flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border bg-background p-10 text-center"
      style={{ height: CV_HEIGHT }}
    >
      <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      <div className="text-xs text-muted-foreground">
        Locating the original file…
      </div>
    </div>
  );
}

function BodyEmpty({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="flex w-full flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border bg-background p-10 text-center"
      style={{ height: CV_HEIGHT }}
    >
      {children}
    </div>
  );
}
