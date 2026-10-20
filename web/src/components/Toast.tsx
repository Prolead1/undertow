import { cn } from "./cn";

export type ToastTone = "info" | "success" | "error";

export interface ToastMessage {
  id: string;
  message: string;
  tone?: ToastTone;
  /**
   * Whether this toast is announced through a screen-reader live region.
   * Set `false` for toasts that merely restate a connection state already
   * announced by another live region (e.g. the connection badge), so the same
   * state is never announced twice.
   */
  announce?: boolean;
}

export interface ToastProps {
  message: string;
  tone?: ToastTone;
  onDismiss?: () => void;
  className?: string;
  announce?: boolean;
}

const TONE: Record<ToastTone, string> = {
  info: "border-accent/50 text-text-primary",
  success: "border-bull/50 text-text-primary",
  error: "border-bear/60 text-text-primary",
};

export function Toast({
  message,
  tone = "info",
  onDismiss,
  className,
  announce = true,
}: ToastProps) {
  const isError = tone === "error";
  return (
    <div
      // Errors are assertive (`role="alert"`); everything else is a polite
      // status region. `announce={false}` renders a visual-only toast so it
      // cannot duplicate a live region that already reports the same state.
      role={announce ? (isError ? "alert" : "status") : undefined}
      aria-live={announce && !isError ? "polite" : undefined}
      aria-hidden={announce ? undefined : true}
      className={cn(
        "flex items-center gap-3 rounded-control border bg-bg-elevated px-3 py-2 text-xs shadow-[0_1px_0_rgba(255,255,255,0.03)]",
        TONE[tone],
        className,
      )}
    >
      <span className="flex-1">{message}</span>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          // A toast hidden from assistive tech must not expose a focusable
          // control either, or `aria-hidden` would wrap focusable content.
          tabIndex={announce ? undefined : -1}
          aria-label="Dismiss notification"
          className="rounded-control px-1 text-text-muted hover:text-text-primary"
        >
          ×
        </button>
      )}
    </div>
  );
}

export interface ToastRegionProps {
  toasts: readonly ToastMessage[];
  onDismiss: (id: string) => void;
}

export function ToastRegion({ toasts, onDismiss }: ToastRegionProps) {
  if (toasts.length === 0) return null;
  return (
    <div className="fixed bottom-4 right-4 z-50 flex w-80 flex-col gap-2">
      {toasts.map((toast) => (
        <Toast
          key={toast.id}
          message={toast.message}
          tone={toast.tone}
          announce={toast.announce}
          onDismiss={() => onDismiss(toast.id)}
        />
      ))}
    </div>
  );
}
