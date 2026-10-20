import { useState } from "react";

import type { RunRecord } from "../../api/client";
import { cn } from "../cn";

export interface ProvenanceCardProps {
  run: RunRecord;
  className?: string;
}

/** Clipboard write with a graceful no-op when the API is unavailable. */
async function writeClipboard(value: string): Promise<boolean> {
  const clipboard = typeof navigator !== "undefined" ? navigator.clipboard : undefined;
  if (!clipboard || typeof clipboard.writeText !== "function") return false;
  try {
    await clipboard.writeText(value);
    return true;
  } catch {
    return false;
  }
}

interface CopyButtonProps {
  value: string;
  label: string;
}

function CopyButton({ value, label }: CopyButtonProps) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      aria-label={`Copy ${label}`}
      onClick={() => {
        void writeClipboard(value).then((ok) => {
          if (ok) setCopied(true);
        });
      }}
      className="rounded-control border border-border px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-text-muted hover:border-accent hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
    >
      {copied ? "copied" : "copy"}
    </button>
  );
}

interface FieldProps {
  label: string;
  value: string | null | undefined;
  mono?: boolean;
  copyable?: boolean;
}

function Field({ label, value, mono = true, copyable = true }: FieldProps) {
  const text = value && value.length > 0 ? value : "—";
  return (
    <div className="min-w-0">
      <div className="flex items-center gap-2">
        <dt className="text-[11px] uppercase tracking-wide text-text-muted">{label}</dt>
        {copyable && text !== "—" && <CopyButton value={text} label={label} />}
      </div>
      <dd
        className={cn(
          "mt-0.5 break-all text-xs text-text-primary",
          mono && "font-mono",
        )}
      >
        {text}
      </dd>
    </div>
  );
}

/**
 * "Can I trust this number?" — the full, untruncated provenance of a run
 * (`config_hash`, git commit, library versions, checkpoint path) with copy
 * buttons for citation.
 */
export function ProvenanceCard({ run, className }: ProvenanceCardProps) {
  const manifest = run.manifest;
  const configHash = run.config_hash ?? manifest?.config_hash ?? null;
  const gitCommit = run.git_commit ?? manifest?.git_commit ?? null;
  const versions = manifest?.library_versions ?? {};
  const versionEntries = Object.entries(versions);
  const summary = [
    `run_id: ${run.run_id}`,
    `experiment: ${run.experiment}`,
    `seed: ${run.seed}`,
    `config_hash: ${configHash ?? ""}`,
    `git_commit: ${gitCommit ?? ""}`,
    `checkpoint_path: ${manifest?.checkpoint_path ?? ""}`,
  ].join("\n");

  return (
    <section
      data-testid="provenance-card"
      aria-label={`Provenance for ${run.run_id}`}
      className={cn(
        "rounded-control border border-border bg-bg-surface p-3",
        className,
      )}
    >
      <div className="mb-3 flex items-center justify-between gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-text-muted">Provenance</h3>
        <CopyButton value={summary} label="full provenance" />
      </div>
      <dl className="grid gap-3 sm:grid-cols-2">
        <Field label="Config hash" value={configHash} />
        <Field label="Git commit" value={gitCommit} />
        <Field label="Checkpoint path" value={manifest?.checkpoint_path} />
        <Field label="Algorithm" value={manifest?.algorithm} mono={false} />
        <Field label="Run id" value={run.run_id} />
        <Field label="Experiment / seed" value={`${run.experiment} / seed ${run.seed}`} mono={false} />
        <div className="min-w-0 sm:col-span-2">
          <dt className="text-[11px] uppercase tracking-wide text-text-muted">Library versions</dt>
          <dd className="mt-0.5 break-all font-mono text-xs text-text-primary">
            {versionEntries.length === 0
              ? "—"
              : versionEntries.map(([name, version]) => `${name} ${version}`).join(" · ")}
          </dd>
        </div>
      </dl>
    </section>
  );
}
