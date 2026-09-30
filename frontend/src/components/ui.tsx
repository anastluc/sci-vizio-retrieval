import type { ReactNode } from "react";

import type { DocumentStatus, FacetValue } from "../api/types";

export function UsageMeter({ label, used, limit }: { label: string; used: number; limit: number }) {
  const fraction = limit > 0 ? Math.min(1, used / limit) : 0;
  const tone = fraction >= 1 ? "full" : fraction >= 0.8 ? "high" : "ok";
  return (
    <div className="meter">
      <div className="meter-label">
        <span>{label}</span>
        <span>
          {used} / {limit}
        </span>
      </div>
      <div
        className={`meter-track meter-${tone}`}
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={limit}
        aria-valuenow={used}
      >
        <div className="meter-fill" style={{ width: `${fraction * 100}%` }} />
      </div>
    </div>
  );
}

const STATUS_LABELS: Record<DocumentStatus, string> = {
  uploading: "Uploading",
  queued: "Queued",
  processing: "Processing",
  ready: "Ready",
  failed: "Failed",
};

export function StatusBadge({ status }: { status: DocumentStatus }) {
  const busy = status === "uploading" || status === "queued" || status === "processing";
  return (
    <span className={`badge badge-${status}`}>
      {busy && <span className="dot-pulse" aria-hidden="true" />}
      {STATUS_LABELS[status]}
    </span>
  );
}

export function Banner({ tone = "info", children, onClose }: {
  tone?: "info" | "error" | "success";
  children: ReactNode;
  onClose?: () => void;
}) {
  return (
    <div className={`banner banner-${tone}`} role={tone === "error" ? "alert" : "status"}>
      <div>{children}</div>
      {onClose && (
        <button className="banner-close" onClick={onClose} aria-label="Dismiss">×</button>
      )}
    </div>
  );
}

export function FacetGroup({ title, values, selected, onToggle, emptyText = "None" }: {
  title: string;
  values: FacetValue[];
  selected: string[];
  onToggle: (value: string) => void;
  emptyText?: string;
}) {
  return (
    <fieldset className="facet">
      <legend>{title}</legend>
      {values.length === 0 && <p className="muted small">{emptyText}</p>}
      {values.map((v) => (
        <label key={v.value} className="facet-option">
          <input type="checkbox" checked={selected.includes(v.value)} onChange={() => onToggle(v.value)} />
          <span className="facet-label" title={v.label}>{v.label}</span>
          <span className="facet-count">{v.count}</span>
        </label>
      ))}
    </fieldset>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children}
    </div>
  );
}
