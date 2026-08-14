import type { ReactNode } from "react";
import type { TaskPayload } from "../types";

export function TaskHero({ kicker, title, accent, summary, metric, value, detail }: { kicker: string; title: string; accent: string; summary: string; metric: string; value: string; detail: string }) {
  return (
    <section className="task-hero section-shell">
      <div><span className="eyebrow">{kicker}</span><h1 className="display-title"><span className="display-title__line">{title}</span><em>{accent}</em></h1><p className="lead">{summary}</p></div>
      <div className="hero-metric"><span>{metric}</span><strong>{value}</strong><small>{detail}</small></div>
    </section>
  );
}

export function SegmentedControl<T extends string | number>({ label, options, value, onChange, format = String }: { label: string; options: T[]; value: T; onChange: (value: T) => void; format?: (value: T) => string }) {
  return <fieldset className="segmented"><legend>{label}</legend><div>{options.map((option) => <button type="button" aria-pressed={value === option} className={value === option ? "active" : ""} key={option} onClick={() => onChange(option)}>{format(option)}</button>)}</div></fieldset>;
}

export function DataProvenance({ metadata }: { metadata: TaskPayload["metadata"] }) {
  return (
    <aside className="provenance">
      <div><span className="status-dot" />Original result</div>
      <p>{metadata.note}</p>
      <details><summary>Sources used</summary><ul>{metadata.sourceFiles.map((source) => <li key={source}>{source}</li>)}</ul></details>
    </aside>
  );
}

export function ResultTable({ rows, columns, showBaselines = true }: { rows: Record<string, string | number>[]; columns: { key: string; label: string; format?: (value: string | number) => string }[]; showBaselines?: boolean }) {
  const visible = showBaselines ? rows : rows.filter((row) => String(row.method).includes("Agentic ESOpt"));
  return (
    <div className="table-wrap"><table className="results-table"><thead><tr>{columns.map((column) => <th key={column.key}>{column.label}</th>)}</tr></thead><tbody>{visible.map((row, index) => <tr className={String(row.method).includes("Agentic ESOpt") ? "highlight-row" : ""} key={`${row.method}-${index}`}>{columns.map((column) => <td key={column.key}>{column.format ? column.format(row[column.key]) : row[column.key]}</td>)}</tr>)}</tbody></table></div>
  );
}

export function SectionHeading({ eyebrow, title, children, aside }: { eyebrow: string; title: string; children?: ReactNode; aside?: ReactNode }) {
  return <div className="section-heading"><div><span className="eyebrow">{eyebrow}</span><h2>{title}</h2>{children && <p>{children}</p>}</div>{aside}</div>;
}

export function BaselineToggle({ checked, onChange }: { checked: boolean; onChange: (value: boolean) => void }) {
  return <label className="switch"><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><span /><b>Show baselines</b></label>;
}
