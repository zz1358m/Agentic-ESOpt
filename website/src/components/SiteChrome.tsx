import type { ReactNode } from "react";
import { paper } from "../paper";
import { currentRoute, siteHref } from "../site";

const taskLinks = [
  ["Sudoku", "tasks/sudoku/"],
  ["Math", "tasks/math/"],
  ["DocVQA", "tasks/docvqa/"],
  ["WebArena", "tasks/webarena/"],
  ["AHD", "tasks/ahd/"],
];

export function SiteHeader() {
  const route = currentRoute();
  return (
    <header className="site-header">
      <a className="wordmark" href={siteHref("")} aria-label="Agentic ESOpt home">Agentic ESOpt</a>
      <nav aria-label="Primary navigation">
        <div className="nav-tasks">
          {taskLinks.map(([label, path]) => <a key={path} className={route.includes(path) ? "active" : ""} href={siteHref(path)}>{label}</a>)}
        </div>
        <details className="mobile-task-menu">
          <summary>More</summary>
          <div>{taskLinks.map(([label, path]) => <a key={path} className={route.includes(path) ? "active" : ""} href={siteHref(path)}>{label}</a>)}<a href={siteHref("paper/")}>Paper</a></div>
        </details>
        <div className="nav-resources" role="group" aria-label="Research links">
          <a className="nav-resource nav-resource--huggingface" href={paper.checkpointCollectionUrl} target="_blank" rel="noreferrer">Hugging Face ↗</a>
          <a className="nav-resource nav-resource--code" href={paper.githubUrl} target="_blank" rel="noreferrer">Code ↗</a>
        </div>
        <a className="nav-paper" href={siteHref("paper/")}>Paper ↗</a>
      </nav>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div><span className="eyebrow">Agentic ESOpt</span><p>Fine-tuning long-horizon LLM agents with trajectory-level black-box feedback.</p></div>
      <div className="footer-links"><a href={siteHref("paper/")}>Paper</a><a href={paper.githubUrl}>Code</a><a className="footer-resource" href={paper.checkpointCollectionUrl}>Hugging Face</a><a className="footer-resource" href={paper.dailyPapersUrl}>Browse Daily Papers</a></div>
    </footer>
  );
}

export function PageFrame({ children }: { children: ReactNode }) {
  return <><SiteHeader /><main>{children}</main><SiteFooter /></>;
}

export function LoadingState({ error }: { error?: string }) {
  return <PageFrame><section className="loading-state"><span className="loading-orbit" />{error || "Loading the selected original results…"}</section></PageFrame>;
}
