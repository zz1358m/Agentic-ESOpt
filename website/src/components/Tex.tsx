import katex from "katex";

export function Tex({ math, label, className = "" }: { math: string; label: string; className?: string }) {
  return <span className={`tex ${className}`.trim()} role="math" aria-label={label} dangerouslySetInnerHTML={{ __html: katex.renderToString(math, { throwOnError: false, output: "html" }) }}/>;
}
