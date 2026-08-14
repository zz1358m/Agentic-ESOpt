const TOKEN_PATTERN = /(#.*$|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:def|return|if|else|elif|for|while|in|and|or|not|True|False|None|import|from|as|try|except|raise|with|class|lambda)\b|\b\d+(?:\.\d+)?\b)/gm;

function tokenClass(token: string) {
  if (token.startsWith("#")) return "code-token--comment";
  if (token.startsWith('"') || token.startsWith("'")) return "code-token--string";
  if (/^\d/.test(token)) return "code-token--number";
  return "code-token--keyword";
}

export function HighlightedCode({ children, className = "" }: { children: string; className?: string }) {
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  for (const match of children.matchAll(TOKEN_PATTERN)) {
    const index = match.index ?? 0;
    if (index > cursor) parts.push(children.slice(cursor, index));
    parts.push(<span className={tokenClass(match[0])} key={`${index}-${match[0]}`}>{match[0]}</span>);
    cursor = index + match[0].length;
  }
  parts.push(children.slice(cursor));
  return <code className={className}>{parts}</code>;
}
