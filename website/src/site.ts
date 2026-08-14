export function siteRoot(): string {
  const path = window.location.pathname;
  for (const marker of ["/tasks/", "/scaling/", "/paper/"]) {
    const index = path.indexOf(marker);
    if (index >= 0) return `${path.slice(0, index)}/`;
  }
  return path.endsWith("/") ? path : path.slice(0, path.lastIndexOf("/") + 1);
}

export function siteHref(path: string): string {
  return `${siteRoot()}${path.replace(/^\//, "")}`;
}

export function currentRoute(): string {
  const path = window.location.pathname.replace(/index\.html$/, "");
  const root = siteRoot();
  const relative = path.startsWith(root) ? path.slice(root.length) : path.replace(/^\//, "");
  return `/${relative}`.replace(/\/+$/, "/");
}
