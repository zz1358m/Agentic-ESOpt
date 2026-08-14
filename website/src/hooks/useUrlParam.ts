import { useEffect, useState } from "react";

export function useUrlParam<T>(
  key: string,
  fallback: T,
  parse: (raw: string, fallback: T) => T,
  serialize: (value: T) => string,
) {
  const [value, setValue] = useState(() => {
    const raw = new URLSearchParams(window.location.search).get(key);
    return raw === null ? fallback : parse(raw, fallback);
  });
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set(key, serialize(value));
    window.history.replaceState(null, "", `${window.location.pathname}?${params.toString()}`);
  }, [key, serialize, value]);
  return [value, setValue] as const;
}
