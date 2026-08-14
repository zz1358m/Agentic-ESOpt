import { useEffect, useState } from "react";
import { siteHref } from "../site";
import type { TaskPayload } from "../types";

export function useTaskData(name: string) {
  const [data, setData] = useState<TaskPayload | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let live = true;
    fetch(siteHref(`data/${name}.json`))
      .then((response) => {
        if (!response.ok) throw new Error(`Could not load ${name} data.`);
        return response.json() as Promise<TaskPayload>;
      })
      .then((payload) => { if (live) setData(payload); })
      .catch((reason: unknown) => { if (live) setError(reason instanceof Error ? reason.message : "Data unavailable."); });
    return () => { live = false; };
  }, [name]);

  return { data, error };
}
