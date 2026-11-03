// Background jobs, polled while the GUI is open: fast while something runs,
// slowly otherwise. Started once by the layout.

import { fetchJobs } from "./daemon";
import type { Job } from "./types";

class Jobs {
  list = $state<Job[]>([]);
  get running() {
    return this.list.filter((j) => j.state === "running");
  }
  byId(id: number | null): Job | undefined {
    return id == null ? undefined : this.list.find((j) => j.id === id);
  }
}

export const jobs = new Jobs();

export async function refreshJobs() {
  try {
    jobs.list = await fetchJobs();
  } catch {
    // offline: the header shows it; keep the last list
  }
}

export function startJobPolling(): () => void {
  let timer: ReturnType<typeof setTimeout>;
  let stopped = false;
  const tick = async () => {
    await refreshJobs();
    if (!stopped) timer = setTimeout(tick, jobs.running.length ? 1000 : 4000);
  };
  tick();
  return () => {
    stopped = true;
    clearTimeout(timer);
  };
}
