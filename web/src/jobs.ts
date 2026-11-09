import { api, type JobStatus } from "./api";

export const terminalJobStates = new Set([
  "completed", "partial", "failed", "cancelled", "stale",
]);

const delay = (milliseconds: number) =>
  new Promise((resolve) => window.setTimeout(resolve, milliseconds));

export async function waitForJob(
  initial: JobStatus,
  onUpdate: (job: JobStatus) => void,
): Promise<JobStatus> {
  let current = initial;
  onUpdate(current);
  while (!terminalJobStates.has(current.status)) {
    await delay(500);
    current = await api.job(current.id);
    onUpdate(current);
  }
  return current;
}
