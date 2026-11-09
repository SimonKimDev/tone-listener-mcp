import type { AnalysisSourceInput } from "./analysisPlan";
import type { components } from "./api/schema";

type RequiredFields<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>;
export type ChainNode = RequiredFields<
  components["schemas"]["ChainNode"],
  "enabled" | "params" | "asset" | "note"
>;
type ChainDraft = Omit<components["schemas"]["ChainDraft"], "revision" | "nodes"> & {
  revision: number;
  nodes: ChainNode[];
};
type ToneIntent = RequiredFields<
  components["schemas"]["ToneIntent"],
  "observations" | "uncertainties" | "gear_notes"
>;
export type ToneProject = Omit<
  components["schemas"]["ToneProject"],
  "id" | "intent" | "chain"
> & {
  id: string;
  selected_analysis_index: number;
  intent: ToneIntent;
  chain: ChainDraft;
};
export type Interval = components["schemas"]["Interval"];
export type AdapterCatalog = components["schemas"]["AdapterCatalog"];
export type JobStatus = components["schemas"]["JobStatus"];
export type PreviewSnapshot = RequiredFields<components["schemas"]["PreviewSnapshot"], "id">;
export type ToneProfileResult = components["schemas"]["ListeningResult"];

export type Health = components["schemas"]["HealthResponse"];

export type ProjectSummary = components["schemas"]["ProjectSummary"];

export type UploadResult = components["schemas"]["UploadResponse"];

export type AnalysisPreflight = components["schemas"]["AnalysisPreflight"];

export type DiSample = "riff-1" | "riff-2" | "lead-1" | "lead-2";
export const DI_SAMPLES: readonly DiSample[] = ["riff-1", "riff-2", "lead-1", "lead-2"];
export const DI_SAMPLE_LABELS: Record<DiSample, string> = {
  "riff-1": "Riff 1 · 16.8 sec · chordal rhythm · CC0 direct DI",
  "riff-2": "Riff 2 · 12.3 sec · low-register funk chords · CC0 direct DI",
  "lead-1": "Lead 1 · 15.2 sec · upper-register lead · Guitar-TECHS, CC BY 4.0",
  "lead-2": "Lead 2 · 12.4 sec · single-note pedal riff · Guitar-TECHS, CC BY 4.0",
};

let requestToken = "";
let sessionPromise: Promise<void> | undefined;

async function establishSession(): Promise<void> {
  if (requestToken) return;
  if (!sessionPromise) {
    sessionPromise = fetch("/api/session", { credentials: "same-origin" })
      .then(async (response) => {
        if (!response.ok) throw new Error("Could not establish a local workbench session.");
        const result = (await response.json()) as { token: string };
        requestToken = result.token;
      })
      .finally(() => {
        sessionPromise = undefined;
      });
  }
  await sessionPromise;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  await establishSession();
  const headers = new Headers(init.headers);
  headers.set("X-Workbench-Token", requestToken);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, {
    ...init,
    headers,
    credentials: "same-origin",
  });
  if (!response.ok) {
    const text = await response.text();
    let detail = text;
    try {
      const body = JSON.parse(text) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // The response body is plain text.
    }
    throw new Error(detail || `Local request failed (${response.status}).`);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function requestBlob(path: string, init: RequestInit = {}): Promise<Blob> {
  await establishSession();
  const headers = new Headers(init.headers);
  headers.set("X-Workbench-Token", requestToken);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, {
    ...init,
    headers,
    credentials: "same-origin",
  });
  if (!response.ok) {
    const text = await response.text();
    let detail = text;
    try {
      const body = JSON.parse(text) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // The response body is plain text.
    }
    throw new Error(detail || `Local request failed (${response.status}).`);
  }
  return response.blob();
}

const json = (value: unknown): RequestInit => ({
  method: "POST",
  body: JSON.stringify(value),
});

export const api = {
  health: () => request<Health>("/api/health"),
  catalog: () => request<AdapterCatalog>("/api/catalog"),
  projects: () => request<ProjectSummary[]>("/api/projects"),
  project: (id: string) => request<ToneProject>(`/api/projects/${id}`),
  upload: async (file: File) => {
    const body = new FormData();
    body.set("file", file);
    return request<UploadResult>("/api/uploads", { method: "POST", body });
  },
  discardUpload: (uploadId: string) =>
    request<void>(`/api/uploads/${uploadId}`, { method: "DELETE" }),
  preflightAnalysis: (body: AnalysisSourceInput) =>
    request<AnalysisPreflight>("/api/analysis/preflight", json(body)),
  analyze: (body: AnalysisSourceInput & {
    external_consent: true;
    max_new_requests: number;
    name?: string;
  }) => request<JobStatus>("/api/jobs/analysis", json(body)),
  updateChain: (projectId: string, body: {
    base_revision: number;
    name?: string;
    nodes: ChainNode[];
  }) =>
    request<ToneProject>(`/api/projects/${projectId}/chain`, {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  selectProfile: (projectId: string, baseRevision: number, analysisIndex: number) =>
    request<ToneProject>(`/api/projects/${projectId}/profile`, json({
      base_revision: baseRevision, analysis_index: analysisIndex,
    })),
  preview: (projectId: string, body: {
    project_revision: number;
    chain_revision: number;
    sample_name: DiSample;
  }) => request<JobStatus>(`/api/projects/${projectId}/jobs/preview`, json(body)),
  job: (jobId: string) => request<JobStatus>(`/api/jobs/${jobId}`),
  cancelJob: (jobId: string) =>
    request<JobStatus>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
  previews: (projectId: string) =>
    request<PreviewSnapshot[]>(`/api/projects/${projectId}/previews`),
  audio: (projectId: string, previewId: string) =>
    requestBlob(`/api/projects/${projectId}/previews/${previewId}/audio`),
  sample: (name: DiSample) => requestBlob(`/api/samples/${name}`),
  exportPreset: (projectId: string, previewId: string) =>
    requestBlob(
      `/api/projects/${projectId}/export`,
      json({ preview_id: previewId }),
    ),
};
