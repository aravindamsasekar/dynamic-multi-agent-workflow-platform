import type {
  ExecutionResponse,
  ExtensionListResponse,
  GeneratePlanResponse,
  InstalledExtensionListResponse,
} from '../types/api'

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? ''

export class ApiError extends Error {
  statusCode: number

  constructor(message: string, statusCode: number) {
    super(message)
    this.statusCode = statusCode
    this.name = 'ApiError'
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })
  } catch {
    throw new ApiError(
      'Cannot connect to API server. Is the backend running at localhost:8000?',
      0,
    )
  }
  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const body = (await response.json()) as { detail?: unknown }
      if (body.detail) detail = String(body.detail)
    } catch {
      // ignore parse errors
    }
    throw new ApiError(detail, response.status)
  }
  return response.json() as Promise<T>
}

export const api = {
  generatePlan: (goal: string) =>
    apiFetch<GeneratePlanResponse>('/planner/generate', {
      method: 'POST',
      body: JSON.stringify({ goal }),
    }),

  installAndRegenerate: (planId: string) =>
    apiFetch<GeneratePlanResponse>(`/planner/${planId}/install`, {
      method: 'POST',
    }),

  approvePlan: (planId: string, inputData: string) =>
    apiFetch<ExecutionResponse>(`/planner/${planId}/approve`, {
      method: 'POST',
      body: JSON.stringify({ input_data: inputData }),
    }),

  listExtensions: () => apiFetch<ExtensionListResponse>('/extensions'),

  listInstalledExtensions: () =>
    apiFetch<InstalledExtensionListResponse>('/extensions/installed'),
}

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message
  if (err instanceof Error) return err.message
  return 'An unexpected error occurred.'
}
