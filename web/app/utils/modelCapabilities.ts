// Backend model-capabilities contract (spec section: MODEL CAPABILITY
// CONTRACT). The backend is authoritative -- this file only types and
// parses what the server actually sent inside the STREAM_ACCEPTED
// payload; it does not hardcode any model's capabilities on the frontend.
// If the server sends nothing (empty payload -- an older receiver build,
// or an accept before this field existed), parseCapabilities returns
// null and every caller must render that as NOT AVAILABLE, never as a
// guessed default.

export interface ModelCapabilities {
  modelId: string
  modelVersion: string
  ready: boolean
  task: string
  requiresEnrollment: boolean
  supportsEnrollment: boolean
  supportsEnrollmentFreeDetection: boolean
  supportsPassivePersonalization: boolean
  supportsOverlap: boolean
  supportsEnvironmentActivity: boolean
  supportsSourceSeparation: boolean
  outputLabels: string[]
}

function isModelCapabilities(v: unknown): v is ModelCapabilities {
  if (!v || typeof v !== 'object') return false
  const o = v as Record<string, unknown>
  return (
    typeof o.modelId === 'string' &&
    typeof o.modelVersion === 'string' &&
    typeof o.ready === 'boolean' &&
    typeof o.task === 'string' &&
    typeof o.requiresEnrollment === 'boolean' &&
    typeof o.supportsEnrollment === 'boolean' &&
    typeof o.supportsEnrollmentFreeDetection === 'boolean' &&
    typeof o.supportsPassivePersonalization === 'boolean' &&
    typeof o.supportsOverlap === 'boolean' &&
    typeof o.supportsEnvironmentActivity === 'boolean' &&
    typeof o.supportsSourceSeparation === 'boolean' &&
    Array.isArray(o.outputLabels)
  )
}

/** `payload` is the raw bytes of a STREAM_ACCEPTED frame. Returns null
 * (never a fabricated default) if empty, malformed, or missing required
 * fields -- callers must treat null as "capabilities NOT AVAILABLE". */
export function parseCapabilities(payload: Uint8Array): ModelCapabilities | null {
  if (payload.length === 0) return null
  let parsed: unknown
  try {
    parsed = JSON.parse(new TextDecoder().decode(payload))
  } catch {
    return null
  }
  return isModelCapabilities(parsed) ? parsed : null
}
