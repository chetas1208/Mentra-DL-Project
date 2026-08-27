// Backend model-catalog contract (MODEL_CATALOG / SESSION_CONFIG_ACK).
//
// The backend is authoritative about which models exist and which are ready.
// Nothing here hardcodes a model, a display name or a readiness value: a
// receiver that never sends a catalog (an older build, or one started before
// this message type existed) simply yields null, and the UI then offers only
// the single model the STREAM_ACCEPTED capabilities already describe -- which
// is the honest answer, because that receiver genuinely can serve only one.
//
// Mirrors server/models/registry.py's `LoadedModel.catalog_entry()` and
// `LiveSession.select_model()` exactly; the field names are the existing
// capability-schema names, not a second parallel vocabulary.

import type { ModelCapabilities } from '~/utils/modelCapabilities'

export interface CatalogModel {
  id: string
  displayName: string
  modelVersion: string
  ready: boolean
  experimental: boolean
  requiresEnrollment: boolean
  supportsEnrollment: boolean
  supportsEnrollmentFreeDetection: boolean
  supportsEnvironmentActivity: boolean
  supportsOverlap: boolean
  supportsSourceSeparation: boolean
  /** Why a model is not ready. Null when it is. Shown in the debug drawer and
   * in the existing error surface, never invented client-side. */
  unavailableReason: string | null
  /** Debug-drawer provenance only. Absent from an older receiver. */
  checkpointSha256Prefix?: string | null
}

export interface ModelCatalog {
  defaultModel: string
  models: CatalogModel[]
}

export interface SessionConfigAck {
  /** The model the session is ACTUALLY bound to after this ack. */
  model: string
  /** What the client asked for; differs from `model` when the request failed. */
  requestedModel: string | null
  ready: boolean
  error: string | null
  capabilities: ModelCapabilities | null
}

function isCatalogModel(v: unknown): v is CatalogModel {
  if (!v || typeof v !== 'object') return false
  const o = v as Record<string, unknown>
  return (
    typeof o.id === 'string' &&
    typeof o.displayName === 'string' &&
    typeof o.modelVersion === 'string' &&
    typeof o.ready === 'boolean' &&
    typeof o.experimental === 'boolean' &&
    typeof o.requiresEnrollment === 'boolean' &&
    typeof o.supportsEnrollment === 'boolean' &&
    typeof o.supportsEnrollmentFreeDetection === 'boolean' &&
    typeof o.supportsEnvironmentActivity === 'boolean' &&
    typeof o.supportsOverlap === 'boolean' &&
    typeof o.supportsSourceSeparation === 'boolean' &&
    (o.unavailableReason === null || typeof o.unavailableReason === 'string')
  )
}

/** `payload` is the raw bytes of a MODEL_CATALOG frame. Returns null (never a
 * fabricated list) if empty, malformed, or missing required fields. */
export function parseModelCatalog(payload: Uint8Array): ModelCatalog | null {
  if (payload.length === 0) return null
  let parsed: unknown
  try {
    parsed = JSON.parse(new TextDecoder().decode(payload))
  } catch {
    return null
  }
  if (!parsed || typeof parsed !== 'object') return null
  const o = parsed as Record<string, unknown>
  if (o.type !== 'model_catalog') return null
  if (typeof o.defaultModel !== 'string' || !Array.isArray(o.models)) return null
  if (!o.models.every(isCatalogModel)) return null
  return { defaultModel: o.defaultModel, models: o.models as CatalogModel[] }
}

/** `payload` is the raw bytes of a SESSION_CONFIG_ACK frame. */
export function parseSessionConfigAck(payload: Uint8Array): SessionConfigAck | null {
  if (payload.length === 0) return null
  let parsed: unknown
  try {
    parsed = JSON.parse(new TextDecoder().decode(payload))
  } catch {
    return null
  }
  if (!parsed || typeof parsed !== 'object') return null
  const o = parsed as Record<string, unknown>
  if (o.type !== 'session_config_ack') return null
  if (typeof o.model !== 'string' || typeof o.ready !== 'boolean') return null
  return {
    model: o.model,
    requestedModel: typeof o.requestedModel === 'string' ? o.requestedModel : null,
    ready: o.ready,
    error: typeof o.error === 'string' ? o.error : null,
    capabilities: (o.capabilities ?? null) as ModelCapabilities | null,
  }
}

/** The selector's options. With no catalog there is exactly one option -- the
 * model the connected receiver already reported through STREAM_ACCEPTED -- so
 * the control still tells the truth about a single-model backend instead of
 * listing models that backend cannot serve. */
export function selectableModels(
  catalog: ModelCatalog | null,
  capabilities: ModelCapabilities | null,
): CatalogModel[] {
  if (catalog) return catalog.models
  if (!capabilities) return []
  return [{
    id: capabilities.modelId,
    displayName: capabilities.modelId,
    modelVersion: capabilities.modelVersion,
    ready: capabilities.ready,
    experimental: capabilities.experimental,
    requiresEnrollment: capabilities.requiresEnrollment,
    supportsEnrollment: capabilities.supportsEnrollment,
    supportsEnrollmentFreeDetection: capabilities.supportsEnrollmentFreeDetection,
    supportsEnvironmentActivity: capabilities.supportsEnvironmentActivity,
    supportsOverlap: capabilities.supportsOverlap,
    supportsSourceSeparation: capabilities.supportsSourceSeparation,
    unavailableReason: null,
    checkpointSha256Prefix: null,
  }]
}
