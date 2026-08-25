// Single documented source for the GLOBAL STATUS state machine (spec
// section 28) and for wearer/environment state semantics (section 33).
// Keeping this derivation in one place means no component invents its
// own "is it live" logic that could disagree with another component.

export type GlobalStatus =
  | 'SELECT AUDIO SOURCE'
  | 'READY TO START'
  | 'INITIALIZING AUDIO'
  | 'CONNECTING BACKEND'
  | 'CALIBRATION REQUIRED'
  | 'LIVE'
  | 'INPUT LOST'
  | 'BACKEND UNAVAILABLE'
  | 'MODEL UNAVAILABLE'
  | 'STOPPED'
  | 'ERROR'

export interface RuntimeStatusInputs {
  hasSelectedSource: boolean
  sourceLost: boolean
  isStarting: boolean
  captureActive: boolean
  wsState: 'IDLE' | 'CONNECTING' | 'CONNECTED' | 'RECONNECTING' | 'ERROR' | 'CLOSING' | 'CLOSED'
  modelReady: boolean | null // null = not yet known (no handshake received)
  requiresEnrollment: boolean | null // null = capability not yet known
  enrollmentSatisfied: boolean
  audioFlowing: boolean // frames have actually been sent since session start
  hasError: boolean
  wasStopped: boolean
}

/** Pure function: given the current state of every subsystem, returns the
 * ONE status string the UI should show. Never infer LIVE from the socket
 * being open alone (section 24/27) -- LIVE requires audio to actually be
 * flowing end to end. */
export function deriveGlobalStatus(s: RuntimeStatusInputs): GlobalStatus {
  if (s.hasError) return 'ERROR'
  if (s.wasStopped) return 'STOPPED'
  if (s.sourceLost) return 'INPUT LOST'
  if (!s.hasSelectedSource) return 'SELECT AUDIO SOURCE'
  if (s.wsState === 'ERROR') return 'BACKEND UNAVAILABLE'
  if (s.isStarting && !s.captureActive) return 'INITIALIZING AUDIO'
  if (s.wsState === 'CONNECTING' || s.wsState === 'RECONNECTING') return 'CONNECTING BACKEND'
  if (s.wsState === 'CONNECTED' && s.modelReady === false) return 'MODEL UNAVAILABLE'
  if (s.wsState === 'CONNECTED' && s.requiresEnrollment === true && !s.enrollmentSatisfied) {
    return 'CALIBRATION REQUIRED'
  }
  if (s.wsState === 'CONNECTED' && s.audioFlowing) return 'LIVE'
  if (s.wsState === 'CONNECTED' && s.captureActive) return 'INITIALIZING AUDIO'
  return 'READY TO START'
}

// Wearer/environment state semantics (section 33): the STATE label shown
// in the inference panel comes from the backend's `state` field
// (mentra/audio/consumer.py's hysteresis-banded classification) whenever
// it is present -- this is backend-authoritative, not a frontend
// `probability > threshold` guess. SILENCE/UNCERTAIN below are frontend
// fallbacks ONLY for the narrow window before the very first backend
// result arrives; they are never used once real results are flowing.
export type InferenceState = 'SILENCE' | 'WEARER' | 'ENVIRONMENT' | 'OVERLAP' | 'UNCERTAIN'

const KNOWN_STATES: readonly InferenceState[] = ['SILENCE', 'WEARER', 'ENVIRONMENT', 'OVERLAP', 'UNCERTAIN']

export function normalizeInferenceState(backendState: string | null | undefined): InferenceState {
  if (backendState && (KNOWN_STATES as readonly string[]).includes(backendState)) {
    return backendState as InferenceState
  }
  return 'SILENCE' // no result yet -- not a guess, an explicit "nothing received"
}
