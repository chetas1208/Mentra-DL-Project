import test from 'node:test'
import assert from 'node:assert/strict'
import { classifySource } from '../app/utils/sourceClassification.ts'
import { parseCapabilities } from '../app/utils/modelCapabilities.ts'
import { BoundedAudioQueue } from '../app/utils/boundedAudioQueue.ts'
import { nextBackoffDelayMs, DEFAULT_BACKOFF } from '../app/utils/reconnectBackoff.ts'
import { deriveGlobalStatus, normalizeInferenceState } from '../app/utils/runtimeStatus.ts'
import { computeSignalIntegrity } from '../app/utils/signalIntegrity.ts'

// --- source classification -------------------------------------------
test('classifySource: empty/unlabeled -> UNKNOWN, never Mentra', () => {
  assert.equal(classifySource(''), 'UNKNOWN AUDIO INPUT')
  assert.equal(classifySource(null), 'UNKNOWN AUDIO INPUT')
  assert.equal(classifySource(undefined), 'UNKNOWN AUDIO INPUT')
})
test('classifySource: matches "mentra" case-insensitively', () => {
  assert.equal(classifySource('Mentra Live Microphone'), 'MENTRA LIVE')
  assert.equal(classifySource('MENTRA-BT-042'), 'MENTRA LIVE')
})
test('classifySource: ordinary device -> BROWSER MICROPHONE', () => {
  assert.equal(classifySource('MacBook Pro Microphone'), 'BROWSER MICROPHONE')
})

// --- capability parsing -------------------------------------------------
const FULL_CAPS = {
  modelId: 'speakernet-wearer-detector', modelVersion: 'v1', ready: true, task: 'wearer_activity',
  requiresEnrollment: true, supportsEnrollment: true, supportsEnrollmentFreeDetection: false,
  supportsPassivePersonalization: false, supportsOverlap: false, supportsEnvironmentActivity: true,
  supportsSourceSeparation: false, audioPolicy: 'passthrough', experimental: false,
  outputLabels: ['silence', 'wearer', 'environment'],
}
test('parseCapabilities: empty payload -> null (NOT AVAILABLE)', () => {
  assert.equal(parseCapabilities(new Uint8Array(0)), null)
})
test('parseCapabilities: malformed JSON -> null', () => {
  assert.equal(parseCapabilities(new TextEncoder().encode('{not json')), null)
})
test('parseCapabilities: missing required field -> null, never a partial guess', () => {
  const partial = { ...FULL_CAPS } as Record<string, unknown>
  delete partial.requiresEnrollment
  assert.equal(parseCapabilities(new TextEncoder().encode(JSON.stringify(partial))), null)
})
test('parseCapabilities: valid payload parses through exactly', () => {
  const parsed = parseCapabilities(new TextEncoder().encode(JSON.stringify(FULL_CAPS)))
  assert.deepEqual(parsed, FULL_CAPS)
})

// --- bounded queue / drop-oldest ----------------------------------------
function frame(seq: number, bytes = 4): { sequenceNumber: number; captureTimestampNs: bigint; pcm16: Uint8Array } {
  return { sequenceNumber: seq, captureTimestampNs: BigInt(seq), pcm16: new Uint8Array(bytes) }
}
test('BoundedAudioQueue: drops OLDEST once over capacity, keeps newest', () => {
  const q = new BoundedAudioQueue(3)
  assert.equal(q.push(frame(0)), null)
  assert.equal(q.push(frame(1)), null)
  assert.equal(q.push(frame(2)), null)
  const drop = q.push(frame(3))
  assert.ok(drop)
  assert.equal(drop!.framesDropped, 1)
  assert.equal(q.depth, 3)
  assert.equal(q.shift()!.sequenceNumber, 1) // 0 was dropped, 1 is now oldest
})
test('BoundedAudioQueue: tracks highWaterMark and cumulative drops', () => {
  const q = new BoundedAudioQueue(2)
  for (let i = 0; i < 5; i++) q.push(frame(i, 10))
  // high-water mark is measured before the over-capacity item is dropped,
  // so it reflects the true peak the queue reached, not the post-drop size
  assert.equal(q.highWaterMark, 3)
  assert.equal(q.framesDropped, 3)
  assert.equal(q.bytesDropped, 30)
})

// --- reconnect backoff ---------------------------------------------------
test('nextBackoffDelayMs: bounded, monotonically non-decreasing envelope, stops after maxAttempts', () => {
  const cfg = { baseMs: 100, maxMs: 1000, maxAttempts: 4 }
  const envelope = [1, 2, 3, 4].map((a) => nextBackoffDelayMs(a, cfg, () => 1)!) // random()=1 -> full envelope value
  assert.deepEqual(envelope, [100, 200, 400, 800])
  assert.equal(nextBackoffDelayMs(5, cfg), null)
})
test('nextBackoffDelayMs: caps at maxMs even for large attempt numbers', () => {
  const cfg = { baseMs: 100, maxMs: 1000, maxAttempts: 20 }
  const delay = nextBackoffDelayMs(20, cfg, () => 1)!
  assert.equal(delay, 1000)
})
test('nextBackoffDelayMs: jitter stays within [0, envelope]', () => {
  const cfg = DEFAULT_BACKOFF
  for (const r of [0, 0.25, 0.5, 0.75, 1]) {
    const delay = nextBackoffDelayMs(3, cfg, () => r)!
    const envelope = Math.min(cfg.maxMs, cfg.baseMs * 4)
    assert.ok(delay >= 0 && delay <= envelope)
  }
})

// --- global status derivation --------------------------------------------
const BASE_STATUS_INPUT = {
  hasSelectedSource: true, sourceLost: false, isStarting: false, captureActive: false,
  wsState: 'IDLE' as const, modelReady: null, requiresEnrollment: null, enrollmentSatisfied: false,
  audioFlowing: false, hasError: false, wasStopped: false,
}
test('deriveGlobalStatus: no source selected yet', () => {
  assert.equal(deriveGlobalStatus({ ...BASE_STATUS_INPUT, hasSelectedSource: false }), 'SELECT AUDIO SOURCE')
})
test('deriveGlobalStatus: source selected, nothing started', () => {
  assert.equal(deriveGlobalStatus(BASE_STATUS_INPUT), 'READY TO START')
})
test('deriveGlobalStatus: input lost overrides everything except explicit stop/error', () => {
  assert.equal(deriveGlobalStatus({ ...BASE_STATUS_INPUT, sourceLost: true }), 'INPUT LOST')
})
test('deriveGlobalStatus: connecting backend', () => {
  assert.equal(deriveGlobalStatus({ ...BASE_STATUS_INPUT, captureActive: true, wsState: 'CONNECTING' }), 'CONNECTING BACKEND')
})
test('deriveGlobalStatus: connected but model not ready', () => {
  assert.equal(
    deriveGlobalStatus({ ...BASE_STATUS_INPUT, captureActive: true, wsState: 'CONNECTED', modelReady: false }),
    'MODEL UNAVAILABLE',
  )
})
test('deriveGlobalStatus: connected, model ready, enrollment required and not satisfied', () => {
  assert.equal(
    deriveGlobalStatus({
      ...BASE_STATUS_INPUT, captureActive: true, wsState: 'CONNECTED', modelReady: true,
      requiresEnrollment: true, enrollmentSatisfied: false,
    }),
    'CALIBRATION REQUIRED',
  )
})
test('deriveGlobalStatus: LIVE only once audio is actually flowing, not merely connected', () => {
  const connectedNotFlowing = {
    ...BASE_STATUS_INPUT, captureActive: true, wsState: 'CONNECTED' as const, modelReady: true,
    requiresEnrollment: false, audioFlowing: false,
  }
  assert.notEqual(deriveGlobalStatus(connectedNotFlowing), 'LIVE')
  assert.equal(deriveGlobalStatus({ ...connectedNotFlowing, audioFlowing: true }), 'LIVE')
})
test('deriveGlobalStatus: explicit stop and error states take priority', () => {
  assert.equal(deriveGlobalStatus({ ...BASE_STATUS_INPUT, wasStopped: true }), 'STOPPED')
  assert.equal(deriveGlobalStatus({ ...BASE_STATUS_INPUT, hasError: true, wasStopped: true }), 'ERROR')
})

test('normalizeInferenceState: passes through known backend states, falls back to SILENCE otherwise', () => {
  assert.equal(normalizeInferenceState('WEARER'), 'WEARER')
  assert.equal(normalizeInferenceState('ENVIRONMENT'), 'ENVIRONMENT')
  assert.equal(normalizeInferenceState(null), 'SILENCE')
  assert.equal(normalizeInferenceState('garbage'), 'SILENCE')
})

// --- signal integrity (used only in debug/research telemetry) -----------
test('computeSignalIntegrity: silence has zero rms/peak/clipping', () => {
  const s = computeSignalIntegrity(new Float32Array(100))
  assert.equal(s.rms, 0)
  assert.equal(s.peak, 0)
  assert.equal(s.clippingFraction, 0)
})
test('computeSignalIntegrity: detects clipping fraction correctly', () => {
  const samples = new Float32Array([1, 1, 1, 0.1, -1, 0.2, 0.3, -0.9, 0.05, 0.05])
  const s = computeSignalIntegrity(samples)
  assert.equal(s.clippingFraction, 0.4) // 4 of 10 samples at |v| >= 1
})
test('computeSignalIntegrity: crest factor for a pure tone-like burst', () => {
  const samples = new Float32Array([0, 1, 0, -1, 0, 1, 0, -1])
  const s = computeSignalIntegrity(samples)
  assert.ok(s.crestFactor > 1) // peak exceeds rms for a non-DC signal
})
