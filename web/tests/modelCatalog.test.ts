import test from 'node:test'
import assert from 'node:assert/strict'
import {
  parseModelCatalog, parseSessionConfigAck, selectableModels,
  type CatalogModel,
} from '../app/utils/modelCatalog.ts'
import { deriveGlobalStatus } from '../app/utils/runtimeStatus.ts'
import type { ModelCapabilities } from '../app/utils/modelCapabilities.ts'

const SPEAKERNET: CatalogModel = {
  id: 'speakernet', displayName: 'SpeakerNet', modelVersion: 'nemo-speakerverification-1',
  ready: true, experimental: false, requiresEnrollment: true, supportsEnrollment: true,
  supportsEnrollmentFreeDetection: false, supportsEnvironmentActivity: false,
  supportsOverlap: false, supportsSourceSeparation: false, unavailableReason: null,
}
const GEOWEARNET: CatalogModel = {
  id: 'geowearnet_g2', displayName: 'GeoWearNet G2', modelVersion: 'g2-real-mmcsg-v1',
  ready: true, experimental: true, requiresEnrollment: false, supportsEnrollment: false,
  supportsEnrollmentFreeDetection: true, supportsEnvironmentActivity: true,
  supportsOverlap: true, supportsSourceSeparation: false, unavailableReason: null,
}
const CATALOG = { type: 'model_catalog', defaultModel: 'speakernet', models: [SPEAKERNET, GEOWEARNET] }

function encode(v: unknown): Uint8Array {
  return new TextEncoder().encode(JSON.stringify(v))
}

// --- catalog parsing ------------------------------------------------------
test('parseModelCatalog: empty payload -> null, never a fabricated list', () => {
  assert.equal(parseModelCatalog(new Uint8Array(0)), null)
})
test('parseModelCatalog: malformed JSON -> null', () => {
  assert.equal(parseModelCatalog(new TextEncoder().encode('{nope')), null)
})
test('parseModelCatalog: wrong message type -> null', () => {
  assert.equal(parseModelCatalog(encode({ ...CATALOG, type: 'something_else' })), null)
})
test('parseModelCatalog: an entry missing a required field invalidates the catalog', () => {
  const broken = { ...SPEAKERNET } as Record<string, unknown>
  delete broken.ready
  assert.equal(parseModelCatalog(encode({ ...CATALOG, models: [broken] })), null)
})
test('parseModelCatalog: valid catalog parses through exactly', () => {
  const parsed = parseModelCatalog(encode(CATALOG))
  assert.ok(parsed)
  assert.equal(parsed!.defaultModel, 'speakernet')
  assert.deepEqual(parsed!.models.map((m) => m.id), ['speakernet', 'geowearnet_g2'])
})
test('parseModelCatalog: GeoWearNet is experimental, enrollment-free, and never claims separation', () => {
  const parsed = parseModelCatalog(encode(CATALOG))!
  const geo = parsed.models.find((m) => m.id === 'geowearnet_g2')!
  assert.equal(geo.experimental, true)
  assert.equal(geo.requiresEnrollment, false)
  assert.equal(geo.supportsEnrollment, false)
  assert.equal(geo.supportsEnvironmentActivity, true)
  assert.equal(geo.supportsOverlap, true)
  assert.equal(geo.supportsSourceSeparation, false)
})
test('parseModelCatalog: an unavailable model keeps its backend-reported reason', () => {
  const down = { ...GEOWEARNET, ready: false, unavailableReason: 'checkpoint sha256 mismatch' }
  const parsed = parseModelCatalog(encode({ ...CATALOG, models: [SPEAKERNET, down] }))!
  assert.equal(parsed.models[1]!.ready, false)
  assert.equal(parsed.models[1]!.unavailableReason, 'checkpoint sha256 mismatch')
  assert.equal(parsed.models[0]!.ready, true) // one model down never marks the other down
})

// --- session_config_ack ---------------------------------------------------
test('parseSessionConfigAck: successful binding reports the model the backend bound', () => {
  const ack = parseSessionConfigAck(encode({
    type: 'session_config_ack', model: 'geowearnet_g2', requestedModel: 'geowearnet_g2',
    ready: true, error: null, capabilities: null,
  }))
  assert.ok(ack)
  assert.equal(ack!.model, 'geowearnet_g2')
  assert.equal(ack!.ready, true)
  assert.equal(ack!.error, null)
})
test('parseSessionConfigAck: a rejected request keeps the previously bound model', () => {
  const ack = parseSessionConfigAck(encode({
    type: 'session_config_ack', model: 'speakernet', requestedModel: 'geowearnet_g2',
    ready: true, error: 'GeoWearNet G2 unavailable: checkpoint missing', capabilities: null,
  }))!
  assert.equal(ack.model, 'speakernet')       // still SpeakerNet -- not the requested model
  assert.equal(ack.requestedModel, 'geowearnet_g2')
  assert.match(ack.error!, /unavailable/)
})
test('parseSessionConfigAck: malformed payload -> null', () => {
  assert.equal(parseSessionConfigAck(new Uint8Array(0)), null)
  assert.equal(parseSessionConfigAck(encode({ type: 'model_catalog' })), null)
})

// --- selector options -----------------------------------------------------
const CAPS: ModelCapabilities = {
  modelId: 'speakernet', modelVersion: 'nemo-speakerverification-1', ready: true,
  task: 'speaker_verification', requiresEnrollment: true, supportsEnrollment: true,
  supportsEnrollmentFreeDetection: false, supportsPassivePersonalization: false,
  supportsOverlap: false, supportsEnvironmentActivity: false, supportsSourceSeparation: false,
  audioPolicy: 'passthrough', experimental: false, outputLabels: ['wearer', 'not_wearer'],
}
test('selectableModels: the backend catalog is authoritative when present', () => {
  const models = selectableModels({ defaultModel: 'speakernet', models: [SPEAKERNET, GEOWEARNET] }, CAPS)
  assert.deepEqual(models.map((m) => m.id), ['speakernet', 'geowearnet_g2'])
})
test('selectableModels: no catalog -> only the model that receiver actually reported', () => {
  const models = selectableModels(null, CAPS)
  assert.equal(models.length, 1)
  assert.equal(models[0]!.id, 'speakernet')
  assert.equal(models[0]!.ready, true) // mirrors the payload, never asserted locally
})
test('selectableModels: nothing connected -> no options, never a hardcoded list', () => {
  assert.deepEqual(selectableModels(null, null), [])
})
test('selectableModels: a not-ready capability payload stays not ready', () => {
  const models = selectableModels(null, { ...CAPS, ready: false })
  assert.equal(models[0]!.ready, false)
})

// --- capability-driven session gating ------------------------------------
const BASE = {
  hasSelectedSource: true, sourceLost: false, isStarting: false, captureActive: true,
  wsState: 'CONNECTED' as const, modelReady: true, enrollmentSatisfied: false,
  audioFlowing: true, hasError: false, wasStopped: false,
}
test('GeoWearNet (requiresEnrollment=false) goes LIVE without any enrollment', () => {
  assert.equal(deriveGlobalStatus({ ...BASE, requiresEnrollment: false }), 'LIVE')
})
test('SpeakerNet (requiresEnrollment=true) still blocks on calibration', () => {
  assert.equal(deriveGlobalStatus({ ...BASE, requiresEnrollment: true }), 'CALIBRATION REQUIRED')
  assert.equal(
    deriveGlobalStatus({ ...BASE, requiresEnrollment: true, enrollmentSatisfied: true }),
    'LIVE',
  )
})
test('a model the backend says is not ready never reads as LIVE', () => {
  assert.equal(
    deriveGlobalStatus({ ...BASE, requiresEnrollment: false, modelReady: false }),
    'MODEL UNAVAILABLE',
  )
})
