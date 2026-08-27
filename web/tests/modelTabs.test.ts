import test from 'node:test'
import assert from 'node:assert/strict'
import {
  hasLabeledInputs, isSelectedDeviceMissing, mapAudioInputDevices,
} from '../app/utils/audioInputDevices.ts'
import { selectableModels, type CatalogModel } from '../app/utils/modelCatalog.ts'
import type { ModelCapabilities } from '../app/utils/modelCapabilities.ts'

// --- multiple audio inputs (microphones, not SpeakerNet) ------------------
const MACBOOK = { deviceId: 'mac-default', kind: 'audioinput', label: 'MacBook Pro Microphone', groupId: 'g1' }
const USB_MIC = { deviceId: 'usb-001', kind: 'audioinput', label: 'Blue Yeti USB', groupId: 'g2' }
const MENTRA = { deviceId: 'mentra-bt', kind: 'audioinput', label: 'Mentra Live Microphone', groupId: 'g3' }
const UNLABELED = { deviceId: 'anon-abc', kind: 'audioinput', label: '', groupId: 'g4' }
const SPEAKER_OUT = { deviceId: 'spk-1', kind: 'audiooutput', label: 'Built-in Output', groupId: 'g5' }

test('mapAudioInputDevices: lists every audioinput, ignores outputs', () => {
  const mapped = mapAudioInputDevices([MACBOOK, USB_MIC, MENTRA, SPEAKER_OUT])
  assert.equal(mapped.length, 3)
  assert.deepEqual(mapped.map((d) => d.deviceId), ['mac-default', 'usb-001', 'mentra-bt'])
})

test('mapAudioInputDevices: classifies each input independently', () => {
  const mapped = mapAudioInputDevices([MACBOOK, USB_MIC, MENTRA, UNLABELED])
  assert.equal(mapped[0]!.classification, 'BROWSER MICROPHONE')
  assert.equal(mapped[1]!.classification, 'BROWSER MICROPHONE')
  assert.equal(mapped[2]!.classification, 'MENTRA LIVE')
  assert.equal(mapped[3]!.classification, 'UNKNOWN AUDIO INPUT')
})

test('mapAudioInputDevices: three distinct mics stay distinct by deviceId', () => {
  const mapped = mapAudioInputDevices([MACBOOK, USB_MIC, MENTRA])
  const ids = new Set(mapped.map((d) => d.deviceId))
  assert.equal(ids.size, 3)
})

test('hasLabeledInputs: false until permission populates labels', () => {
  assert.equal(hasLabeledInputs([UNLABELED]), false)
  assert.equal(hasLabeledInputs([UNLABELED, SPEAKER_OUT]), false)
})

test('hasLabeledInputs: true once any audioinput has a label', () => {
  assert.equal(hasLabeledInputs([UNLABELED, MACBOOK]), true)
})

test('isSelectedDeviceMissing: default mic (undefined id) never counts as lost', () => {
  const devices = mapAudioInputDevices([MACBOOK, USB_MIC])
  assert.equal(isSelectedDeviceMissing(undefined, devices), false)
})

test('isSelectedDeviceMissing: selected USB mic still present after refresh', () => {
  const devices = mapAudioInputDevices([MACBOOK, USB_MIC, MENTRA])
  assert.equal(isSelectedDeviceMissing('usb-001', devices), false)
})

test('isSelectedDeviceMissing: unplugged mic triggers input-lost', () => {
  const before = mapAudioInputDevices([MACBOOK, USB_MIC, MENTRA])
  assert.equal(isSelectedDeviceMissing('usb-001', before), false)
  const after = mapAudioInputDevices([MACBOOK, MENTRA]) // USB mic unplugged
  assert.equal(isSelectedDeviceMissing('usb-001', after), true)
})

test('switching between two browser mics changes classification but not Mentra claim', () => {
  const devices = mapAudioInputDevices([MACBOOK, MENTRA])
  const mac = devices.find((d) => d.deviceId === 'mac-default')!
  const mentra = devices.find((d) => d.deviceId === 'mentra-bt')!
  assert.equal(mac.classification, 'BROWSER MICROPHONE')
  assert.equal(mentra.classification, 'MENTRA LIVE')
})

// --- model tabs catalog (SpeakerNet | GeoWearNet G2) --------------------
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
const CAPS: ModelCapabilities = {
  modelId: 'speakernet', modelVersion: 'nemo-speakerverification-1', ready: true,
  task: 'speaker_verification', requiresEnrollment: true, supportsEnrollment: true,
  supportsEnrollmentFreeDetection: false, supportsPassivePersonalization: false,
  supportsOverlap: false, supportsEnvironmentActivity: false, supportsSourceSeparation: false,
  audioPolicy: 'passthrough', experimental: false, outputLabels: ['wearer', 'not_wearer'],
}

test('model tabs: backend catalog yields exactly two visible options', () => {
  const tabs = selectableModels(
    { defaultModel: 'speakernet', models: [SPEAKERNET, GEOWEARNET] },
    CAPS,
  )
  assert.deepEqual(tabs.map((m) => m.displayName), ['SpeakerNet', 'GeoWearNet G2'])
})

test('model tabs: SpeakerNet is default when catalog says so', () => {
  const tabs = selectableModels(
    { defaultModel: 'speakernet', models: [SPEAKERNET, GEOWEARNET] },
    CAPS,
  )
  assert.equal(tabs[0]!.id, 'speakernet')
  assert.equal(tabs[0]!.requiresEnrollment, true)
})

test('model tabs: GeoWearNet is experimental and enrollment-free', () => {
  const tabs = selectableModels(
    { defaultModel: 'speakernet', models: [SPEAKERNET, GEOWEARNET] },
    CAPS,
  )
  const geo = tabs.find((m) => m.id === 'geowearnet_g2')!
  assert.equal(geo.experimental, true)
  assert.equal(geo.requiresEnrollment, false)
})

test('model tabs: unavailable GeoWearNet stays in list but not ready', () => {
  const down = { ...GEOWEARNET, ready: false, unavailableReason: 'checkpoint missing' }
  const tabs = selectableModels(
    { defaultModel: 'speakernet', models: [SPEAKERNET, down] },
    CAPS,
  )
  assert.equal(tabs.length, 2)
  assert.equal(tabs[1]!.ready, false)
  assert.equal(tabs[0]!.ready, true)
})
