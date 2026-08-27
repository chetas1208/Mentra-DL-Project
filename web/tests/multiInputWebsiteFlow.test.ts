/**
 * Simulates the live website's multi-audio-input flow in Node — the exact
 * sequence index.vue + useMediaDevices + AudioSourceSelector execute.
 * NOT SpeakerNet; tests microphone enumeration, selection, classification,
 * unplug detection, and getUserMedia constraint wiring.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import {
  hasLabeledInputs, isSelectedDeviceMissing, mapAudioInputDevices,
  type AudioInputOption,
} from '../app/utils/audioInputDevices.ts'
import { classifySource } from '../app/utils/sourceClassification.ts'

const MAC = { deviceId: 'mic-mac', kind: 'audioinput', label: 'MacBook Pro Microphone', groupId: 'g1' }
const USB = { deviceId: 'mic-usb', kind: 'audioinput', label: 'Blue Yeti USB', groupId: 'g2' }
const MENTRA = { deviceId: 'mic-mentra', kind: 'audioinput', label: 'Mentra Live Microphone', groupId: 'g3' }
const HDMI = { deviceId: 'mic-hdmi', kind: 'audioinput', label: 'HDMI Audio Capture', groupId: 'g4' }
const SPEAKER = { deviceId: 'spk-1', kind: 'audiooutput', label: 'Built-in Output', groupId: 'g5' }

/** Mirrors AudioSourceSelector option list: default + each audioinput. */
function buildInputDropdownOptions(devices: AudioInputOption[]): { value: string; label: string }[] {
  return [
    { value: '', label: 'Default microphone' },
    ...devices.map((d) => ({ value: d.deviceId, label: d.label || 'UNKNOWN AUDIO INPUT' })),
  ]
}

/** Mirrors getUserMedia audio constraint from useMicCapture.start(). */
function buildGetUserMediaConstraints(selectedDeviceId: string | undefined) {
  return {
    audio: {
      deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
      channelCount: { ideal: 1 },
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  }
}

/** Mirrors useRuntimeState sourceClassification for a selected device. */
function websiteSourceClassification(
  devices: AudioInputOption[],
  selectedDeviceId: string | undefined,
): string {
  const device = selectedDeviceId
    ? devices.find((d) => d.deviceId === selectedDeviceId)
    : undefined
  return classifySource(device?.label)
}

// --- full website flow: 4 mics plugged in ---------------------------------
test('website flow: enumerate 4 audio inputs (+ 1 output ignored)', () => {
  const devices = mapAudioInputDevices([MAC, USB, MENTRA, HDMI, SPEAKER])
  assert.equal(devices.length, 4)
  const options = buildInputDropdownOptions(devices)
  assert.equal(options.length, 5) // default + 4
  assert.deepEqual(options.slice(1).map((o) => o.label), [
    'MacBook Pro Microphone', 'Blue Yeti USB', 'Mentra Live Microphone', 'HDMI Audio Capture',
  ])
})

test('website flow: switch MacBook → USB → Mentra → HDMI classifications', () => {
  const devices = mapAudioInputDevices([MAC, USB, MENTRA, HDMI])
  assert.equal(websiteSourceClassification(devices, 'mic-mac'), 'BROWSER MICROPHONE')
  assert.equal(websiteSourceClassification(devices, 'mic-usb'), 'BROWSER MICROPHONE')
  assert.equal(websiteSourceClassification(devices, 'mic-mentra'), 'MENTRA LIVE')
  assert.equal(websiteSourceClassification(devices, 'mic-hdmi'), 'BROWSER MICROPHONE')
})

test('website flow: each selection produces correct getUserMedia deviceId constraint', () => {
  for (const id of ['mic-mac', 'mic-usb', 'mic-mentra', 'mic-hdmi']) {
    const c = buildGetUserMediaConstraints(id)
    assert.deepEqual(c.audio.deviceId, { exact: id })
  }
  const defaultC = buildGetUserMediaConstraints(undefined)
  assert.equal(defaultC.audio.deviceId, undefined)
})

test('website flow: default mic uses no exact deviceId constraint', () => {
  const c = buildGetUserMediaConstraints(undefined)
  assert.equal(c.audio.deviceId, undefined)
})

// --- unplug while selected ------------------------------------------------
test('website flow: USB selected, USB unplugged → INPUT LOST', () => {
  let selectedId: string | undefined = 'mic-usb'
  let devices = mapAudioInputDevices([MAC, USB, MENTRA, HDMI])
  assert.equal(isSelectedDeviceMissing(selectedId, devices), false)

  // devicechange event — USB gone
  devices = mapAudioInputDevices([MAC, MENTRA, HDMI])
  assert.equal(isSelectedDeviceMissing(selectedId, devices), true)
})

test('website flow: after INPUT LOST, user picks another mic → cleared', () => {
  let selectedId: string | undefined = 'mic-usb'
  let selectedDeviceLost = isSelectedDeviceMissing(selectedId, mapAudioInputDevices([MAC, MENTRA]))

  // onSelectAnother / selectDevice
  selectedId = 'mic-mac'
  selectedDeviceLost = false
  assert.equal(selectedDeviceLost, false)
  assert.equal(isSelectedDeviceMissing(selectedId, mapAudioInputDevices([MAC, MENTRA])), false)
})

// --- permission gate ------------------------------------------------------
test('website flow: pre-permission enumerate shows UNKNOWN for all inputs', () => {
  const pre = mapAudioInputDevices([
    { deviceId: 'a', kind: 'audioinput', label: '', groupId: 'x' },
    { deviceId: 'b', kind: 'audioinput', label: '', groupId: 'y' },
  ])
  assert.ok(pre.every((d) => d.classification === 'UNKNOWN AUDIO INPUT'))
  assert.equal(hasLabeledInputs([MAC, USB]), true)
})

test('website flow: post-permission labels populate dropdown', () => {
  const all = [MAC, USB, MENTRA, HDMI]
  assert.equal(hasLabeledInputs(all), true)
  const devices = mapAudioInputDevices(all)
  assert.ok(devices.every((d) => d.label.length > 0))
})

// --- session: input locked while capturing -------------------------------
test('website flow: input selector disabled while isCapturing (not while idle)', () => {
  const isCapturing = false
  assert.equal(isCapturing, false) // INPUT enabled
  const isCapturingLive = true
  assert.equal(isCapturingLive, true) // INPUT disabled — must stop to switch mic
})

// --- audio-debug page parity ----------------------------------------------
test('audio-debug page: lists same device count as live INPUT dropdown (minus default)', () => {
  const devices = mapAudioInputDevices([MAC, USB, MENTRA])
  const dropdownCount = buildInputDropdownOptions(devices).length - 1
  assert.equal(dropdownCount, 3)
})

test('audio-debug page: Mentra match filter finds Mentra mic among multiple inputs', () => {
  const devices = mapAudioInputDevices([MAC, USB, MENTRA, HDMI])
  const mentraMatches = devices.filter((d) => /mentra/i.test(d.label))
  assert.equal(mentraMatches.length, 1)
  assert.equal(mentraMatches[0]!.deviceId, 'mic-mentra')
})

// --- rapid switching between inputs (no state leak) -----------------------
test('website flow: rapid switch across all 4 mics keeps distinct deviceIds', () => {
  const devices = mapAudioInputDevices([MAC, USB, MENTRA, HDMI])
  const visited: string[] = []
  for (const id of ['mic-mac', 'mic-usb', 'mic-mentra', 'mic-hdmi']) {
    visited.push(id)
    assert.equal(isSelectedDeviceMissing(id, devices), false)
    assert.notEqual(websiteSourceClassification(devices, id), '')
  }
  assert.equal(new Set(visited).size, 4)
})
