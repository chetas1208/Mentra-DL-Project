import test from 'node:test'
import assert from 'node:assert/strict'
import {
  hasLabeledInputs, isSelectedDeviceMissing, mapAudioInputDevices,
} from '../app/utils/audioInputDevices.ts'

const MAC = { deviceId: 'mac', kind: 'audioinput', label: 'MacBook Pro Microphone', groupId: 'a' }
const USB = { deviceId: 'usb', kind: 'audioinput', label: 'USB PnP Sound Device', groupId: 'b' }
const HDMI = { deviceId: 'hdmi', kind: 'audioinput', label: 'HDMI Audio Input', groupId: 'c' }

test('multiple audio inputs: enumerate three mics', () => {
  const inputs = mapAudioInputDevices([MAC, USB, HDMI])
  assert.equal(inputs.length, 3)
})

test('multiple audio inputs: each maps to BROWSER MICROPHONE', () => {
  const inputs = mapAudioInputDevices([MAC, USB, HDMI])
  assert.ok(inputs.every((d) => d.classification === 'BROWSER MICROPHONE'))
})

test('multiple audio inputs: selecting second device does not affect first entry', () => {
  const list1 = mapAudioInputDevices([MAC, USB])
  const list2 = mapAudioInputDevices([MAC, USB, HDMI])
  assert.equal(list1[0]!.deviceId, list2[0]!.deviceId)
  assert.equal(list2.length, 3)
})

test('multiple audio inputs: device refresh after unplug', () => {
  const full = mapAudioInputDevices([MAC, USB, HDMI])
  assert.equal(isSelectedDeviceMissing('usb', full), false)
  const partial = mapAudioInputDevices([MAC, HDMI])
  assert.equal(isSelectedDeviceMissing('usb', partial), true)
})

test('multiple audio inputs: permission gate — unlabeled until granted', () => {
  const pre = [{ deviceId: 'x', kind: 'audioinput', label: '', groupId: 'z' }]
  assert.equal(hasLabeledInputs(pre), false)
  const post = mapAudioInputDevices([{ deviceId: 'x', kind: 'audioinput', label: 'Real Mic', groupId: 'z' }])
  assert.equal(post[0]!.classification, 'BROWSER MICROPHONE')
})
