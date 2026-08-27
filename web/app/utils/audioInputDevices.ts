// Pure helpers for audio-input enumeration — independently testable so
// multi-device switching behaviour can be verified without a real browser.

import { classifySource, type SourceClassification } from './sourceClassification'

export interface RawMediaDevice {
  deviceId: string
  kind: string
  label: string
  groupId: string
}

export interface AudioInputOption {
  deviceId: string
  label: string
  groupId: string
  classification: SourceClassification
}

/** Map browser enumerateDevices() audioinput entries to labeled options. */
export function mapAudioInputDevices(all: RawMediaDevice[]): AudioInputOption[] {
  return all
    .filter((d) => d.kind === 'audioinput')
    .map((d) => ({
      deviceId: d.deviceId,
      label: d.label,
      groupId: d.groupId,
      classification: classifySource(d.label),
    }))
}

/** True once at least one input has a real label (permission granted). */
export function hasLabeledInputs(inputs: RawMediaDevice[]): boolean {
  return inputs.some((d) => d.kind === 'audioinput' && Boolean(d.label))
}

/** Selected device vanished from the refreshed list — section 5 input-lost. */
export function isSelectedDeviceMissing(
  selectedDeviceId: string | undefined,
  devices: AudioInputOption[],
): boolean {
  if (!selectedDeviceId) return false
  return !devices.some((d) => d.deviceId === selectedDeviceId)
}
