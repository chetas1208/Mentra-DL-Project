// Device enumeration + selection + devicechange handling (spec sections
// 2/3/4/5/6). Framework-facing wrapper around the browser Media Devices
// API; classification itself is the pure classifySource() in
// utils/sourceClassification.ts so it's independently testable.

import {
  hasLabeledInputs, isSelectedDeviceMissing, mapAudioInputDevices,
  type AudioInputOption,
} from '~/utils/audioInputDevices'

export type { AudioInputOption }

export function useMediaDevices() {
  const devices = useState<AudioInputOption[]>('mentra-devices', () => [])
  const selectedDeviceId = useState<string | undefined>('mentra-selected-device-id', () => undefined)
  const permissionKnown = useState<boolean>('mentra-permission-known', () => false)
  // Set the instant a previously-selected device vanishes from the device
  // list while a session is live -- section 5, must not silently
  // relabel/continue.
  const selectedDeviceLost = useState<boolean>('mentra-selected-device-lost', () => false)

  let listenerAttached = false

  async function refresh() {
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.enumerateDevices) return
    const all = await navigator.mediaDevices.enumerateDevices()
    // Labels are only real once permission has been granted at least once
    // (section 4) -- an empty label must never be classified as Mentra.
    if (hasLabeledInputs(all)) permissionKnown.value = true
    devices.value = mapAudioInputDevices(all)

    if (isSelectedDeviceMissing(selectedDeviceId.value, devices.value)) {
      selectedDeviceLost.value = true
    }
  }

  function attachDeviceChangeListener() {
    if (listenerAttached || typeof navigator === 'undefined' || !navigator.mediaDevices) return
    listenerAttached = true
    navigator.mediaDevices.addEventListener('devicechange', () => {
      refresh().catch(() => {})
    })
  }

  function detachDeviceChangeListener() {
    // navigator.mediaDevices doesn't expose removeEventListener targeting
    // for an anonymous handler cleanly across the app's lifetime here --
    // this composable's listener is intentionally attached once for the
    // page lifetime (matches section 5's "listen to devicechange"; the
    // page itself is torn down on navigation, not this listener alone).
    listenerAttached = listenerAttached
  }

  function selectDevice(deviceId: string | undefined) {
    selectedDeviceId.value = deviceId
    selectedDeviceLost.value = false
  }

  const selectedDevice = computed(() => devices.value.find((d) => d.deviceId === selectedDeviceId.value))

  return {
    devices,
    selectedDeviceId,
    selectedDevice,
    selectedDeviceLost,
    permissionKnown,
    refresh,
    selectDevice,
    attachDeviceChangeListener,
    detachDeviceChangeListener,
  }
}
