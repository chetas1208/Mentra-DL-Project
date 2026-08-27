// One coherent runtime store (spec section 26) -- derives a single
// structured view from the underlying composables (useMicCapture,
// useMentraTransport, useMediaDevices) rather than duplicating their
// state. Every visible UI element should read from here (or from the
// same underlying useState sources) so no two components can disagree
// about what's actually happening.

import { deriveGlobalStatus, type GlobalStatus } from '~/utils/runtimeStatus'
import { classifySource } from '~/utils/sourceClassification'
import { selectableModels } from '~/utils/modelCatalog'

export function useRuntimeState() {
  const {
    devices, selectedDeviceId, selectedDevice, selectedDeviceLost,
    refresh: refreshDevices, selectDevice, attachDeviceChangeListener,
  } = useMediaDevices()
  const mic = useMicCapture()
  const transport = useMentraTransport()

  const sessionStartedAt = useState<number | null>('mentra-session-started-at', () => null)
  const wasStopped = useState<boolean>('mentra-was-stopped', () => true)
  const hasError = useState<boolean>('mentra-has-error', () => false)
  const isStarting = useState<boolean>('mentra-is-starting', () => false)

  // Once capture has actually started, the browser may have resolved
  // "default device" to a concrete deviceId (mic.captureFormat.deviceId)
  // that differs from what was requested -- prefer that real value for
  // both the label and the classification (section 3: verify what the
  // browser actually granted, don't just echo back the request).
  const actualDevice = computed(() => {
    const actualId = mic.captureFormat.value.deviceId
    if (actualId) return devices.value.find((d) => d.deviceId === actualId)
    return selectedDevice.value
  })
  const sourceLabel = computed(() =>
    actualDevice.value?.label || (selectedDeviceId.value ? '' : 'Default microphone'),
  )
  const sourceClassification = computed(() => classifySource(actualDevice.value?.label))

  const requiresEnrollment = computed<boolean | null>(() => transport.capabilities.value?.requiresEnrollment ?? null)
  const supportsEnrollment = computed<boolean | null>(() => transport.capabilities.value?.supportsEnrollment ?? null)
  const modelReady = computed<boolean | null>(() => transport.capabilities.value?.ready ?? null)

  // --- multi-model runtime view -------------------------------------------
  // The backend's catalog is authoritative. With no catalog (a receiver that
  // predates model selection) this collapses to the single model that
  // receiver actually reported, so the selector still tells the truth.
  const availableModels = computed(() =>
    selectableModels(transport.modelCatalog.value, transport.capabilities.value),
  )
  /** The model the backend has CONFIRMED for this session, resolved to its
   * product display name. Never derived from the user's pending choice. */
  const activeModelName = computed<string | null>(() => {
    const id = transport.activeModel.value
    if (!id) return null
    return availableModels.value.find((m) => m.id === id)?.displayName ?? id
  })
  const activeModelExperimental = computed<boolean>(() =>
    transport.capabilities.value?.experimental ?? false,
  )
  /** A model switch needs a stopped session: never carry rolling context,
   * latches or an enrollment across a live stream (preferred option A). */
  const modelSelectionLocked = computed(() => mic.isCapturing.value)

  const enrollmentSatisfied = computed(() =>
    requiresEnrollment.value === false ? true : transport.enrollmentStatus.value === 'REAL',
  )

  const audioFlowing = computed(() => mic.isCapturing.value && transport.framesSent.value > 0)

  const globalStatus = computed<GlobalStatus>(() => deriveGlobalStatus({
    hasSelectedSource: true, // a default device always counts as selected (section 3)
    sourceLost: selectedDeviceLost.value || mic.inputLost.value,
    isStarting: isStarting.value,
    captureActive: mic.isCapturing.value,
    wsState: transport.wsState.value,
    modelReady: modelReady.value,
    requiresEnrollment: requiresEnrollment.value,
    enrollmentSatisfied: enrollmentSatisfied.value,
    audioFlowing: audioFlowing.value,
    hasError: hasError.value,
    wasStopped: wasStopped.value,
  }))

  const sessionUptimeS = ref(0)
  let uptimeTimer: ReturnType<typeof setInterval> | null = null
  watch(globalStatus, (s) => {
    if (s === 'LIVE' && !uptimeTimer) {
      if (sessionStartedAt.value === null) sessionStartedAt.value = Date.now()
      uptimeTimer = setInterval(() => {
        sessionUptimeS.value = sessionStartedAt.value ? (Date.now() - sessionStartedAt.value) / 1000 : 0
      }, 1000)
    }
    if ((s === 'STOPPED' || s === 'ERROR' || s === 'SELECT AUDIO SOURCE') && uptimeTimer) {
      clearInterval(uptimeTimer)
      uptimeTimer = null
      sessionStartedAt.value = null
      sessionUptimeS.value = 0
    }
  })

  return {
    devices, selectedDeviceId, selectedDevice, selectedDeviceLost,
    refreshDevices, selectDevice, attachDeviceChangeListener,
    mic, transport,
    sourceLabel, sourceClassification,
    availableModels, activeModelName, activeModelExperimental, modelSelectionLocked,
    requiresEnrollment, supportsEnrollment, modelReady, enrollmentSatisfied, audioFlowing,
    globalStatus, sessionUptimeS,
    isStarting, wasStopped, hasError,
  }
}
