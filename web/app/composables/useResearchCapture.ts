// G4 WS6 -- research capture mode, layered on the EXISTING capture chain.
//
// This adds no new way of getting audio. It reuses useMicCapture (getUserMedia
// -> AudioWorklet -> streaming resampler, AGC/NS/EC requested off and the
// GRANTED settings read back) and useMentraTransport (the same MTRA binary
// WebSocket protocol) exactly as the live page does. What it adds is the
// research bookkeeping a pilot session needs and the live demo never did:
//
//   * a session id and an ANONYMOUS wearer id (random; not derived from the
//     person, their voice, or their device)
//   * a required condition label -- an unlabelled take cannot be compared to
//     anything, so the backend rejects one
//   * the device/format provenance actually granted by the browser
//   * the NATIVE-rate raw PCM, tapped before the resampler, in addition to
//     the derived 16kHz stream the model consumes
//
// HONESTY: `sourceKind` is declared by the operator and travels verbatim to
// the backend, which stamps `is_mentra_hardware_audio` from it. Selecting the
// wrong value is the one thing this UI cannot catch, which is why the value is
// an explicit choice with no default that implies Mentra hardware.

import { float32ToPcm16LE } from '~/utils/mentraProtocol'

export const SOURCE_KINDS = [
  'MENTRA_LIVE_GLASSES_MIC',
  'MENTRA_COMPATIBLE_GLASSES_MIC',
  'BROWSER_DEFAULT_MIC',
  'SYNTHETIC_TEST_INPUT_NOT_MENTRA_AUDIO',
] as const
export type SourceKind = (typeof SOURCE_KINDS)[number]

export const MENTRA_HARDWARE_SOURCES: SourceKind[] = [
  'MENTRA_LIVE_GLASSES_MIC',
  'MENTRA_COMPATIBLE_GLASSES_MIC',
]

export interface CaptureRequest {
  sessionId: string
  wearerId: string
  condition: string
  sourceKind: SourceKind
  glassesModel?: string
  mentraosVersion?: string
  firmwareVersion?: string
  notes?: string
}

/** Random, opaque, and not reconstructible from anything about the person.
 * Re-using a handle across sessions is the operator's decision, recorded on
 * their own run sheet -- deliberately not something this code can infer. */
export function newAnonymousWearerId(): string {
  return `W-${crypto.randomUUID().replace(/-/g, '').slice(0, 8)}`
}

export function newSessionId(): string {
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+$/, '')
  return `S-${stamp}-${crypto.randomUUID().slice(0, 8)}`
}

export function useResearchCapture() {
  const isRecording = useState('mentra-capture-recording', () => false)
  const activeSessionId = useState<string | null>('mentra-capture-session', () => null)
  const nativeSampleRate = useState<number | null>('mentra-capture-native-rate', () => null)
  const rawSecondsSent = useState<number>('mentra-capture-raw-seconds', () => 0)
  const captureError = useState<string | null>('mentra-capture-error-research', () => null)

  const mic = useMicCapture()
  const transport = useMentraTransport()

  let declaredRate: number | null = null
  let declaredChannels = 1

  /** Build the metadata payload from what the browser ACTUALLY granted, never
   * from what was requested. Unknown fields stay null rather than guessed. */
  function buildMetadata(request: CaptureRequest, sampleRate: number, channels: number) {
    const format = mic.captureFormat.value
    return {
      session_id: request.sessionId,
      wearer_id: request.wearerId,
      condition: request.condition,
      source_kind: request.sourceKind,
      started_utc: new Date().toISOString(),
      raw_sample_rate: sampleRate,
      raw_channels: channels,
      derived_sample_rate: 16000,
      derived_channels: 1,
      bits_per_sample: 16,
      echo_cancellation: format.echoCancellation,
      noise_suppression: format.noiseSuppression,
      auto_gain_control: format.autoGainControl,
      device_label: null,
      device_id_hash: null,
      user_agent: typeof navigator !== 'undefined' ? navigator.userAgent : null,
      platform: typeof navigator !== 'undefined' ? navigator.platform : null,
      client_version: 'mentra-web-capture-0.1',
      glasses_model: request.glassesModel || null,
      mentraos_version: request.mentraosVersion || null,
      firmware_version: request.firmwareVersion || null,
      notes: request.notes || '',
    }
  }

  /** Begin recording. Capture must already be running (the transport's 16kHz
   * AUDIO_FRAMEs are the derived stream), so this only attaches the raw tap
   * and opens the backend session. */
  function begin(request: CaptureRequest): boolean {
    captureError.value = null
    const rate = mic.captureFormat.value.sampleRate
    if (!mic.isCapturing.value || !rate) {
      captureError.value = 'Start audio capture before beginning a research take.'
      return false
    }
    if (!request.condition.trim()) {
      captureError.value = 'A condition label is required.'
      return false
    }
    declaredRate = rate
    declaredChannels = mic.captureFormat.value.channelCount ?? 1
    nativeSampleRate.value = rate

    const ok = transport.sendCaptureMeta(
      buildMetadata(request, declaredRate, declaredChannels),
    )
    if (!ok) {
      captureError.value = 'Not connected to the backend.'
      return false
    }

    mic.setNativeBlockListener((samples, sampleRate, captureTimestampNs) => {
      // If the browser changes the context rate mid-session the take is no
      // longer one consistent recording. Stop rather than silently mixing
      // rates -- the backend would reject the chunk anyway.
      if (declaredRate !== null && sampleRate !== declaredRate) {
        captureError.value = `Native sample rate changed ${declaredRate} -> ${sampleRate} Hz; take stopped.`
        end()
        return
      }
      transport.sendCaptureRawPcm(
        float32ToPcm16LE(samples), sampleRate, declaredChannels, captureTimestampNs,
      )
      rawSecondsSent.value += samples.length / sampleRate
    })

    activeSessionId.value = request.sessionId
    isRecording.value = true
    rawSecondsSent.value = 0
    return true
  }

  /** Stop the raw tap. The backend closes, validates and writes the session
   * when the transport session ends, so the operator disconnects to finalise. */
  function end() {
    mic.setNativeBlockListener(null)
    isRecording.value = false
  }

  return {
    isRecording,
    activeSessionId,
    nativeSampleRate,
    rawSecondsSent,
    captureError,
    captureAck: transport.captureAck,
    rawFramesSent: transport.captureRawFramesSent,
    rawFramesDropped: transport.captureRawDropped,
    begin,
    end,
    newAnonymousWearerId,
    newSessionId,
  }
}
