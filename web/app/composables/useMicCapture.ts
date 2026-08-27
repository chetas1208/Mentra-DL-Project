// Browser mic capture: getUserMedia -> AudioWorklet -> streaming resample
// -> PCM16 frames (spec sections 9-16, 27, 39, 45).
//
// CRITICAL AUDIO-PRESERVATION RULE (see top-level spec): this file must
// NEVER normalize amplitude, apply AGC, peak-normalize, or otherwise
// rescale the signal. The only transformations applied are: channel
// downmix (only if capture unexpectedly delivers >1 channel), resampling
// to 16kHz (a mathematically necessary rate conversion, not a level
// change), and clipping-safe Float32->PCM16 conversion. Browser echo
// cancellation/noise suppression/AGC are requested OFF wherever the
// browser allows it, and the ACTUAL granted settings are always read back
// from track.getSettings() and exposed -- never assumed to match the
// request (section 9).

import { float32ToPcm16LE, nowNs } from '~/utils/mentraProtocol'
import { StreamingResampler, FixedSizeFramer } from '~/utils/streamingResampler'

const TRANSPORT_SAMPLE_RATE = 16000
const TRANSPORT_FRAME_SAMPLES = 160 // 10ms @ 16kHz, matches the backend's expected cadence
const WAVEFORM_RING_SAMPLES = TRANSPORT_SAMPLE_RATE * 2 // 2s of real outgoing audio, bounded history

export interface CaptureFormat {
  sampleRate: number | null
  channelCount: number | null
  echoCancellation: boolean | null
  noiseSuppression: boolean | null
  autoGainControl: boolean | null
  deviceId: string | null
  latencyS: number | null
}

export interface RequestedConstraints {
  echoCancellation: boolean
  noiseSuppression: boolean
  autoGainControl: boolean
  channelCount: number
}

export const REQUESTED_CONSTRAINTS: RequestedConstraints = {
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
  channelCount: 1,
}

// Real outgoing 16kHz PCM, module-level plain ring buffer (NOT reactive
// Vue state) -- section 46: audio-rate data must not flow through Vue's
// reactivity system. The waveform component polls getWaveformSnapshot()
// itself on its own ~15fps timer.
let waveformRing = new Float32Array(WAVEFORM_RING_SAMPLES)
let waveformWriteIndex = 0
let waveformFilled = false

function pushToWaveformRing(samples: Float32Array) {
  for (let i = 0; i < samples.length; i++) {
    waveformRing[waveformWriteIndex] = samples[i]!
    waveformWriteIndex = (waveformWriteIndex + 1) % WAVEFORM_RING_SAMPLES
    if (waveformWriteIndex === 0) waveformFilled = true
  }
}

/** Returns the most recent `maxSamples` of REAL outgoing 16kHz PCM, oldest
 * first. Empty when nothing has been captured yet -- never synthetic. */
export function getWaveformSnapshot(maxSamples = TRANSPORT_SAMPLE_RATE): Float32Array {
  const available = waveformFilled ? WAVEFORM_RING_SAMPLES : waveformWriteIndex
  const n = Math.min(maxSamples, available)
  if (n === 0) return new Float32Array(0)
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const idx = (waveformWriteIndex - n + i + WAVEFORM_RING_SAMPLES) % WAVEFORM_RING_SAMPLES
    out[i] = waveformRing[idx]!
  }
  return out
}

function resetWaveformRing() {
  waveformRing = new Float32Array(WAVEFORM_RING_SAMPLES)
  waveformWriteIndex = 0
  waveformFilled = false
}

export function useMicCapture() {
  const isCapturing = useState('mentra-is-capturing', () => false)
  const captureFormat = useState<CaptureFormat>('mentra-capture-format', () => ({
    sampleRate: null, channelCount: null, echoCancellation: null, noiseSuppression: null,
    autoGainControl: null, deviceId: null, latencyS: null,
  }))
  const micLevel = useState<number>('mentra-mic-level', () => 0) // real RMS of the last captured block, 0..1-ish
  const framesCaptured = useState<number>('mentra-frames-captured', () => 0) // native worklet blocks received
  const framesConverted = useState<number>('mentra-frames-converted', () => 0) // 16kHz/160-sample frames emitted
  const audioContextState = useState<AudioContextState | null>('mentra-audiocontext-state', () => null)
  const inputLost = useState<boolean>('mentra-input-lost', () => false)
  const captureError = useState<string | null>('mentra-capture-error', () => null)

  let audioContext: AudioContext | null = null
  let mediaStream: MediaStream | null = null
  let workletNode: AudioWorkletNode | null = null
  let sourceNode: MediaStreamAudioSourceNode | null = null
  let resampler: StreamingResampler | null = null
  let framer: FixedSizeFramer | null = null
  let onAudioFrame: ((pcm16: Uint8Array, captureTimestampNs: bigint) => void) | null = null
  // G4 WS6: optional tap on the PRE-RESAMPLE, native-rate blocks. Research
  // capture needs the least-processed audio the application layer can see;
  // the transport path still gets the same 16kHz frames it always did, and
  // this callback is null in the normal live-demo path.
  let onNativeBlock: ((samples: Float32Array, nativeSampleRate: number, captureTimestampNs: bigint) => void) | null = null
  let trackEndedHandler: (() => void) | null = null

  /** Attach/detach the native-rate tap. Never rescales or copies the signal
   * path -- the same Float32 block that feeds the resampler is handed over. */
  function setNativeBlockListener(
    listener: ((samples: Float32Array, nativeSampleRate: number, captureTimestampNs: bigint) => void) | null,
  ) {
    onNativeBlock = listener
  }

  async function start(deviceId: string | undefined, onFrame: (pcm16: Uint8Array, captureTimestampNs: bigint) => void) {
    if (typeof window === 'undefined') throw new Error('mic capture is client-only')
    if (!window.isSecureContext) {
      throw new Error('Microphone access requires a secure context (HTTPS or localhost).')
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('This browser does not support microphone capture (getUserMedia unavailable).')
    }
    if (!('audioWorklet' in AudioContext.prototype) && !(window as unknown as { AudioWorklet?: unknown }).AudioWorklet) {
      // best-effort check; actual failure surfaces at addModule() below regardless
    }

    captureError.value = null
    inputLost.value = false
    onAudioFrame = onFrame
    resetWaveformRing()

    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        channelCount: { ideal: REQUESTED_CONSTRAINTS.channelCount },
        echoCancellation: REQUESTED_CONSTRAINTS.echoCancellation,
        noiseSuppression: REQUESTED_CONSTRAINTS.noiseSuppression,
        autoGainControl: REQUESTED_CONSTRAINTS.autoGainControl,
      },
    })

    const track = mediaStream.getAudioTracks()[0]
    if (!track) throw new Error('getUserMedia returned no audio track')

    // Never assume the browser honored what was requested (section 3/9) --
    // read back the actual granted settings.
    const settings = track.getSettings()
    captureFormat.value = {
      sampleRate: settings.sampleRate ?? null,
      channelCount: settings.channelCount ?? null,
      echoCancellation: settings.echoCancellation ?? null,
      noiseSuppression: settings.noiseSuppression ?? null,
      autoGainControl: settings.autoGainControl ?? null,
      deviceId: settings.deviceId ?? null,
      latencyS: (settings as { latency?: number }).latency ?? null,
    }

    trackEndedHandler = () => {
      // The selected device disappeared mid-session (unplugged, OS revoked
      // it, etc). Section 5: stop capture safely, mark INPUT LOST, do not
      // silently continue or relabel.
      inputLost.value = true
      stop()
    }
    track.addEventListener('ended', trackEndedHandler)

    // Do NOT force a sampleRate on the AudioContext -- browsers commonly
    // ignore that anyway (verified in this project's own testing), and
    // silently proceeding on a false assumption is worse than reading the
    // real rate back and resampling for it. AudioContext is created only
    // now, after an explicit user action (section 12), never at page load.
    audioContext = new AudioContext()
    audioContextState.value = audioContext.state
    audioContext.addEventListener('statechange', () => {
      if (audioContext) audioContextState.value = audioContext.state
    })

    resampler = new StreamingResampler(audioContext.sampleRate, TRANSPORT_SAMPLE_RATE)
    framer = new FixedSizeFramer(TRANSPORT_FRAME_SAMPLES)

    await audioContext.audioWorklet.addModule('/mentra-capture-worklet.js')
    sourceNode = audioContext.createMediaStreamSource(mediaStream)
    workletNode = new AudioWorkletNode(audioContext, 'mentra-capture-processor')

    workletNode.port.onmessage = (event: MessageEvent<{ samples: Float32Array; nativeSampleRate: number }>) => {
      const captureTimestampNs = nowNs() // browser-block-entered-pipeline timestamp, monotonic clock (section 18)
      const { samples } = event.data
      framesCaptured.value++

      let sumSquares = 0
      for (let i = 0; i < samples.length; i++) sumSquares += (samples[i] ?? 0) ** 2
      micLevel.value = Math.sqrt(sumSquares / samples.length)

      // Native-rate tap FIRST, before any resampling, so research capture
      // records the least-processed audio available to this layer.
      onNativeBlock?.(samples, audioContext!.sampleRate, captureTimestampNs)

      const resampled = resampler!.process(samples)
      const outFrames = framer!.push(resampled)
      for (const frame of outFrames) {
        pushToWaveformRing(frame)
        framesConverted.value++
        const pcm16 = float32ToPcm16LE(frame) // clamps, does NOT normalize -- see file header
        onAudioFrame?.(pcm16, captureTimestampNs)
      }
    }

    sourceNode.connect(workletNode)
    // Intentionally NOT connected to audioContext.destination -- we don't
    // want to play the mic input back out the speakers (feedback loop).
    isCapturing.value = true
  }

  async function resumeAudioContext() {
    if (audioContext && audioContext.state === 'suspended') {
      await audioContext.resume()
      audioContextState.value = audioContext.state
    }
  }

  function stop() {
    const track = mediaStream?.getAudioTracks()[0]
    if (track && trackEndedHandler) track.removeEventListener('ended', trackEndedHandler)
    trackEndedHandler = null

    sourceNode?.disconnect()
    workletNode?.disconnect()
    if (workletNode) workletNode.port.onmessage = null
    mediaStream?.getTracks().forEach((t) => t.stop())
    audioContext?.close()

    sourceNode = null
    workletNode = null
    mediaStream = null
    audioContext = null
    resampler = null
    framer = null
    onAudioFrame = null
    onNativeBlock = null

    isCapturing.value = false
    audioContextState.value = null
    micLevel.value = 0
  }

  return {
    isCapturing,
    captureFormat,
    micLevel,
    framesCaptured,
    framesConverted,
    audioContextState,
    inputLost,
    captureError,
    transportSampleRate: TRANSPORT_SAMPLE_RATE,
    requestedConstraints: REQUESTED_CONSTRAINTS,
    start,
    stop,
    resumeAudioContext,
    setNativeBlockListener,
  }
}
