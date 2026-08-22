// Browser mic capture: getUserMedia -> AudioWorklet -> PCM16 frames.
// Browser DSP (echo cancellation/noise suppression/AGC) disabled by
// default -- we want the model to see real acoustic characteristics, not
// what Chrome/Safari decides speech should sound like (matches the
// project's existing "measure, don't assume" stance on audio processing).

import { float32ToPcm16LE, nowNs } from '~/utils/mentraProtocol'

export function useMicCapture() {
  const isCapturing = useState('mentra-is-capturing', () => false)
  const inputDevices = useState<MediaDeviceInfo[]>('mentra-input-devices', () => [])
  const actualSampleRate = useState<number | null>('mentra-actual-sample-rate', () => null)
  const micLevel = useState<number>('mentra-mic-level', () => 0) // real RMS of the last captured frame, 0..1-ish

  let audioContext: AudioContext | null = null
  let mediaStream: MediaStream | null = null
  let workletNode: AudioWorkletNode | null = null
  let sourceNode: MediaStreamAudioSourceNode | null = null

  async function listInputDevices() {
    // device labels are only populated after a getUserMedia permission
    // grant -- caller should list again after start() if labels are needed
    // before the user has granted permission once.
    const devices = await navigator.mediaDevices.enumerateDevices()
    inputDevices.value = devices.filter((d) => d.kind === 'audioinput')
  }

  async function start(deviceId: string | undefined, onFrame: (pcm16: Uint8Array, captureTimestampNs: bigint) => void) {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    })

    const track = mediaStream.getAudioTracks()[0]
    const settings = track?.getSettings()
    actualSampleRate.value = settings?.sampleRate ?? null
    // Browsers commonly refuse to honor a requested sampleRate on the
    // input constraints -- request 16kHz on the AudioContext instead
    // (more reliably honored) and log if the actual device rate differs,
    // rather than silently assuming 16kHz.
    // eslint-disable-next-line no-console
    console.info(`mic track settings: sampleRate=${settings?.sampleRate}, deviceId=${settings?.deviceId}`)

    audioContext = new AudioContext({ sampleRate: 16000 })
    if (audioContext.sampleRate !== 16000) {
      console.warn(`AudioContext honored sampleRate=${audioContext.sampleRate}, not 16000 -- ` +
        `resampling not yet implemented client-side, backend expects 16kHz`)
    }

    await audioContext.audioWorklet.addModule('/mentra-capture-worklet.js')
    sourceNode = audioContext.createMediaStreamSource(mediaStream)
    workletNode = new AudioWorkletNode(audioContext, 'mentra-capture-processor')

    workletNode.port.onmessage = (event: MessageEvent<Float32Array>) => {
      const captureTimestampNs = nowNs()
      const samples = event.data
      let sumSquares = 0
      for (let i = 0; i < samples.length; i++) sumSquares += (samples[i] ?? 0) ** 2
      micLevel.value = Math.sqrt(sumSquares / samples.length)
      const pcm16 = float32ToPcm16LE(samples)
      onFrame(pcm16, captureTimestampNs)
    }

    sourceNode.connect(workletNode)
    // Intentionally NOT connected to audioContext.destination -- we don't
    // want to play the mic input back out the speakers (feedback loop).
    isCapturing.value = true
  }

  function stop() {
    sourceNode?.disconnect()
    workletNode?.disconnect()
    mediaStream?.getTracks().forEach((t) => t.stop())
    audioContext?.close()
    sourceNode = null
    workletNode = null
    mediaStream = null
    audioContext = null
    isCapturing.value = false
  }

  return { isCapturing, inputDevices, actualSampleRate, micLevel, listInputDevices, start, stop }
}
