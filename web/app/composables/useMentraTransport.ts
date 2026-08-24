// Browser-side transport client speaking the SAME binary protocol as
// mentra/audio/transport/websocket_transport.py. Talks directly to
// server/audio/remote_receiver.py -- no separate frontend backend/proxy.

import { encodeFrame, decodeFrame, MessageType, Codec, nowNs, type MentraFrame } from '~/utils/mentraProtocol'

export interface DetectionResult {
  wearerScore: number
  state: string
  contextMs: number
  inferenceMs: number
  captureToPredictionMs: number
}

export type ConnectionState = 'DISCONNECTED' | 'CONNECTING' | 'CONNECTED' | 'REJECTED' | 'DEAD'

export interface GatedAudioHandlers {
  onWearerPcm?: (bytes: Uint8Array) => void
  onEnvironmentPcm?: (bytes: Uint8Array) => void
}

export function useMentraTransport() {
  const config = useRuntimeConfig()
  const connectionState = useState<ConnectionState>('mentra-connection-state', () => 'DISCONNECTED')
  const lastResult = useState<DetectionResult | null>('mentra-last-result', () => null)
  const framesSent = useState<number>('mentra-frames-sent', () => 0)
  const lastRttMs = useState<number | null>('mentra-rtt-ms', () => null)
  const transcript = useState<string>('mentra-transcript', () => '')
  const enrollmentStatus = useState<'PLACEHOLDER' | 'ENROLLING' | 'REAL'>('mentra-enrollment-status', () => 'PLACEHOLDER')

  let ws: WebSocket | null = null
  let enrollSequenceNumber = 0
  let sequenceNumber = 0
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null
  let lastPingSentNs: bigint | null = null

  // Imperative callbacks, not reactive state -- gated audio arrives every
  // ~10ms and pushing that through Vue's reactivity system would add
  // needless overhead for something that's just handed straight to
  // useGatedAudioPlayback's scheduler.
  function connect(gatedAudioHandlers?: GatedAudioHandlers): Promise<void> {
    return new Promise((resolve, reject) => {
      connectionState.value = 'CONNECTING'
      ws = new WebSocket(config.public.backendWsUrl)
      ws.binaryType = 'arraybuffer'
      sequenceNumber = 0

      ws.onopen = () => {
        const sessionId = crypto.randomUUID()
        const startFrame = encodeFrame({
          sequenceNumber: 0,
          captureTimestampNs: nowNs(),
          sampleRate: 16000, channels: 1, bitsPerSample: 16,
          payload: new TextEncoder().encode(`${sessionId}|mentra-web-0.1`),
          messageType: MessageType.STREAM_START,
          codec: Codec.PCM16,
        })
        ws!.send(startFrame)
      }

      ws.onmessage = (event: MessageEvent<ArrayBuffer>) => {
        let frame: MentraFrame
        try {
          frame = decodeFrame(event.data)
        } catch (e) {
          console.warn('malformed frame from server', e)
          return
        }

        if (frame.messageType === MessageType.STREAM_ACCEPTED) {
          connectionState.value = 'CONNECTED'
          startHeartbeat()
          resolve()
        } else if (frame.messageType === MessageType.STREAM_REJECTED) {
          connectionState.value = 'REJECTED'
          reject(new Error(new TextDecoder().decode(frame.payload)))
        } else if (frame.messageType === MessageType.PONG) {
          if (lastPingSentNs !== null) {
            lastRttMs.value = Number(nowNs() - lastPingSentNs) / 1e6
          }
        } else if (frame.messageType === MessageType.DETECTION) {
          const parsed = JSON.parse(new TextDecoder().decode(frame.payload))
          // Round-trip latency computed entirely with THIS browser's own clock
          // (send timestamp echoed back by the server, compared against our own
          // nowNs()) -- never trust a latency figure computed by mixing the
          // server's clock with a timestamp from a different machine, that
          // produces nonsense (real bug found and fixed on the server side:
          // it used to report ~811,000,000ms).
          const echoedNs = BigInt(parsed.echoed_capture_timestamp_ns)
          lastResult.value = {
            wearerScore: parsed.wearer_score,
            state: parsed.state,
            contextMs: parsed.context_ms,
            inferenceMs: parsed.inference_ms,
            captureToPredictionMs: Number(nowNs() - echoedNs) / 1e6,
          }
        } else if (frame.messageType === MessageType.ENROLL_DONE) {
          enrollmentStatus.value = 'REAL'
        } else if (frame.messageType === MessageType.TRANSCRIPT) {
          const parsed = JSON.parse(new TextDecoder().decode(frame.payload))
          transcript.value = parsed.text
        } else if (frame.messageType === MessageType.WEARER_PCM) {
          gatedAudioHandlers?.onWearerPcm?.(frame.payload)
        } else if (frame.messageType === MessageType.ENVIRONMENT_PCM) {
          gatedAudioHandlers?.onEnvironmentPcm?.(frame.payload)
        }
      }

      ws.onclose = () => {
        connectionState.value = 'DISCONNECTED'
        stopHeartbeat()
      }
      ws.onerror = (e) => {
        connectionState.value = 'DEAD'
        reject(e)
      }
    })
  }

  function startHeartbeat() {
    heartbeatTimer = setInterval(() => {
      if (ws?.readyState !== WebSocket.OPEN) return
      lastPingSentNs = nowNs()
      ws.send(encodeFrame({
        sequenceNumber: 0, captureTimestampNs: lastPingSentNs,
        sampleRate: 16000, channels: 1, bitsPerSample: 16,
        payload: new Uint8Array(0), messageType: MessageType.PING,
      }))
    }, 5000)
  }

  function stopHeartbeat() {
    if (heartbeatTimer) clearInterval(heartbeatTimer)
    heartbeatTimer = null
  }

  // Tells the server which of WEARER_PCM/ENVIRONMENT_PCM we actually want,
  // so it can stop building+sending the unselected stream entirely instead
  // of us just zeroing its gain locally after it's already been sent --
  // real bandwidth/compute saved server-side, not just a client-side mute.
  function setPlaybackModePreference(mode: 'both' | 'wearer' | 'environment' | 'muted') {
    if (ws?.readyState !== WebSocket.OPEN) return
    ws.send(encodeFrame({
      sequenceNumber: 0, captureTimestampNs: nowNs(),
      sampleRate: 16000, channels: 1, bitsPerSample: 16,
      payload: new TextEncoder().encode(mode),
      messageType: MessageType.SET_PLAYBACK_MODE,
    }))
  }

  function sendEnrollmentAudio(pcm16: Uint8Array) {
    if (ws?.readyState !== WebSocket.OPEN) return
    enrollmentStatus.value = 'ENROLLING'
    ws.send(encodeFrame({
      sequenceNumber: enrollSequenceNumber++,
      captureTimestampNs: nowNs(),
      sampleRate: 16000, channels: 1, bitsPerSample: 16,
      payload: pcm16,
      messageType: MessageType.ENROLL_AUDIO,
      codec: Codec.PCM16,
    }))
  }

  function sendAudioFrame(pcm16: Uint8Array, captureTimestampNs: bigint) {
    if (ws?.readyState !== WebSocket.OPEN) return
    ws.send(encodeFrame({
      sequenceNumber: sequenceNumber++,
      captureTimestampNs,
      sampleRate: 16000, channels: 1, bitsPerSample: 16,
      payload: pcm16,
      messageType: MessageType.AUDIO_FRAME,
      codec: Codec.PCM16,
    }))
    framesSent.value++
  }

  function disconnect() {
    stopHeartbeat()
    ws?.close()
    ws = null
    connectionState.value = 'DISCONNECTED'
  }

  return { connectionState, lastResult, framesSent, lastRttMs, transcript, enrollmentStatus, connect, disconnect, sendAudioFrame, sendEnrollmentAudio, setPlaybackModePreference }
}
