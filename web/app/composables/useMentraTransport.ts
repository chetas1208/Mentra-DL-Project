// Browser-side transport client speaking the SAME binary protocol as
// mentra/audio/transport/websocket_transport.py. Talks directly to
// server/audio/remote_receiver.py -- no separate frontend backend/proxy.
//
// Implements: explicit WebSocket state machine (section 20), bounded
// exponential-backoff reconnect (section 21), bounded backpressure queue
// with drop-oldest + real drop telemetry (section 22), the existing
// PING/PONG heartbeat (section 23), and parses the model-capabilities
// payload the backend now sends inside STREAM_ACCEPTED (section 24) --
// this is the SAME message type as before (STREAM_ACCEPTED), just no
// longer always empty; a receiver build that still sends an empty accept
// payload parses to capabilities=null (NOT AVAILABLE), never a guess.

import { encodeFrame, decodeFrame, MessageType, Codec, nowNs, type MentraFrame } from '~/utils/mentraProtocol'
import { parseCapabilities, type ModelCapabilities } from '~/utils/modelCapabilities'
import { BoundedAudioQueue, type QueuedAudioFrame } from '~/utils/boundedAudioQueue'
import { nextBackoffDelayMs, DEFAULT_BACKOFF } from '~/utils/reconnectBackoff'

export interface DetectionResult {
  wearerScore: number
  state: string
  contextMs: number
  inferenceMs: number
  captureToPredictionMs: number
}

// Explicit states (spec section 20) -- never collapse this to one boolean.
export type WsState = 'IDLE' | 'CONNECTING' | 'CONNECTED' | 'RECONNECTING' | 'ERROR' | 'CLOSING' | 'CLOSED'

export interface GatedAudioHandlers {
  onWearerPcm?: (bytes: Uint8Array) => void
  onEnvironmentPcm?: (bytes: Uint8Array) => void
}

const MAX_QUEUE_FRAMES = 50 // ~500ms of backlog at 10ms/frame -- recent audio matters more than stale backlog
const BUFFERED_AMOUNT_HIGH_WATER = 64 * 1024 // bytes; above this, new AUDIO_FRAMEs queue instead of sending immediately
const QUEUE_DRAIN_INTERVAL_MS = 50

export function useMentraTransport() {
  const config = useRuntimeConfig()
  const wsState = useState<WsState>('mentra-ws-state', () => 'IDLE')
  const rejectionReason = useState<string | null>('mentra-rejection-reason', () => null)
  const lastResult = useState<DetectionResult | null>('mentra-last-result', () => null)
  const framesSent = useState<number>('mentra-frames-sent', () => 0)
  const lastRttMs = useState<number | null>('mentra-rtt-ms', () => null)
  const transcript = useState<string>('mentra-transcript', () => '')
  const enrollmentStatus = useState<'PLACEHOLDER' | 'ENROLLING' | 'REAL'>('mentra-enrollment-status', () => 'PLACEHOLDER')
  const capabilities = useState<ModelCapabilities | null>('mentra-capabilities', () => null)
  const sessionId = useState<string | null>('mentra-session-id', () => null)
  const queueDepth = useState<number>('mentra-queue-depth', () => 0)
  const queueHighWaterMark = useState<number>('mentra-queue-high-water', () => 0)
  const framesDropped = useState<number>('mentra-frames-dropped', () => 0)
  const bytesDropped = useState<number>('mentra-bytes-dropped', () => 0)
  const reconnectCount = useState<number>('mentra-reconnect-count', () => 0)

  let ws: WebSocket | null = null
  let enrollSequenceNumber = 0
  let sequenceNumber = 0
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null
  let drainTimer: ReturnType<typeof setInterval> | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let lastPingSentNs: bigint | null = null
  let userRequestedDisconnect = false
  let reconnectAttempt = 0
  let currentGatedHandlers: GatedAudioHandlers | undefined
  const outboundQueue = new BoundedAudioQueue(MAX_QUEUE_FRAMES)

  function resetSessionCounters() {
    sequenceNumber = 0
    enrollSequenceNumber = 0
    reconnectAttempt = 0
    outboundQueue.clear()
    queueDepth.value = 0
    queueHighWaterMark.value = 0
    framesDropped.value = 0
    bytesDropped.value = 0
    reconnectCount.value = 0
  }

  function openSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      wsState.value = reconnectAttempt > 0 ? 'RECONNECTING' : 'CONNECTING'
      ws = new WebSocket(config.public.backendWsUrl)
      ws.binaryType = 'arraybuffer'

      ws.onopen = () => {
        const sid = sessionId.value ?? crypto.randomUUID()
        sessionId.value = sid
        const startFrame = encodeFrame({
          sequenceNumber: 0,
          captureTimestampNs: nowNs(),
          sampleRate: 16000, channels: 1, bitsPerSample: 16,
          payload: new TextEncoder().encode(`${sid}|mentra-web-0.1`),
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
          wsState.value = 'CONNECTED'
          capabilities.value = parseCapabilities(frame.payload)
          reconnectAttempt = 0
          startHeartbeat()
          startQueueDrain()
          resolve()
        } else if (frame.messageType === MessageType.STREAM_REJECTED) {
          rejectionReason.value = new TextDecoder().decode(frame.payload)
          wsState.value = 'ERROR'
          reject(new Error(rejectionReason.value))
        } else if (frame.messageType === MessageType.PONG) {
          if (lastPingSentNs !== null) {
            lastRttMs.value = Number(nowNs() - lastPingSentNs) / 1e6
          }
        } else if (frame.messageType === MessageType.DETECTION) {
          const parsed = JSON.parse(new TextDecoder().decode(frame.payload))
          // Round-trip latency computed entirely with THIS browser's own clock
          // (send timestamp echoed back by the server, compared against our own
          // nowNs()) -- never trust a latency figure computed by mixing the
          // server's clock with a timestamp from a different machine.
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
          currentGatedHandlers?.onWearerPcm?.(frame.payload)
        } else if (frame.messageType === MessageType.ENVIRONMENT_PCM) {
          currentGatedHandlers?.onEnvironmentPcm?.(frame.payload)
        }
      }

      ws.onclose = () => {
        stopHeartbeat()
        stopQueueDrain()
        if (userRequestedDisconnect || wsState.value === 'ERROR') {
          wsState.value = 'CLOSED'
          return
        }
        scheduleReconnect()
      }
      ws.onerror = (e) => {
        wsState.value = 'ERROR'
        reject(e)
      }
    })
  }

  function scheduleReconnect() {
    reconnectAttempt += 1
    const delay = nextBackoffDelayMs(reconnectAttempt, DEFAULT_BACKOFF)
    if (delay === null) {
      wsState.value = 'ERROR'
      return
    }
    wsState.value = 'RECONNECTING'
    reconnectCount.value++
    reconnectTimer = setTimeout(() => {
      openSocket().catch(() => {
        // openSocket's own onclose/onerror handlers drive the next retry
        // or terminal ERROR state -- nothing further to do here.
      })
    }, delay)
  }

  function connect(gatedAudioHandlers?: GatedAudioHandlers): Promise<void> {
    userRequestedDisconnect = false
    currentGatedHandlers = gatedAudioHandlers
    resetSessionCounters()
    sessionId.value = null
    rejectionReason.value = null
    return openSocket()
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

  // --- backpressure / bounded outbound queue (section 22) -----------------
  function sendQueuedFrame(qf: QueuedAudioFrame) {
    ws!.send(encodeFrame({
      sequenceNumber: qf.sequenceNumber,
      captureTimestampNs: qf.captureTimestampNs,
      sampleRate: 16000, channels: 1, bitsPerSample: 16,
      payload: qf.pcm16,
      messageType: MessageType.AUDIO_FRAME,
      codec: Codec.PCM16,
    }))
    framesSent.value++
  }

  function drainQueue() {
    if (ws?.readyState !== WebSocket.OPEN) return
    while (outboundQueue.depth > 0 && ws.bufferedAmount < BUFFERED_AMOUNT_HIGH_WATER) {
      const qf = outboundQueue.shift()
      if (!qf) break
      sendQueuedFrame(qf)
    }
    queueDepth.value = outboundQueue.depth
  }

  function startQueueDrain() {
    drainTimer = setInterval(drainQueue, QUEUE_DRAIN_INTERVAL_MS)
  }

  function stopQueueDrain() {
    if (drainTimer) clearInterval(drainTimer)
    drainTimer = null
  }

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
    if (!ws || (ws.readyState !== WebSocket.OPEN)) return
    const seq = sequenceNumber++
    const qf: QueuedAudioFrame = { sequenceNumber: seq, captureTimestampNs, pcm16 }

    if (outboundQueue.depth === 0 && ws.bufferedAmount < BUFFERED_AMOUNT_HIGH_WATER) {
      sendQueuedFrame(qf)
      return
    }
    const drop = outboundQueue.push(qf)
    if (drop) {
      framesDropped.value += drop.framesDropped
      bytesDropped.value += drop.bytesDropped
    }
    queueDepth.value = outboundQueue.depth
    queueHighWaterMark.value = Math.max(queueHighWaterMark.value, outboundQueue.highWaterMark)
  }

  function disconnect() {
    userRequestedDisconnect = true
    if (reconnectTimer) clearTimeout(reconnectTimer)
    reconnectTimer = null
    stopHeartbeat()
    stopQueueDrain()
    outboundQueue.clear()
    queueDepth.value = 0
    wsState.value = 'CLOSING'
    ws?.close()
    ws = null
    wsState.value = 'CLOSED'
    capabilities.value = null
    sessionId.value = null
  }

  /** On-demand, non-reactive -- deliberately not pushed through Vue state
   * every frame (section 46); the debug drawer polls this itself at a low
   * rate. Returns null when there's no live socket, never a stale 0. */
  function getBufferedAmount(): number | null {
    return ws?.bufferedAmount ?? null
  }

  return {
    wsState, lastResult, framesSent, lastRttMs, transcript, enrollmentStatus,
    capabilities, sessionId, rejectionReason,
    queueDepth, queueHighWaterMark, framesDropped, bytesDropped, reconnectCount,
    connect, disconnect, sendAudioFrame, sendEnrollmentAudio, setPlaybackModePreference,
    getBufferedAmount,
  }
}
