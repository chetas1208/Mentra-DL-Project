// Binary audio frame protocol -- mirrors mentra/audio/frame.py exactly.
// Fixed 30-byte header, network byte order (big-endian), no padding.
// See that file's docstring for the authoritative field-by-field spec;
// this is a same-wire-format reimplementation, not a copy of the Python.

export const MAGIC = new Uint8Array([0x4d, 0x54, 0x52, 0x41]) // "MTRA"
export const PROTOCOL_VERSION = 1
export const HEADER_SIZE = 30

export enum MessageType {
  STREAM_START = 1,
  STREAM_ACCEPTED = 2,
  STREAM_REJECTED = 3,
  AUDIO_FRAME = 4,
  PING = 5,
  PONG = 6,
  DETECTION = 7,
  WEARER_PCM = 8,
  ENVIRONMENT_PCM = 9,
  TRANSCRIPT = 10,
  ENROLL_AUDIO = 11,
  ENROLL_DONE = 12,
  SET_PLAYBACK_MODE = 13,
}

export enum Codec {
  PCM16 = 1,
  LC3 = 2,
}

export interface MentraFrame {
  sequenceNumber: number
  captureTimestampNs: bigint
  sampleRate: number
  channels: number
  bitsPerSample: number
  payload: Uint8Array
  messageType: MessageType
  codec: Codec
  flags: number
}

export function encodeFrame(f: Partial<MentraFrame> & { payload: Uint8Array }): ArrayBuffer {
  const messageType = f.messageType ?? MessageType.AUDIO_FRAME
  const codec = f.codec ?? Codec.PCM16
  const buf = new ArrayBuffer(HEADER_SIZE + f.payload.length)
  const view = new DataView(buf)
  const bytes = new Uint8Array(buf)

  bytes.set(MAGIC, 0)
  view.setUint8(4, PROTOCOL_VERSION)
  view.setUint8(5, messageType)
  view.setUint8(6, codec)
  view.setUint8(7, f.flags ?? 0)
  view.setUint32(8, (f.sequenceNumber ?? 0) >>> 0, false)
  view.setBigUint64(12, f.captureTimestampNs ?? 0n, false)
  view.setUint32(20, f.sampleRate ?? 16000, false)
  view.setUint8(24, f.channels ?? 1)
  view.setUint8(25, f.bitsPerSample ?? 16)
  view.setUint32(26, f.payload.length, false)
  bytes.set(f.payload, HEADER_SIZE)

  return buf
}

export function decodeFrame(raw: ArrayBuffer): MentraFrame {
  if (raw.byteLength < HEADER_SIZE) {
    throw new Error(`truncated header: got ${raw.byteLength} bytes, need >= ${HEADER_SIZE}`)
  }
  const view = new DataView(raw)
  const bytes = new Uint8Array(raw)

  for (let i = 0; i < 4; i++) {
    if (bytes[i] !== MAGIC[i]) throw new Error('bad magic')
  }
  const version = view.getUint8(4)
  if (version !== PROTOCOL_VERSION) throw new Error(`unsupported protocol version: ${version}`)

  const messageType = view.getUint8(5) as MessageType
  const codec = view.getUint8(6) as Codec
  const flags = view.getUint8(7)
  const sequenceNumber = view.getUint32(8, false)
  const captureTimestampNs = view.getBigUint64(12, false)
  const sampleRate = view.getUint32(20, false)
  const channels = view.getUint8(24)
  const bitsPerSample = view.getUint8(25)
  const payloadLength = view.getUint32(26, false)

  if (raw.byteLength - HEADER_SIZE < payloadLength) {
    throw new Error(`truncated payload: declared ${payloadLength}, got ${raw.byteLength - HEADER_SIZE}`)
  }
  const payload = bytes.slice(HEADER_SIZE, HEADER_SIZE + payloadLength)

  return { sequenceNumber, captureTimestampNs, sampleRate, channels, bitsPerSample, payload, messageType, codec, flags }
}

/** Converts Float32 samples in [-1, 1] (Web Audio's native format) to
 * little-endian PCM16 bytes -- the payload format the backend expects
 * (research/sherpa_onnx side already assumes LE int16, matching
 * mentra/audio/consumer.py's pcm16_bytes_to_float32). */
export function float32ToPcm16LE(samples: Float32Array): Uint8Array {
  const out = new Uint8Array(samples.length * 2)
  const view = new DataView(out.buffer)
  for (let i = 0; i < samples.length; i++) {
    const clipped = Math.max(-1, Math.min(1, samples[i] ?? 0))
    const int16 = clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff
    view.setInt16(i * 2, Math.round(int16), true) // little-endian, matches PCM16 payload convention
  }
  return out
}

/** Inverse of float32ToPcm16LE -- little-endian PCM16 bytes back to Float32
 * in [-1, 1], for scheduling gated-audio playback via Web Audio. */
export function pcm16LEToFloat32(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const out = new Float32Array(bytes.length / 2)
  for (let i = 0; i < out.length; i++) {
    out[i] = view.getInt16(i * 2, true) / 32768
  }
  return out
}

export function nowNs(): bigint {
  // performance.now() is float milliseconds with sub-ms precision but not
  // a true monotonic nanosecond clock -- good enough for relative latency
  // measurement across this session, not for cross-process clock sync.
  return BigInt(Math.round(performance.now() * 1e6))
}
