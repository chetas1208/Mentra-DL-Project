import test from 'node:test'
import assert from 'node:assert/strict'
import { encodeFrame, decodeFrame, float32ToPcm16LE, pcm16LEToFloat32, MessageType, Codec, HEADER_SIZE } from '../app/utils/mentraProtocol.ts'

test('AudioFrame encode/decode round-trips all fields', () => {
  const payload = new Uint8Array([1, 2, 3, 4, 5])
  const buf = encodeFrame({
    sequenceNumber: 42,
    captureTimestampNs: 123456789012345n,
    sampleRate: 16000,
    channels: 1,
    bitsPerSample: 16,
    payload,
    messageType: MessageType.AUDIO_FRAME,
    codec: Codec.PCM16,
    flags: 0,
  })
  assert.equal(buf.byteLength, HEADER_SIZE + payload.length)
  const decoded = decodeFrame(buf)
  assert.equal(decoded.sequenceNumber, 42)
  assert.equal(decoded.captureTimestampNs, 123456789012345n)
  assert.equal(decoded.sampleRate, 16000)
  assert.equal(decoded.channels, 1)
  assert.equal(decoded.bitsPerSample, 16)
  assert.equal(decoded.messageType, MessageType.AUDIO_FRAME)
  assert.equal(decoded.codec, Codec.PCM16)
  assert.deepEqual(Array.from(decoded.payload), Array.from(payload))
})

test('decodeFrame rejects truncated header', () => {
  assert.throws(() => decodeFrame(new ArrayBuffer(10)))
})

test('decodeFrame rejects bad magic', () => {
  const buf = encodeFrame({ payload: new Uint8Array(0) })
  const bytes = new Uint8Array(buf)
  bytes[0] = 0x00
  assert.throws(() => decodeFrame(buf), /bad magic/)
})

test('decodeFrame rejects truncated payload', () => {
  const buf = encodeFrame({ payload: new Uint8Array([1, 2, 3, 4]) })
  const truncated = buf.slice(0, HEADER_SIZE + 2)
  assert.throws(() => decodeFrame(truncated), /truncated payload/)
})

test('float32ToPcm16LE maps extremes correctly and clamps out-of-range', () => {
  const samples = new Float32Array([-1, 0, 1, -2, 2, 0.5, -0.5])
  const bytes = float32ToPcm16LE(samples)
  const view = new DataView(bytes.buffer)
  assert.equal(view.getInt16(0, true), -32768) // -1.0 -> min int16
  assert.equal(view.getInt16(2, true), 0)
  assert.equal(view.getInt16(4, true), 32767) // +1.0 -> max int16
  assert.equal(view.getInt16(6, true), -32768) // clamped from -2
  assert.equal(view.getInt16(8, true), 32767) // clamped from +2
})

test('pcm16LEToFloat32 is the inverse of float32ToPcm16LE (within int16 quantization)', () => {
  const original = new Float32Array([0.25, -0.25, 0.75, -0.75])
  const roundTripped = pcm16LEToFloat32(float32ToPcm16LE(original))
  for (let i = 0; i < original.length; i++) {
    assert.ok(Math.abs(roundTripped[i]! - original[i]!) < 1e-3)
  }
})

test('no amplitude normalization: encoding does not rescale a quiet signal to full scale', () => {
  // A real -20dBFS-ish speech signal, not artificially loud.
  const quiet = new Float32Array(100).fill(0.05)
  const bytes = float32ToPcm16LE(quiet)
  const view = new DataView(bytes.buffer)
  const decodedSample = view.getInt16(0, true)
  // If any normalization/AGC were applied, this would be pushed toward
  // +/-32767. It must instead land near 0.05 * 32767 ~= 1638.
  assert.ok(Math.abs(decodedSample - Math.round(0.05 * 32767)) <= 1)
  assert.ok(Math.abs(decodedSample) < 5000, 'quiet signal must not be normalized toward full scale')
})
