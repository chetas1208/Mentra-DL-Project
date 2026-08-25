import test from 'node:test'
import assert from 'node:assert/strict'
import { StreamingResampler, FixedSizeFramer } from '../app/utils/streamingResampler.ts'

function sineWave(n: number, freq: number, sampleRate: number): Float32Array {
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = Math.sin((2 * Math.PI * freq * i) / sampleRate) * 0.5
  return out
}

test('16k -> 16k is exact passthrough (identity, no drift)', () => {
  const r = new StreamingResampler(16000, 16000)
  const input = sineWave(1600, 300, 16000)
  const output = r.process(input)
  assert.deepEqual(Array.from(output), Array.from(input))
})

test('48k -> 16k: output count converges to input/3 across many chunks', () => {
  const r = new StreamingResampler(48000, 16000)
  const chunkSize = 480 // 10ms @ 48kHz
  const numChunks = 100
  let totalOut = 0
  for (let c = 0; c < numChunks; c++) {
    const chunk = sineWave(chunkSize, 300, 48000)
    totalOut += r.process(chunk).length
  }
  const totalIn = chunkSize * numChunks
  const expected = totalIn / 3
  // allow +/- 2 samples of rounding across the whole run
  assert.ok(Math.abs(totalOut - expected) <= 2, `expected ~${expected}, got ${totalOut}`)
})

test('44.1k -> 16k: output count converges to the correct ratio', () => {
  const r = new StreamingResampler(44100, 16000)
  const chunkSize = 441 // 10ms @ 44.1kHz
  const numChunks = 200
  let totalOut = 0
  for (let c = 0; c < numChunks; c++) {
    totalOut += r.process(sineWave(chunkSize, 300, 44100)).length
  }
  const totalIn = chunkSize * numChunks
  const expected = totalIn * (16000 / 44100)
  const relError = Math.abs(totalOut - expected) / expected
  assert.ok(relError < 0.01, `expected ~${expected}, got ${totalOut} (relError=${relError})`)
})

test('phase carries continuously across chunk boundaries (no periodic reset click)', () => {
  // Process the same signal as one big chunk vs many small chunks --
  // results must match closely; a phase-reset bug would show up as
  // periodic large deviations at chunk boundaries.
  const full = sineWave(4800, 300, 48000)
  const whole = new StreamingResampler(48000, 16000).process(full)

  const chunked = new StreamingResampler(48000, 16000)
  const pieces: Float32Array[] = []
  for (let i = 0; i < full.length; i += 160) {
    pieces.push(chunked.process(full.slice(i, i + 160)))
  }
  const stitched = new Float32Array(pieces.reduce((n, p) => n + p.length, 0))
  let offset = 0
  for (const p of pieces) {
    stitched.set(p, offset)
    offset += p.length
  }

  assert.ok(Math.abs(whole.length - stitched.length) <= 2)
  const n = Math.min(whole.length, stitched.length)
  let maxDiff = 0
  for (let i = 0; i < n; i++) maxDiff = Math.max(maxDiff, Math.abs(whole[i]! - stitched[i]!))
  assert.ok(maxDiff < 0.05, `chunk-boundary phase discontinuity too large: ${maxDiff}`)
})

test('resampler does not blow up on empty or single-sample chunks', () => {
  const r = new StreamingResampler(48000, 16000)
  assert.equal(r.process(new Float32Array(0)).length, 0)
  const single = r.process(new Float32Array([0.3]))
  assert.ok(single.length <= 1)
})

test('FixedSizeFramer releases exactly frameSamples-sized frames and carries remainder', () => {
  const framer = new FixedSizeFramer(160)
  const frames1 = framer.push(new Float32Array(100))
  assert.equal(frames1.length, 0)
  assert.equal(framer.pendingSamples, 100)

  const frames2 = framer.push(new Float32Array(100))
  assert.equal(frames2.length, 1)
  assert.equal(frames2[0]!.length, 160)
  assert.equal(framer.pendingSamples, 40)
})

test('FixedSizeFramer can release multiple frames from one large push', () => {
  const framer = new FixedSizeFramer(160)
  const frames = framer.push(new Float32Array(500))
  assert.equal(frames.length, 3) // 480 samples released, 20 carried
  assert.equal(framer.pendingSamples, 20)
})
