// Incremental streaming audio resampler -- linear interpolation.
//
// Why linear interpolation and not a big-tap FIR/polyphase filter: this
// pipeline is feeding a speech classifier at 16kHz, not mastering audio.
// Linear interpolation is the standard low-latency choice for this job --
// it costs one multiply-add per output sample, needs only ONE sample of
// carried state across chunk boundaries (the previous chunk's last
// sample), and never buffers or looks ahead into audio that hasn't
// arrived yet. A large-tap FIR would add real algorithmic lookahead
// latency (tens of milliseconds) for anti-aliasing quality this use case
// doesn't need. This is this project's own tradeoff, not a citation of
// any external resampling paper.
//
// State carried across process() calls, per instance:
//   - `phase`: fractional read position into the CURRENT chunk, carried
//     forward so consecutive chunks stay phase-continuous (no periodic
//     "reset to 0" artifacts at chunk boundaries).
//   - `prevSample`: last sample of the previous chunk, used as the left
//     interpolation anchor for the first output sample(s) of the next
//     chunk (avoids discarding the boundary or introducing a click).
//
// Algorithm and latency: for each output sample at virtual input index
// `pos`, linearly interpolate between the input samples at floor(pos) and
// floor(pos)+1 (or `prevSample` when floor(pos) < 0, i.e. it falls in the
// previous chunk). `pos` advances by `ratio = inputRate / outputRate`
// each output sample. Algorithmic lookahead is at most 1 input sample
// (~21us at 48kHz) -- effectively zero added latency versus the input
// stream itself.
//
// Passthrough (inputRate === outputRate): bypassed entirely, samples are
// copied through unchanged with no interpolation error and no phase
// drift, ever.

export interface ResampleStats {
  inputSamplesTotal: number
  outputSamplesTotal: number
}

export class StreamingResampler {
  readonly inputRate: number
  readonly outputRate: number
  private readonly ratio: number
  private readonly passthrough: boolean
  private phase = 0 // fractional index into the chunk currently being read, carries across calls
  private prevSample = 0
  private inputSamplesTotal = 0
  private outputSamplesTotal = 0

  constructor(inputRate: number, outputRate: number) {
    if (!(inputRate > 0) || !(outputRate > 0)) {
      throw new Error(`StreamingResampler: invalid rates ${inputRate} -> ${outputRate}`)
    }
    this.inputRate = inputRate
    this.outputRate = outputRate
    this.ratio = inputRate / outputRate
    this.passthrough = inputRate === outputRate
  }

  /** Resamples one chunk, maintaining phase/edge-sample state for the
   * next call. Chunks may be any length, including very short ones. */
  process(input: Float32Array): Float32Array {
    this.inputSamplesTotal += input.length
    if (input.length === 0) return new Float32Array(0)

    if (this.passthrough) {
      this.outputSamplesTotal += input.length
      return input.slice()
    }

    const out: number[] = []
    let pos = this.phase
    while (pos < input.length) {
      const leftIndex = Math.floor(pos)
      const frac = pos - leftIndex
      const left = leftIndex < 0 ? this.prevSample : input[leftIndex]!
      const rightIndex = leftIndex + 1
      const right = rightIndex < 0
        ? this.prevSample
        : rightIndex < input.length
          ? input[rightIndex]!
          : input[input.length - 1]! // last sample repeats as the anchor until the next chunk arrives
      out.push(left + (right - left) * frac)
      pos += this.ratio
    }
    this.phase = pos - input.length // carry the fractional remainder into the next chunk
    this.prevSample = input[input.length - 1]!
    this.outputSamplesTotal += out.length
    return Float32Array.from(out)
  }

  get stats(): ResampleStats {
    return { inputSamplesTotal: this.inputSamplesTotal, outputSamplesTotal: this.outputSamplesTotal }
  }

  reset(): void {
    this.phase = 0
    this.prevSample = 0
    this.inputSamplesTotal = 0
    this.outputSamplesTotal = 0
  }
}

/** Accumulates resampled Float32 samples and releases fixed-size frames
 * as soon as enough samples are available -- used to turn the
 * resampler's variable-length output back into the transport protocol's
 * fixed 160-sample (10ms @ 16kHz) cadence, without ever concatenating the
 * whole session's history (bounded to under one frame of carry-over). */
export class FixedSizeFramer {
  private carry: number[] = []
  private readonly frameSamples: number

  constructor(frameSamples: number) {
    if (frameSamples <= 0) throw new Error('FixedSizeFramer: frameSamples must be > 0')
    this.frameSamples = frameSamples
  }

  /** Pushes new samples in, returns zero or more complete frames. */
  push(samples: Float32Array): Float32Array[] {
    for (let i = 0; i < samples.length; i++) this.carry.push(samples[i]!)
    const frames: Float32Array[] = []
    while (this.carry.length >= this.frameSamples) {
      frames.push(Float32Array.from(this.carry.splice(0, this.frameSamples)))
    }
    return frames
  }

  get pendingSamples(): number {
    return this.carry.length
  }
}
