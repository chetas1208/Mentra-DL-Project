// Research-signal telemetry (spec section 38/39) -- computed from the
// ACTUAL outgoing PCM, shown only in the DEBUG/RESEARCH drawer, never
// used for classification here (the backend detector does that).
// These are exactly the physical-acoustic cues the future
// enrollment-free wearer detector may depend on, which is also why the
// capture pipeline must never normalize/AGC/EQ them away before this
// point (see mentraProtocol.ts and useMicCapture.ts's "no normalization"
// comments).

export interface SignalIntegrityStats {
  rms: number
  peak: number
  crestFactor: number // peak / rms, Infinity if rms === 0 and peak > 0
  clippingFraction: number // fraction of samples at or beyond +/-1.0
}

export function computeSignalIntegrity(samples: Float32Array): SignalIntegrityStats {
  if (samples.length === 0) {
    return { rms: 0, peak: 0, crestFactor: 0, clippingFraction: 0 }
  }
  let sumSquares = 0
  let peak = 0
  let clipped = 0
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i]!
    const abs = Math.abs(v)
    sumSquares += v * v
    if (abs > peak) peak = abs
    if (abs >= 1) clipped++
  }
  const rms = Math.sqrt(sumSquares / samples.length)
  const crestFactor = rms === 0 ? (peak > 0 ? Infinity : 0) : peak / rms
  return { rms, peak, crestFactor, clippingFraction: clipped / samples.length }
}
