// AudioWorklet processor -- runs on the dedicated audio rendering thread,
// not the main thread, so mic capture doesn't stall on UI work (same
// "the callback must do almost nothing" principle as onMicPcm() in the
// Kotlin interfaces this whole project is built around).
//
// Web Audio delivers audio in fixed 128-sample render quanta at whatever
// rate the AudioContext actually runs at (commonly 48000Hz, sometimes
// 44100Hz -- browsers do NOT reliably honor a requested 16000Hz
// AudioContext sample rate, verified in this project's own testing). This
// processor does NOT resample or assume any particular rate -- it only
// buffers native-rate render quanta into ~10ms blocks and hands them to
// the main thread untouched. Resampling to 16kHz happens on the main
// thread (see useStreamingResampler.ts / useMicCapture.ts) -- keeping
// that math out of the audio-rendering thread entirely, per this
// project's rule that the audio callback must do almost nothing:
// no networking, no DOM access, no JSON, no large allocations, no
// resampling math here.
//
// `sampleRate` is a global provided by the AudioWorkletGlobalScope, set
// to the real AudioContext sample rate -- used only to size the ~10ms
// buffer, not to convert anything.

class MentraCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this._blockSamples = Math.max(1, Math.round(sampleRate * 0.01)) // ~10ms at the ACTUAL native rate
    this._buffer = new Float32Array(this._blockSamples)
    this._writeIndex = 0
  }

  process(inputs) {
    const input = inputs[0]
    if (!input || input.length === 0) return true

    // Deterministic downmix if the capture unexpectedly delivers more
    // than one channel: average all channels sample-by-sample rather
    // than silently taking channel 0 only. Current deployment requests
    // channelCount:1 so this path is not normally exercised, but it must
    // not silently drop channels if the browser ever hands back more.
    const channelCount = input.length
    const channelData = input[0]
    if (!channelData) return true

    for (let i = 0; i < channelData.length; i++) {
      let sample = channelData[i]
      if (channelCount > 1) {
        let sum = sample
        for (let c = 1; c < channelCount; c++) sum += input[c][i]
        sample = sum / channelCount
      }
      this._buffer[this._writeIndex++] = sample
      if (this._writeIndex >= this._blockSamples) {
        // transfer ownership of the buffer to avoid a copy; allocate a fresh one
        this.port.postMessage({ samples: this._buffer, nativeSampleRate: sampleRate }, [this._buffer.buffer])
        this._buffer = new Float32Array(this._blockSamples)
        this._writeIndex = 0
      }
    }
    return true // keep the processor alive
  }
}

registerProcessor('mentra-capture-processor', MentraCaptureProcessor)
