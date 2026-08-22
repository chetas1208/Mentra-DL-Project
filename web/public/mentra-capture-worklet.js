// AudioWorklet processor -- runs on the dedicated audio rendering thread,
// not the main thread, so mic capture doesn't stall on UI work (same
// "the callback must do almost nothing" principle as onMicPcm() in the
// Kotlin interfaces this whole project is built around).
//
// Web Audio delivers audio in fixed 128-sample render quanta regardless of
// sample rate. This buffers those into 160-sample (10ms @ 16kHz) frames
// matching the backend's expected cadence (mentra/audio/consumer.py /
// scripts/mentra/live_replay_demo.py both assume 10ms frames) before
// posting to the main thread.

const FRAME_SAMPLES = 160 // 10ms @ 16kHz

class MentraCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this._buffer = new Float32Array(FRAME_SAMPLES)
    this._writeIndex = 0
  }

  process(inputs) {
    const input = inputs[0]
    if (!input || input.length === 0) return true
    const channelData = input[0] // mono -- AudioContext configured for 1 channel
    if (!channelData) return true

    for (let i = 0; i < channelData.length; i++) {
      this._buffer[this._writeIndex++] = channelData[i]
      if (this._writeIndex >= FRAME_SAMPLES) {
        // transfer ownership of the buffer to avoid a copy; allocate a fresh one
        this.port.postMessage(this._buffer, [this._buffer.buffer])
        this._buffer = new Float32Array(FRAME_SAMPLES)
        this._writeIndex = 0
      }
    }
    return true // keep the processor alive
  }
}

registerProcessor('mentra-capture-processor', MentraCaptureProcessor)
