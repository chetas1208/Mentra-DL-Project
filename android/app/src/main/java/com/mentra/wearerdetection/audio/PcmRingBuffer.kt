package com.mentra.wearerdetection.audio

/**
 * Fixed-capacity ring buffer for 16-bit PCM samples. Bridges Mentra's Bluetooth
 * callback frame size to whatever fixed frame length a model backend requires.
 * Not thread-safe by itself — caller must synchronize if write/read happen on
 * different threads (Bluetooth callback vs inference executor).
 */
class PcmRingBuffer(capacitySamples: Int) {
    private val buffer = ShortArray(capacitySamples)
    private var writeIndex = 0
    private var available = 0
    private var totalWritten = 0L
    private var totalRead = 0L
    private var totalDropped = 0L

    val capacity: Int get() = buffer.size
    val fill: Int get() = available
    val samplesReceived: Long get() = totalWritten
    val samplesProcessed: Long get() = totalRead
    val samplesDropped: Long get() = totalDropped

    fun write(samples: ShortArray) {
        for (s in samples) {
            buffer[writeIndex] = s
            writeIndex = (writeIndex + 1) % buffer.size
            if (available < buffer.size) {
                available++
            } else {
                totalDropped++
            }
        }
        totalWritten += samples.size
    }

    /** Reads exactly [frameSize] samples, oldest first. Returns null if not enough buffered yet. */
    fun readFrame(frameSize: Int): ShortArray? {
        if (available < frameSize) return null
        val readStart = (writeIndex - available + buffer.size) % buffer.size
        val out = ShortArray(frameSize)
        for (i in 0 until frameSize) {
            out[i] = buffer[(readStart + i) % buffer.size]
        }
        available -= frameSize
        totalRead += frameSize
        return out
    }
}
