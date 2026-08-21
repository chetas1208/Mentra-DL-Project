package com.mentra.wearerdetection.audio

/**
 * VAD abstraction. Implementations to benchmark: Mentra-reported speaking_status,
 * Silero VAD, model-native VAD. Do NOT treat any of these as wearer-vs-environment
 * signal on their own — see docs/ARCHITECTURE.md and sprint spec section 10.
 */
interface VoiceActivityDetector {
    fun process(samples: ShortArray): VadResult
}

data class VadResult(
    val speechPresent: Boolean,
    val probability: Float
)
