package com.mentra.wearerdetection.inference

/**
 * Backend abstraction so Eagle / sherpa-onnx / custom ONNX can be A/B tested
 * without touching the Mentra Bluetooth integration. See docs/ARCHITECTURE.md.
 */
interface WearerDetector {
    fun enroll(samples: ShortArray): EnrollmentResult

    fun resetEnrollment()

    fun process(samples: ShortArray): DetectionResult

    fun close()
}

enum class SpeechSource { SILENCE, WEARER, ENVIRONMENT, UNCERTAIN, OVERLAP }

data class EnrollmentResult(
    val success: Boolean,
    val durationMs: Long,
    val message: String? = null
)

data class DetectionResult(
    val state: SpeechSource,
    val wearerScore: Float,
    val confidence: Float,
    val inferenceMs: Float
)
