<script setup lang="ts">
import { float32ToPcm16LE } from '~/utils/mentraProtocol'
import { StreamingResampler } from '~/utils/streamingResampler'
import { normalizeInferenceState } from '~/utils/runtimeStatus'

// Row state used by SystemStatus.vue's (non-exported) prop contract --
// duplicated here rather than imported since the component doesn't
// export the type.
type RowState = 'ok' | 'warn' | 'bad' | 'neutral'

const {
  devices, selectedDeviceId, selectedDeviceLost,
  refreshDevices, selectDevice, attachDeviceChangeListener,
  mic, transport,
  sourceLabel, sourceClassification,
  availableModels, activeModelName, modelSelectionLocked,
  requiresEnrollment, supportsEnrollment,
  globalStatus, sessionUptimeS,
  isStarting, wasStopped, hasError,
} = useRuntimeState()

const {
  isCapturing, captureFormat, micLevel, framesCaptured, framesConverted,
  audioContextState, inputLost, transportSampleRate,
  start: startMic, stop: stopMic,
} = mic

const {
  wsState, lastResult, framesSent, lastRttMs, transcript, enrollmentStatus,
  capabilities, sessionId, rejectionReason,
  modelCatalog, selectedModel, activeModel, modelError,
  queueDepth, queueHighWaterMark, framesDropped, bytesDropped, reconnectCount,
  connect, disconnect, sendAudioFrame, sendEnrollmentAudio, setPlaybackModePreference,
  setSelectedModel, applyModelSelection,
  getBufferedAmount,
} = transport

const { playbackMode, pushWearerPcm, pushEnvironmentPcm, setPlaybackMode, close: closePlayback } = useGatedAudioPlayback()

// Local gain-node muting (instant, client-side) AND tell the server so it
// can stop sending the unselected stream entirely -- not just muted after
// the fact.
function onSetPlaybackMode(mode: 'both' | 'wearer' | 'environment' | 'muted') {
  setPlaybackMode(mode)
  setPlaybackModePreference(mode)
}

const errorMessage = ref<string | null>(null)
const isRecordingEnrollment = ref(false)
const ENROLLMENT_DURATION_S = 8

onMounted(() => {
  refreshDevices().catch(() => {})
  attachDeviceChangeListener()
  // The page is ready the moment it mounts -- STOPPED is reserved for
  // "session explicitly stopped", never the initial un-started state
  // (deriveGlobalStatus checks wasStopped before anything else).
  wasStopped.value = false
  // Connect eagerly (not starting the mic) purely to run the capabilities
  // handshake -- OptionalCalibration/RuntimeStatus/SystemStatus need real
  // capabilities to render honestly, and that only arrives over an open
  // WebSocket (STREAM_ACCEPTED payload). If the backend is unreachable
  // this settles to wsState 'CLOSED' harmlessly; startSession() retries.
  ensureTransportConnected().catch(() => {})
})

async function ensureTransportConnected() {
  if (wsState.value === 'CONNECTED' || wsState.value === 'CONNECTING') return
  await connect({ onWearerPcm: pushWearerPcm, onEnvironmentPcm: pushEnvironmentPcm })
}

function onSelectAnother() {
  selectedDeviceLost.value = false
}

/** Model choice is a session-scoped binding, not a page-global setting: the
 * click only records the request, and the backend's ack is what makes it
 * active. The selector is disabled while capturing, so this always runs on a
 * stopped session and can never hot-swap a model mid-stream. */
async function onSelectModel(modelId: string) {
  setSelectedModel(modelId)
  if (wsState.value !== 'CONNECTED') return
  await applyModelSelection()
}

/** Self-contained enrollment capture -- deliberately NOT reusing
 * useMicCapture()'s shared state (that composable's isCapturing/etc. are
 * keyed reactive state meant for the one main live session; running a
 * second independent capture through it would collide with that state).
 * Uses the same native-rate worklet + StreamingResampler pipeline as the
 * main capture path (see useMicCapture.ts) -- never assumes the browser
 * honors a forced AudioContext sample rate. */
async function recordEnrollment() {
  errorMessage.value = null
  try {
    await ensureTransportConnected()
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: selectedDeviceId.value ? { exact: selectedDeviceId.value } : undefined,
        channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false,
      },
    })
    const ctx = new AudioContext()
    await ctx.audioWorklet.addModule('/mentra-capture-worklet.js')
    const source = ctx.createMediaStreamSource(stream)
    const node = new AudioWorkletNode(ctx, 'mentra-capture-processor')
    const resampler = new StreamingResampler(ctx.sampleRate, 16000)
    const chunks: Uint8Array[] = []
    node.port.onmessage = (e: MessageEvent<{ samples: Float32Array; nativeSampleRate: number }>) => {
      const resampled = resampler.process(e.data.samples)
      if (resampled.length > 0) chunks.push(float32ToPcm16LE(resampled))
    }
    source.connect(node)

    isRecordingEnrollment.value = true
    await new Promise((resolve) => setTimeout(resolve, ENROLLMENT_DURATION_S * 1000))
    isRecordingEnrollment.value = false

    source.disconnect()
    node.disconnect()
    stream.getTracks().forEach((t) => t.stop())
    await ctx.close()

    const total = chunks.reduce((n, c) => n + c.length, 0)
    const merged = new Uint8Array(total)
    let offset = 0
    for (const c of chunks) {
      merged.set(c, offset)
      offset += c.length
    }
    sendEnrollmentAudio(merged)
  } catch (e) {
    isRecordingEnrollment.value = false
    errorMessage.value = e instanceof Error ? e.message : String(e)
  }
}

async function startSession() {
  errorMessage.value = null
  hasError.value = false
  wasStopped.value = false
  isStarting.value = true
  try {
    await ensureTransportConnected()
    // Confirm the selected model is really bound before any audio flows --
    // never start a stream against a model the backend hasn't acked.
    const bound = await applyModelSelection()
    if (!bound) {
      throw new Error(modelError.value ?? 'Selected model unavailable.')
    }
    await startMic(selectedDeviceId.value, (pcm16, captureTimestampNs) => {
      sendAudioFrame(pcm16, captureTimestampNs)
    })
    await refreshDevices() // labels populate once permission is granted
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e)
    stopMic()
    disconnect()
    hasError.value = true
  } finally {
    isStarting.value = false
  }
}

function stopSession() {
  stopMic()
  disconnect()
  closePlayback()
  wasStopped.value = true
}

const stateLabel = computed(() => normalizeInferenceState(lastResult.value?.state))

// --- SystemStatus (INPUT / AUDIO / BACKEND / MODEL) ----------------------
const inputRowState = computed<RowState>(() => {
  if (selectedDeviceLost.value) return 'bad'
  return sourceClassification.value === 'UNKNOWN AUDIO INPUT' ? 'neutral' : 'ok'
})
const audioRowState = computed<RowState>(() => {
  if (inputLost.value) return 'bad'
  return isCapturing.value ? 'ok' : 'neutral'
})
const backendRowState = computed<RowState>(() => {
  if (wsState.value === 'CONNECTED') return 'ok'
  if (wsState.value === 'CONNECTING' || wsState.value === 'RECONNECTING') return 'warn'
  if (wsState.value === 'ERROR') return 'bad'
  return 'neutral'
})
const modelRowState = computed<RowState>(() => {
  if (!capabilities.value) return 'neutral'
  return capabilities.value.ready ? 'ok' : 'warn'
})
const backendRowLabel = computed(() =>
  wsState.value === 'ERROR' && rejectionReason.value ? rejectionReason.value : wsState.value,
)
const modelRowLabel = computed(() => {
  if (!capabilities.value) return 'NOT AVAILABLE'
  // The product name when the backend's catalog gives us one, the raw id
  // otherwise -- still the backend's answer either way, never a local guess.
  const name = activeModelName.value ?? capabilities.value.modelId
  return `${name}${capabilities.value.ready ? '' : ' (not ready)'}`
})
</script>

<template>
  <main class="min-h-screen overflow-x-hidden bg-panel-bg text-ink font-mono flex items-center justify-center p-4 sm:p-8 motion-reduce:[&_*]:!transition-none motion-reduce:[&_*]:!animate-none">
    <div class="w-full max-w-2xl">
      <!-- wordmark + status -->
      <div class="flex flex-wrap items-end justify-between gap-x-4 gap-y-2 mb-6 px-1">
        <div>
          <h1 class="font-display text-2xl sm:text-3xl font-semibold tracking-tight text-ink">
            Mentra<span class="text-wearer">Wear</span>Net
          </h1>
          <p class="text-ink-dim text-xs mt-1 tracking-wide">WEARER-VS-ENVIRONMENT DETECTION · LIVE</p>
        </div>
        <div class="flex flex-col items-end gap-1.5 shrink-0">
          <RuntimeStatus
            :status="globalStatus"
            :source-classification="sourceClassification"
            :model-id="capabilities?.modelId ?? null"
            :model-version="capabilities?.modelVersion ?? null"
            :requires-enrollment="requiresEnrollment"
          />
          <NuxtLink to="/audio-debug" class="text-[10px] text-ink-faint hover:text-ink-dim underline decoration-dotted underline-offset-2 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-wearer rounded-sm">
            AUDIO INPUT DEBUG →
          </NuxtLink>
        </div>
      </div>

      <!-- one-line instruction, load-bearing (audio source honesty) -->
      <p class="mb-4 px-1 text-[11px] text-ink-dim font-sans">
        Select an input below, enroll your voice, then start.
      </p>

      <!-- instrument body -->
      <div class="instrument-panel-enter border border-panel-line rounded-xl bg-panel-surface overflow-hidden">
        <!-- subsystem status breakdown -->
        <div class="px-6 sm:px-10 py-4 border-b border-panel-line">
          <SystemStatus
            :input-label="sourceLabel || 'NONE SELECTED'"
            :input-state="inputRowState"
            :audio-label="isCapturing ? `CAPTURING ${captureFormat.sampleRate ?? '—'}Hz` : 'IDLE'"
            :audio-state="audioRowState"
            :backend-label="backendRowLabel"
            :backend-state="backendRowState"
            :model-label="modelRowLabel"
            :model-state="modelRowState"
          />
        </div>

        <!-- signature element: bipolar meter + rolling timeline -->
        <div class="px-6 sm:px-10 py-8 border-b border-panel-line space-y-4">
          <InferencePanel :wearer-score="lastResult?.wearerScore ?? null" :state="stateLabel" />
          <ProbabilityTimeline :result="lastResult" :model-id="activeModel" />
        </div>

        <!-- controls -->
        <div class="px-6 sm:px-10 py-5 border-b border-panel-line space-y-3">
          <AudioSourceSelector
            :devices="devices"
            :selected-device-id="selectedDeviceId"
            :disabled="isCapturing"
            :source-lost="selectedDeviceLost"
            :source-classification="sourceClassification"
            @select="selectDevice"
            @select-another="onSelectAnother"
          />

          <ModelSelector
            :models="availableModels"
            :selected-model="selectedModel"
            :active-model="activeModel"
            :disabled="modelSelectionLocked"
            :error="modelError"
            @select="onSelectModel"
          />

          <!-- real mic level, driven by actual RMS -->
          <div v-if="isCapturing" class="flex items-center gap-2">
            <span class="text-[11px] text-ink-faint w-10 shrink-0">LEVEL</span>
            <div class="flex-1 h-1.5 bg-panel-bg rounded-full overflow-hidden border border-panel-line">
              <div
                class="h-full bg-ink-dim transition-[width] duration-75"
                :style="{ width: `${Math.min(100, micLevel * 400)}%` }"
              />
            </div>
          </div>
          <InputWaveform :active="isCapturing" />

          <OptionalCalibration
            v-if="!isCapturing"
            :requires-enrollment="requiresEnrollment"
            :supports-enrollment="supportsEnrollment"
            :supports-passive-personalization="capabilities?.supportsPassivePersonalization ?? false"
            :enrollment-status="enrollmentStatus"
            :is-recording="isRecordingEnrollment"
            :duration-s="ENROLLMENT_DURATION_S"
            @record="recordEnrollment"
          />

          <SessionControls
            :is-capturing="isCapturing"
            :is-starting="isStarting"
            :disabled="selectedDeviceLost"
            :error-message="errorMessage"
            @start="startSession"
            @stop="stopSession"
          />
        </div>

        <!-- telemetry -->
        <div class="px-6 sm:px-10 py-4 border-b border-panel-line">
          <TelemetryPanel
            :capture-sample-rate="captureFormat.sampleRate"
            :capture-channels="captureFormat.channelCount"
            :transport-sample-rate="transportSampleRate"
            :transport-channels="1"
            :transport-bits="16"
            :frames-captured="framesCaptured"
            :frames-sent="framesSent"
            :frames-dropped="framesDropped"
            :queue-depth="queueDepth"
            :queue-high-water-mark="queueHighWaterMark"
            :rtt-ms="lastRttMs"
            :model-ms="lastResult?.inferenceMs ?? null"
            :server-queue-ms="null"
            :capture-to-prediction-ms="lastResult?.captureToPredictionMs ?? null"
            :reconnect-count="reconnectCount"
            :session-uptime-s="sessionUptimeS"
          />
          <p v-if="bytesDropped > 0" class="text-[10px] text-red-400 mt-2 font-mono">
            {{ bytesDropped }} BYTES DROPPED (backpressure, drop-oldest)
          </p>
        </div>

        <!-- gated audio playback + live transcript -->
        <div v-if="isCapturing" class="px-6 sm:px-10 py-4 border-b border-panel-line space-y-3">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <span class="text-[11px] text-ink-dim tracking-wide">GATED AUDIO PLAYBACK</span>
            <div class="grid grid-cols-2 sm:flex gap-1.5" role="group" aria-label="Playback mode">
              <button
                type="button"
                :aria-pressed="playbackMode === 'both'"
                class="min-h-[36px] text-[10px] border rounded px-2 py-1.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink-dim transition-colors font-mono tracking-wide"
                :class="playbackMode === 'both' ? 'border-ink-dim text-ink bg-panel-line/40' : 'border-panel-line text-ink-faint hover:border-ink-dim'"
                @click="onSetPlaybackMode('both')"
              >BOTH</button>
              <button
                type="button"
                :aria-pressed="playbackMode === 'wearer'"
                class="min-h-[36px] text-[10px] border rounded px-2 py-1.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-wearer transition-colors font-mono tracking-wide"
                :class="playbackMode === 'wearer' ? 'border-wearer text-wearer bg-wearer/10' : 'border-panel-line text-ink-faint hover:border-wearer/50'"
                @click="onSetPlaybackMode('wearer')"
              >HOST ONLY</button>
              <button
                type="button"
                :aria-pressed="playbackMode === 'environment'"
                class="min-h-[36px] text-[10px] border rounded px-2 py-1.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-environment transition-colors font-mono tracking-wide"
                :class="playbackMode === 'environment' ? 'border-environment text-environment bg-environment/10' : 'border-panel-line text-ink-faint hover:border-environment/50'"
                @click="onSetPlaybackMode('environment')"
              >SURROUND ONLY</button>
              <button
                type="button"
                :aria-pressed="playbackMode === 'muted'"
                class="min-h-[36px] text-[10px] border rounded px-2 py-1.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink-dim transition-colors font-mono tracking-wide"
                :class="playbackMode === 'muted' ? 'border-ink-dim text-ink-faint bg-panel-line/40' : 'border-panel-line text-ink-faint hover:border-ink-dim'"
                @click="onSetPlaybackMode('muted')"
              >MUTED</button>
            </div>
          </div>
          <div>
            <div class="text-[10px] text-ink-faint tracking-wide mb-1">TRANSCRIPT (ENVIRONMENT ASR)</div>
            <div class="text-xs text-ink bg-panel-bg border border-panel-line rounded-md px-3 py-2 min-h-[2.5rem] font-sans" role="log" aria-live="polite">
              {{ transcript || '—' }}
            </div>
          </div>
        </div>

        <DebugDrawer
          :capture-format="captureFormat"
          :audio-context-state="audioContextState"
          :is-capturing="isCapturing"
          :frames-converted="framesConverted"
          :ws-state="wsState"
          :session-id="sessionId"
          :capabilities="capabilities"
          :model-catalog="modelCatalog"
          :selected-model="selectedModel"
          :active-model="activeModel"
          :model-ms="lastResult?.inferenceMs ?? null"
          :queue-depth="queueDepth"
          :queue-high-water-mark="queueHighWaterMark"
          :get-buffered-amount="getBufferedAmount"
        />
      </div>
    </div>
  </main>
</template>

<style scoped>
/* One deliberate load moment, not scattered motion -- the instrument panel
   settles into place once. prefers-reduced-motion is already handled
   globally (see the motion-reduce:[&_*]:!animate-none on <main>, which
   applies animation:none !important to every descendant regardless of
   where the animation was declared). */
.instrument-panel-enter {
  animation: panel-settle 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
}
@keyframes panel-settle {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
