<script setup lang="ts">
import { float32ToPcm16LE } from '~/utils/mentraProtocol'

const { connectionState, lastResult, framesSent, lastRttMs, transcript, enrollmentStatus, connect, disconnect, sendAudioFrame, sendEnrollmentAudio, setPlaybackModePreference } = useMentraTransport()
const { isCapturing, inputDevices, actualSampleRate, micLevel, listInputDevices, start, stop } = useMicCapture()
const { playbackMode, pushWearerPcm, pushEnvironmentPcm, setPlaybackMode, close: closePlayback } = useGatedAudioPlayback()

// Local gain-node muting (instant, client-side) AND tell the server so it
// can stop sending the unselected stream entirely -- not just muted after
// the fact.
function onSetPlaybackMode(mode: 'both' | 'wearer' | 'environment' | 'muted') {
  setPlaybackMode(mode)
  setPlaybackModePreference(mode)
}

const selectedDeviceId = ref<string | undefined>(undefined)
const errorMessage = ref<string | null>(null)
const isRecordingEnrollment = ref(false)
const ENROLLMENT_DURATION_S = 8

onMounted(() => {
  listInputDevices().catch(() => {})
})

/** Self-contained enrollment capture -- deliberately NOT reusing
 * useMicCapture()'s shared state (that composable's isCapturing/etc. are
 * keyed reactive state meant for the one main live session; running a
 * second independent capture through it would collide with that state). */
async function recordEnrollment() {
  errorMessage.value = null
  try {
    if (connectionState.value !== 'CONNECTED') {
      await connect({ onWearerPcm: pushWearerPcm, onEnvironmentPcm: pushEnvironmentPcm })
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: selectedDeviceId.value ? { exact: selectedDeviceId.value } : undefined,
        channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false,
      },
    })
    const ctx = new AudioContext({ sampleRate: 16000 })
    await ctx.audioWorklet.addModule('/mentra-capture-worklet.js')
    const source = ctx.createMediaStreamSource(stream)
    const node = new AudioWorkletNode(ctx, 'mentra-capture-processor')
    const chunks: Uint8Array[] = []
    node.port.onmessage = (e: MessageEvent<Float32Array>) => chunks.push(float32ToPcm16LE(e.data))
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

const isStarting = ref(false)

async function startSession() {
  errorMessage.value = null
  isStarting.value = true
  try {
    await connect({
      onWearerPcm: pushWearerPcm,
      onEnvironmentPcm: pushEnvironmentPcm,
    })
    await start(selectedDeviceId.value, (pcm16, captureTimestampNs) => {
      sendAudioFrame(pcm16, captureTimestampNs)
    })
    await listInputDevices() // labels populate once permission is granted
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e)
    stop()
    disconnect()
  } finally {
    isStarting.value = false
  }
}

function stopSession() {
  stop()
  disconnect()
  closePlayback()
}

// Bipolar needle position, -1 (environment) .. 0 (ambiguous) .. +1 (wearer).
// Centered on the midpoint of the backend's hysteresis band (0.35/0.5,
// see mentra/audio/consumer.py's default thresholds) rather than raw
// score 0 -- that midpoint IS the product's real decision boundary.
const needlePosition = computed(() => {
  const score = lastResult.value?.wearerScore ?? 0.425
  return Math.max(-1, Math.min(1, (score - 0.425) / 0.4))
})

const stateLabel = computed(() => lastResult.value?.state ?? 'SILENCE')
const isWearer = computed(() => stateLabel.value === 'WEARER')
const isEnvironment = computed(() => stateLabel.value === 'ENVIRONMENT')
</script>

<template>
  <main class="min-h-screen overflow-x-hidden bg-panel-bg text-ink font-mono flex items-center justify-center p-4 sm:p-8 motion-reduce:[&_*]:!transition-none motion-reduce:[&_*]:!animate-none">
    <div class="w-full max-w-2xl">
      <!-- wordmark + status LEDs -->
      <div class="flex flex-wrap items-end justify-between gap-x-4 gap-y-2 mb-6 px-1">
        <div>
          <h1 class="font-display text-2xl sm:text-3xl font-semibold tracking-tight text-ink">
            Mentra<span class="text-wearer">Wear</span>Net
          </h1>
          <p class="text-ink-dim text-xs mt-1 tracking-wide">WEARER-VS-ENVIRONMENT DETECTION · LIVE</p>
        </div>
        <div class="flex flex-col items-end gap-1.5 shrink-0">
          <div class="flex items-center gap-2 text-[11px] text-ink-dim" role="status" aria-live="polite">
            <span
              aria-hidden="true"
              class="w-2 h-2 rounded-full transition-colors"
              :class="{
                'bg-wearer shadow-[0_0_6px] shadow-wearer': connectionState === 'CONNECTED',
                'bg-environment animate-pulse': connectionState === 'CONNECTING',
                'bg-ink-faint': connectionState === 'DISCONNECTED',
                'bg-red-500': connectionState === 'REJECTED' || connectionState === 'DEAD',
              }"
            />
            <span class="tracking-wide">{{ connectionState === 'DISCONNECTED' ? 'READY · SESSION NOT STARTED' : connectionState }}</span>
          </div>
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
        <!-- signature element: bipolar meter -->
        <div class="px-6 sm:px-10 py-8 border-b border-panel-line">
          <div class="flex justify-between text-[11px] text-ink-dim tracking-wide mb-3">
            <span :class="{ 'text-environment': isEnvironment }">← ENVIRONMENT</span>
            <span class="font-display text-lg font-semibold" :class="{ 'text-wearer': isWearer, 'text-environment': isEnvironment, 'text-ink-faint': !isWearer && !isEnvironment }" role="status" aria-live="polite">
              {{ stateLabel }}
            </span>
            <span :class="{ 'text-wearer': isWearer }">WEARER →</span>
          </div>

          <div class="relative h-3 rounded-full bg-panel-bg border border-panel-line overflow-hidden">
            <!-- center reference tick -->
            <div class="absolute left-1/2 top-0 bottom-0 w-px bg-ink-faint" />
            <!-- fill grows from center toward whichever side is active -->
            <div
              class="absolute top-0 bottom-0 transition-all duration-200 ease-out"
              :class="needlePosition >= 0 ? 'left-1/2 bg-wearer' : 'right-1/2 bg-environment'"
              :style="{ width: `${Math.abs(needlePosition) * 50}%` }"
            />
            <!-- needle -->
            <div
              class="absolute top-1/2 -translate-y-1/2 w-1 h-5 rounded-full transition-all duration-200 ease-out shadow-[0_0_8px_rgba(0,0,0,0.4)]"
              :class="needlePosition >= 0 ? 'bg-wearer' : 'bg-environment'"
              :style="{ left: `calc(${50 + needlePosition * 50}% - 2px)` }"
            />
          </div>

          <div class="flex justify-between text-[11px] text-ink-faint mt-2 font-mono">
            <span>{{ (lastResult?.wearerScore ?? 0).toFixed(3) }}</span>
            <span>similarity</span>
          </div>
        </div>

        <!-- controls -->
        <div class="px-6 sm:px-10 py-5 border-b border-panel-line space-y-3">
          <div class="flex items-center justify-between gap-3">
            <label for="input-device" class="text-[11px] text-ink-dim tracking-wide shrink-0">INPUT</label>
            <select
              id="input-device"
              v-model="selectedDeviceId"
              :disabled="isCapturing"
              class="bg-panel-bg border border-panel-line rounded-md px-2.5 py-2 text-xs text-ink w-full max-w-[70%] focus:outline-none focus-visible:ring-2 focus-visible:ring-wearer focus-visible:ring-offset-2 focus-visible:ring-offset-panel-surface disabled:opacity-50"
            >
              <option :value="undefined">Default microphone</option>
              <option v-for="d in inputDevices" :key="d.deviceId" :value="d.deviceId">
                {{ d.label || d.deviceId }}
              </option>
            </select>
          </div>

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

          <div v-if="!isCapturing" class="flex items-center justify-between text-[11px]" role="status" aria-live="polite">
            <span class="text-ink-dim tracking-wide">ENROLLMENT</span>
            <span
              class="font-mono"
              :class="{ 'text-amber-400': enrollmentStatus === 'PLACEHOLDER', 'text-environment': enrollmentStatus === 'ENROLLING', 'text-wearer': enrollmentStatus === 'REAL' }"
            >
              {{ enrollmentStatus === 'PLACEHOLDER' ? 'PLACEHOLDER (not you)' : enrollmentStatus === 'ENROLLING' ? `RECORDING ${ENROLLMENT_DURATION_S}s...` : 'REAL (your voice)' }}
            </span>
          </div>
          <button
            v-if="!isCapturing"
            :disabled="isRecordingEnrollment"
            class="w-full min-h-[44px] border border-panel-line text-ink-dim hover:border-environment hover:text-environment focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-environment focus-visible:ring-offset-2 focus-visible:ring-offset-panel-surface transition-colors rounded-md py-2.5 text-xs font-semibold tracking-wide disabled:opacity-50 disabled:cursor-not-allowed"
            @click="recordEnrollment"
          >
            {{ isRecordingEnrollment ? `RECORDING... ${ENROLLMENT_DURATION_S}s` : `ENROLL MY VOICE (${ENROLLMENT_DURATION_S}s)` }}
          </button>

          <button
            v-if="!isCapturing"
            :disabled="isStarting"
            class="w-full min-h-[44px] border border-wearer/40 text-wearer hover:bg-wearer hover:text-panel-bg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-wearer focus-visible:ring-offset-2 focus-visible:ring-offset-panel-surface transition-colors rounded-md py-2.5 text-xs font-semibold tracking-wide disabled:opacity-50 disabled:cursor-not-allowed"
            @click="startSession"
          >
            {{ isStarting ? 'CONNECTING…' : 'START SESSION' }}
          </button>
          <button
            v-else
            class="w-full min-h-[44px] border border-red-500/40 text-red-400 hover:bg-red-500 hover:text-panel-bg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400 focus-visible:ring-offset-2 focus-visible:ring-offset-panel-surface transition-colors rounded-md py-2.5 text-xs font-semibold tracking-wide"
            @click="stopSession"
          >
            STOP SESSION
          </button>

          <p v-if="errorMessage" role="alert" class="text-xs text-red-400 bg-red-950/40 border border-red-900 rounded-md px-3 py-2 font-sans">
            {{ errorMessage }}
          </p>
        </div>

        <!-- telemetry strip -->
        <div class="grid grid-cols-2 sm:grid-cols-4 divide-x divide-panel-line border-b border-panel-line">
          <div class="px-4 py-3">
            <div class="text-[10px] text-ink-faint tracking-wide">CONTEXT</div>
            <div class="text-sm mt-0.5">{{ (lastResult?.contextMs ?? 0).toFixed(0) }}<span class="text-ink-faint text-[10px]"> ms</span></div>
          </div>
          <div class="px-4 py-3">
            <div class="text-[10px] text-ink-faint tracking-wide">INFERENCE</div>
            <div class="text-sm mt-0.5">{{ (lastResult?.inferenceMs ?? 0).toFixed(1) }}<span class="text-ink-faint text-[10px]"> ms</span></div>
          </div>
          <div class="px-4 py-3">
            <div class="text-[10px] text-ink-faint tracking-wide">CAPTURE→PREDICT</div>
            <div class="text-sm mt-0.5">{{ (lastResult?.captureToPredictionMs ?? 0).toFixed(1) }}<span class="text-ink-faint text-[10px]"> ms</span></div>
          </div>
          <div class="px-4 py-3">
            <div class="text-[10px] text-ink-faint tracking-wide">HEARTBEAT RTT</div>
            <div class="text-sm mt-0.5">{{ lastRttMs !== null ? lastRttMs.toFixed(1) : '—' }}<span class="text-ink-faint text-[10px]"> ms</span></div>
          </div>
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

        <div class="px-6 sm:px-10 py-2.5 flex justify-between text-[10px] text-ink-faint">
          <span>FRAMES SENT: {{ framesSent }}</span>
          <span>MIC RATE: {{ actualSampleRate ?? '—' }} Hz</span>
        </div>
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
