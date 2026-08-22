<script setup lang="ts">
const { connectionState, lastResult, framesSent, lastRttMs, connect, disconnect, sendAudioFrame } = useMentraTransport()
const { isCapturing, inputDevices, actualSampleRate, micLevel, listInputDevices, start, stop } = useMicCapture()

const selectedDeviceId = ref<string | undefined>(undefined)
const errorMessage = ref<string | null>(null)

onMounted(() => {
  listInputDevices().catch(() => {})
})

async function startSession() {
  errorMessage.value = null
  try {
    await connect()
    await start(selectedDeviceId.value, (pcm16, captureTimestampNs) => {
      sendAudioFrame(pcm16, captureTimestampNs)
    })
    await listInputDevices() // labels populate once permission is granted
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e)
    stop()
    disconnect()
  }
}

function stopSession() {
  stop()
  disconnect()
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
  <div class="min-h-screen bg-panel-bg text-ink font-mono flex items-center justify-center p-4 sm:p-8">
    <div class="w-full max-w-2xl">
      <!-- wordmark + status LEDs -->
      <div class="flex items-end justify-between mb-6 px-1">
        <div>
          <h1 class="font-display text-2xl sm:text-3xl font-semibold tracking-tight text-ink">
            Mentra<span class="text-wearer">Wear</span>Net
          </h1>
          <p class="text-ink-dim text-xs mt-1 tracking-wide">WEARER-VS-ENVIRONMENT DETECTION · LIVE</p>
        </div>
        <div class="flex items-center gap-2 text-[11px] text-ink-dim pb-1">
          <span
            class="w-2 h-2 rounded-full transition-colors"
            :class="{
              'bg-wearer shadow-[0_0_6px] shadow-wearer': connectionState === 'CONNECTED',
              'bg-environment animate-pulse': connectionState === 'CONNECTING',
              'bg-ink-faint': connectionState === 'DISCONNECTED',
              'bg-red-500': connectionState === 'REJECTED' || connectionState === 'DEAD',
            }"
          />
          <span class="tracking-wide">{{ connectionState }}</span>
        </div>
      </div>

      <!-- instrument body -->
      <div class="border border-panel-line rounded-xl bg-panel-surface overflow-hidden">
        <!-- signature element: bipolar meter -->
        <div class="px-6 sm:px-10 py-8 border-b border-panel-line">
          <div class="flex justify-between text-[11px] text-ink-dim tracking-wide mb-3">
            <span :class="{ 'text-environment': isEnvironment }">← ENVIRONMENT</span>
            <span class="font-display text-lg font-semibold" :class="{ 'text-wearer': isWearer, 'text-environment': isEnvironment, 'text-ink-faint': !isWearer && !isEnvironment }">
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
            <label class="text-[11px] text-ink-dim tracking-wide shrink-0">INPUT</label>
            <select
              v-model="selectedDeviceId"
              :disabled="isCapturing"
              class="bg-panel-bg border border-panel-line rounded-md px-2.5 py-1.5 text-xs text-ink w-full max-w-[70%] focus:outline-none focus:ring-1 focus:ring-wearer disabled:opacity-50"
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

          <button
            v-if="!isCapturing"
            class="w-full border border-wearer/40 text-wearer hover:bg-wearer hover:text-panel-bg transition-colors rounded-md py-2 text-xs font-semibold tracking-wide"
            @click="startSession"
          >
            START SESSION
          </button>
          <button
            v-else
            class="w-full border border-red-500/40 text-red-400 hover:bg-red-500 hover:text-panel-bg transition-colors rounded-md py-2 text-xs font-semibold tracking-wide"
            @click="stopSession"
          >
            STOP SESSION
          </button>

          <p v-if="errorMessage" class="text-xs text-red-400 bg-red-950/40 border border-red-900 rounded-md px-3 py-2 font-sans">
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

        <div class="px-6 sm:px-10 py-2.5 flex justify-between text-[10px] text-ink-faint">
          <span>FRAMES SENT: {{ framesSent }}</span>
          <span>MIC RATE: {{ actualSampleRate ?? '—' }} Hz</span>
        </div>
      </div>

      <p class="text-[11px] text-ink-faint mt-4 px-1 font-sans leading-relaxed">
        Enrollment is currently a placeholder (LibriSpeech clip) on the backend —
        real wearer enrollment isn't wired to this UI yet.
      </p>
    </div>
  </div>
</template>
