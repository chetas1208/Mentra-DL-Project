<script setup lang="ts">
// Developer-only diagnostic drawer, default closed (section 50). No raw
// PCM dump -- only metadata and derived scalar stats.
import { getWaveformSnapshot } from '~/composables/useMicCapture'
import { computeSignalIntegrity } from '~/utils/signalIntegrity'
import { REQUESTED_CONSTRAINTS, type CaptureFormat } from '~/composables/useMicCapture'
import type { ModelCapabilities } from '~/utils/modelCapabilities'
import { PROTOCOL_VERSION } from '~/utils/mentraProtocol'

const props = defineProps<{
  captureFormat: CaptureFormat
  audioContextState: string | null
  isCapturing: boolean
  framesConverted: number
  wsState: string
  sessionId: string | null
  capabilities: ModelCapabilities | null
  queueDepth: number
  queueHighWaterMark: number
  getBufferedAmount: () => number | null
}>()

const open = ref(false)
const bufferedAmount = ref<number | null>(null)
const integrity = ref({ rms: 0, peak: 0, crestFactor: 0, clippingFraction: 0 })
let pollTimer: ReturnType<typeof setInterval> | null = null

function poll() {
  bufferedAmount.value = props.getBufferedAmount()
  integrity.value = computeSignalIntegrity(getWaveformSnapshot(1600)) // last 100ms
}

onMounted(() => {
  pollTimer = setInterval(poll, 500)
})
onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer)
})

const rows = computed(() => [
  ['deviceId', props.captureFormat.deviceId ?? '—'],
  ['audioContext.sampleRate', props.captureFormat.sampleRate !== null ? `${props.captureFormat.sampleRate} Hz` : '—'],
  ['audioContext.state', props.audioContextState ?? '—'],
  ['worklet', props.isCapturing ? 'active' : 'stopped'],
  ['resampler output frames', String(props.framesConverted)],
  ['websocket.state', props.wsState],
  ['websocket.bufferedAmount', bufferedAmount.value !== null ? `${bufferedAmount.value} bytes` : '—'],
  ['sessionId', props.sessionId ?? '—'],
  ['queue depth / peak', `${props.queueDepth} / ${props.queueHighWaterMark}`],
  ['protocol version', String(PROTOCOL_VERSION)],
])

const auditRows = computed(() => [
  ['requested echoCancellation', String(REQUESTED_CONSTRAINTS.echoCancellation)],
  ['actual echoCancellation', String(props.captureFormat.echoCancellation ?? '—')],
  ['requested noiseSuppression', String(REQUESTED_CONSTRAINTS.noiseSuppression)],
  ['actual noiseSuppression', String(props.captureFormat.noiseSuppression ?? '—')],
  ['requested autoGainControl', String(REQUESTED_CONSTRAINTS.autoGainControl)],
  ['actual autoGainControl', String(props.captureFormat.autoGainControl ?? '—')],
  ['frontend gain applied', '1.0 (no gain staging)'],
  ['frontend normalization applied', 'NO'],
])
</script>

<template>
  <div class="border-t border-panel-line">
    <button
      type="button"
      class="w-full flex items-center justify-between px-6 sm:px-10 py-2.5 text-[10px] text-ink-faint hover:text-ink-dim tracking-wide focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-wearer"
      :aria-expanded="open"
      @click="open = !open"
    >
      <span>DEBUG DRAWER</span>
      <span aria-hidden="true">{{ open ? '▲' : '▼' }}</span>
    </button>
    <div v-if="open" class="px-6 sm:px-10 pb-4 space-y-4 font-mono text-[10px]">
      <div>
        <div class="text-ink-faint mb-1">RUNTIME</div>
        <div class="grid grid-cols-2 gap-x-4 gap-y-1">
          <template v-for="[k, v] in rows" :key="k">
            <span class="text-ink-faint">{{ k }}</span><span class="text-ink">{{ v }}</span>
          </template>
        </div>
      </div>
      <div>
        <div class="text-ink-faint mb-1">AUDIO-INTEGRITY AUDIT (section 39)</div>
        <div class="grid grid-cols-2 gap-x-4 gap-y-1">
          <template v-for="[k, v] in auditRows" :key="k">
            <span class="text-ink-faint">{{ k }}</span><span class="text-ink">{{ v }}</span>
          </template>
        </div>
      </div>
      <div>
        <div class="text-ink-faint mb-1">RESEARCH SIGNAL (last ~100ms outgoing, debug only — not used for classification here)</div>
        <div class="grid grid-cols-2 gap-x-4 gap-y-1">
          <span class="text-ink-faint">rms</span><span class="text-ink">{{ integrity.rms.toFixed(4) }}</span>
          <span class="text-ink-faint">peak</span><span class="text-ink">{{ integrity.peak.toFixed(4) }}</span>
          <span class="text-ink-faint">crest factor</span><span class="text-ink">{{ Number.isFinite(integrity.crestFactor) ? integrity.crestFactor.toFixed(2) : '∞' }}</span>
          <span class="text-ink-faint">clipping fraction</span><span class="text-ink">{{ integrity.clippingFraction.toFixed(4) }}</span>
        </div>
      </div>
      <div>
        <div class="text-ink-faint mb-1">MODEL CAPABILITIES (raw, from backend)</div>
        <pre class="text-ink whitespace-pre-wrap break-words">{{ capabilities ? JSON.stringify(capabilities, null, 2) : 'NOT AVAILABLE — no capability payload received' }}</pre>
      </div>
    </div>
  </div>
</template>
