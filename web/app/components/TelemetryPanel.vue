<script setup lang="ts">
// Real measurements only -- unavailable fields render "—", never a
// fabricated placeholder (section 36).
const props = defineProps<{
  captureSampleRate: number | null
  captureChannels: number | null
  transportSampleRate: number
  transportChannels: number
  transportBits: number
  framesCaptured: number
  framesSent: number
  framesDropped: number
  queueDepth: number
  queueHighWaterMark: number
  rttMs: number | null
  modelMs: number | null
  serverQueueMs: number | null // not currently provided by the backend protocol -- always "—" until it is
  captureToPredictionMs: number | null
  reconnectCount: number
  sessionUptimeS: number
}>()

function fmt(v: number | null, digits = 0, unit = ''): string {
  if (v === null || Number.isNaN(v)) return '—'
  return `${v.toFixed(digits)}${unit}`
}
function fmtUptime(s: number): string {
  if (!s) return '—'
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

const cells = computed(() => [
  { label: 'CAPTURE FORMAT', value: props.captureSampleRate ? `${props.captureSampleRate}Hz / ${props.captureChannels ?? '—'}ch` : '—' },
  { label: 'TRANSPORT FORMAT', value: `${props.transportSampleRate}Hz / ${props.transportChannels}ch / PCM${props.transportBits}` },
  { label: 'FRAMES CAPTURED', value: String(props.framesCaptured) },
  { label: 'FRAMES SENT', value: String(props.framesSent) },
  { label: 'FRAMES DROPPED', value: String(props.framesDropped) },
  { label: 'QUEUE DEPTH / PEAK', value: `${props.queueDepth} / ${props.queueHighWaterMark}` },
  { label: 'WS RTT', value: fmt(props.rttMs, 1, ' ms') },
  { label: 'MODEL MS', value: fmt(props.modelMs, 1, ' ms') },
  { label: 'SERVER QUEUE MS', value: fmt(props.serverQueueMs, 1, ' ms') },
  { label: 'CAPTURE→PREDICT', value: fmt(props.captureToPredictionMs, 1, ' ms') },
  { label: 'RECONNECTS', value: String(props.reconnectCount) },
  { label: 'SESSION UPTIME', value: fmtUptime(props.sessionUptimeS) },
])
</script>

<template>
  <div class="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-3 text-[11px]">
    <div v-for="c in cells" :key="c.label">
      <div class="text-ink-faint tracking-wide text-[10px]">{{ c.label }}</div>
      <div class="font-mono text-ink mt-0.5">{{ c.value }}</div>
    </div>
  </div>
</template>
