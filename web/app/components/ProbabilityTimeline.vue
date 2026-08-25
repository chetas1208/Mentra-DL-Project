<script setup lang="ts">
// Real backend inference only, last ~30s -- no fake idle values (section
// 35). Points are pushed only when a real DETECTION result arrives (see
// the `result` watcher below); an idle session simply shows an empty
// canvas, never a flat synthetic line.
import type { DetectionResult } from '~/composables/useMentraTransport'

const props = defineProps<{ result: DetectionResult | null }>()

const WINDOW_MS = 30_000
const canvasRef = ref<HTMLCanvasElement | null>(null)
interface Point { t: number; score: number; state: string }
let points: Point[] = []

function draw() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const dpr = window.devicePixelRatio || 1
  const w = canvas.clientWidth
  const h = canvas.clientHeight
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr
    canvas.height = h * dpr
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)

  const now = Date.now()
  points = points.filter((p) => now - p.t <= WINDOW_MS)
  if (points.length < 2) return

  // hysteresis band reference lines (0.35 / 0.5), documented in InferencePanel.vue
  ctx.strokeStyle = '#223039'
  ctx.lineWidth = 1
  for (const band of [0.35, 0.5]) {
    const y = h - band * h
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(w, y)
    ctx.stroke()
  }

  ctx.strokeStyle = '#5EEAD4'
  ctx.lineWidth = 1.5
  ctx.beginPath()
  for (const p of points) {
    const x = w - ((now - p.t) / WINDOW_MS) * w
    const y = h - Math.max(0, Math.min(1, p.score)) * h
    if (p === points[0]) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()
}

watch(() => props.result, (r) => {
  if (!r) return
  points.push({ t: Date.now(), score: r.wearerScore, state: r.state })
  requestAnimationFrame(draw)
})

let redrawTimer: ReturnType<typeof setInterval> | null = null
onMounted(() => {
  redrawTimer = setInterval(() => requestAnimationFrame(draw), 500) // keeps the window scrolling even between results
})
onBeforeUnmount(() => {
  if (redrawTimer) clearInterval(redrawTimer)
})
</script>

<template>
  <div>
    <div class="text-[10px] text-ink-faint tracking-wide mb-1.5">WEARER SIMILARITY — LAST 30s</div>
    <canvas ref="canvasRef" class="w-full h-16 rounded-md bg-panel-bg border border-panel-line" />
    <p v-if="!result && points.length === 0" class="text-[10px] text-ink-faint mt-1 font-sans">No inference yet.</p>
  </div>
</template>
