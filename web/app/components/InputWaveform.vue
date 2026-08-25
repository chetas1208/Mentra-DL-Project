<script setup lang="ts">
// Real outgoing PCM only, no synthetic animation (section 34). Reads from
// useMicCapture's plain (non-reactive) ring buffer at a fixed ~15fps --
// deliberately NOT driven by Vue reactivity or the audio callback rate
// (section 46).
import { getWaveformSnapshot } from '~/composables/useMicCapture'

const props = defineProps<{ active: boolean }>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let rafHandle: number | null = null
let intervalHandle: ReturnType<typeof setInterval> | null = null
const FPS = 15

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

  if (!props.active) return
  const samples = getWaveformSnapshot(4800) // ~0.3s @ 16kHz, enough to look live without redrawing megabytes
  if (samples.length === 0) return

  ctx.strokeStyle = '#5EEAD4'
  ctx.lineWidth = 1
  ctx.beginPath()
  const step = samples.length / w
  for (let x = 0; x < w; x++) {
    const idx = Math.floor(x * step)
    const v = samples[idx] ?? 0
    const y = h / 2 - v * (h / 2 - 2)
    if (x === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()
}

function loop() {
  intervalHandle = setInterval(() => {
    rafHandle = requestAnimationFrame(draw)
  }, 1000 / FPS)
}

onMounted(loop)
onBeforeUnmount(() => {
  if (intervalHandle) clearInterval(intervalHandle)
  if (rafHandle) cancelAnimationFrame(rafHandle)
})
</script>

<template>
  <div>
    <div class="text-[10px] text-ink-faint tracking-wide mb-1.5">INPUT WAVEFORM (LIVE OUTGOING PCM)</div>
    <canvas ref="canvasRef" class="w-full h-16 rounded-md bg-panel-bg border border-panel-line" />
    <p v-if="!active" class="text-[10px] text-ink-faint mt-1 font-sans">Idle — no audio flowing.</p>
  </div>
</template>
