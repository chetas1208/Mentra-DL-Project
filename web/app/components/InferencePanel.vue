<script setup lang="ts">
// Deliberately shows only what the backend actually sends: a single
// wearer-similarity score (cosine similarity against the enrolled
// embedding, NOT a calibrated 0..1 probability that a second
// "environment probability" would sum against) and the backend's own
// hysteresis-banded state label. Section 25/56: never invent a second
// number the backend doesn't provide.
const props = defineProps<{
  wearerScore: number | null
  state: string
}>()

// Centered on the midpoint of the backend's hysteresis band (0.35/0.5,
// see mentra/audio/consumer.py's default thresholds) rather than raw
// score 0 -- that midpoint IS the product's real decision boundary. This
// constant mirrors the backend's own default and is documented here, not
// silently duplicated -- if the backend's thresholds ever become
// configurable/reported, this should read them instead of a constant.
const HYSTERESIS_MIDPOINT = 0.425
const HYSTERESIS_HALF_RANGE = 0.4

const needlePosition = computed(() => {
  const score = props.wearerScore ?? HYSTERESIS_MIDPOINT
  return Math.max(-1, Math.min(1, (score - HYSTERESIS_MIDPOINT) / HYSTERESIS_HALF_RANGE))
})
const isWearer = computed(() => props.state === 'WEARER')
const isEnvironment = computed(() => props.state === 'ENVIRONMENT')
</script>

<template>
  <div>
    <div class="flex justify-between text-[11px] text-ink-dim tracking-wide mb-3">
      <span :class="{ 'text-environment': isEnvironment }">← ENVIRONMENT</span>
      <span class="font-display text-lg font-semibold" :class="{ 'text-wearer': isWearer, 'text-environment': isEnvironment, 'text-ink-faint': !isWearer && !isEnvironment }" role="status" aria-live="polite">
        {{ state }}
      </span>
      <span :class="{ 'text-wearer': isWearer }">WEARER →</span>
    </div>

    <div class="relative h-3 rounded-full bg-panel-bg border border-panel-line overflow-hidden">
      <div class="absolute left-1/2 top-0 bottom-0 w-px bg-ink-faint" />
      <div
        class="absolute top-0 bottom-0 transition-all duration-200 ease-out"
        :class="needlePosition >= 0 ? 'left-1/2 bg-wearer' : 'right-1/2 bg-environment'"
        :style="{ width: `${Math.abs(needlePosition) * 50}%` }"
      />
      <div
        class="absolute top-1/2 -translate-y-1/2 w-1 h-5 rounded-full transition-all duration-200 ease-out shadow-[0_0_8px_rgba(0,0,0,0.4)]"
        :class="needlePosition >= 0 ? 'bg-wearer' : 'bg-environment'"
        :style="{ left: `calc(${50 + needlePosition * 50}% - 2px)` }"
      />
    </div>

    <div class="flex justify-between text-[11px] text-ink-faint mt-2 font-mono">
      <span>{{ wearerScore !== null ? wearerScore.toFixed(3) : '—' }}</span>
      <span>wearer similarity (not a full probability distribution)</span>
    </div>
  </div>
</template>
