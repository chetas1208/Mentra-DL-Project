<script setup lang="ts">
import type { GlobalStatus } from '~/utils/runtimeStatus'
import type { SourceClassification } from '~/utils/sourceClassification'

const props = defineProps<{
  status: GlobalStatus
  sourceClassification: SourceClassification
  modelId: string | null
  modelVersion: string | null
  requiresEnrollment: boolean | null
}>()

const statusColorClass = computed(() => {
  if (props.status === 'LIVE') return 'bg-wearer shadow-[0_0_6px] shadow-wearer'
  if (props.status === 'CONNECTING BACKEND' || props.status === 'INITIALIZING AUDIO') return 'bg-environment animate-pulse'
  if (['INPUT LOST', 'BACKEND UNAVAILABLE', 'MODEL UNAVAILABLE', 'ERROR'].includes(props.status)) return 'bg-red-500'
  if (props.status === 'CALIBRATION REQUIRED') return 'bg-environment'
  return 'bg-ink-faint'
})

const calibrationLabel = computed(() => {
  if (props.requiresEnrollment === null) return null
  return props.requiresEnrollment
    ? 'VOICE CALIBRATION — REQUIRED BY CURRENT MODEL'
    : 'VOICE CALIBRATION — NOT REQUIRED'
})
</script>

<template>
  <div class="flex flex-col gap-1.5">
    <div class="flex items-center gap-2 text-[11px] text-ink-dim" role="status" aria-live="polite">
      <span aria-hidden="true" class="w-2 h-2 rounded-full transition-colors" :class="statusColorClass" />
      <span class="tracking-wide font-semibold text-ink">{{ status }}</span>
    </div>
    <div class="flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-ink-faint font-mono">
      <span>SOURCE: <span class="text-ink-dim">{{ sourceClassification }}</span></span>
      <span>MODEL: <span class="text-ink-dim">{{ modelId ? `${modelId}${modelVersion ? ' ' + modelVersion : ''}` : '—' }}</span></span>
      <span v-if="calibrationLabel">{{ calibrationLabel }}</span>
    </div>
  </div>
</template>
