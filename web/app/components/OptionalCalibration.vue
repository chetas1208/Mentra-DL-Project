<script setup lang="ts">
// Enrollment UX driven entirely by the backend's capability payload
// (spec: ENROLLMENT UX, cases A-D). Nothing here is hardcoded as a
// product requirement -- if a future model reports
// requiresEnrollment=false / supportsEnrollmentFreeDetection=true, this
// component naturally collapses to nothing blocking the primary session
// path, with no code change needed here.
const props = defineProps<{
  requiresEnrollment: boolean | null // null = capability not yet known (not connected)
  supportsEnrollment: boolean | null
  supportsPassivePersonalization: boolean
  enrollmentStatus: 'PLACEHOLDER' | 'ENROLLING' | 'REAL'
  isRecording: boolean
  durationS: number
}>()
const emit = defineEmits<{ record: [] }>()

// Case A: required. Case B: supported but optional. Case C: enrollment
// not supported/not required -> nothing rendered, session starts without
// it. `null` (capability unknown, e.g. not connected yet) also renders
// nothing -- never guess a requirement before the backend has said so.
const mode = computed<'required' | 'optional' | 'none'>(() => {
  if (props.requiresEnrollment === true) return 'required'
  if (props.supportsEnrollment === true) return 'optional'
  return 'none'
})
</script>

<template>
  <div v-if="mode !== 'none'" class="space-y-2">
    <div class="flex items-center justify-between text-[11px]" role="status" aria-live="polite">
      <span class="text-ink-dim tracking-wide">
        VOICE CALIBRATION
        <span class="text-ink-faint font-sans normal-case">{{ mode === 'required' ? '— required by current model' : '— optional, may improve ambiguous speaker conditions' }}</span>
      </span>
      <span
        class="font-mono"
        :class="{ 'text-amber-400': enrollmentStatus === 'PLACEHOLDER', 'text-environment': enrollmentStatus === 'ENROLLING', 'text-wearer': enrollmentStatus === 'REAL' }"
      >
        {{ enrollmentStatus === 'PLACEHOLDER' ? 'PLACEHOLDER (not you)' : enrollmentStatus === 'ENROLLING' ? `RECORDING ${durationS}s...` : 'REAL (your voice)' }}
      </span>
    </div>
    <button
      :disabled="isRecording"
      class="w-full min-h-[44px] border border-panel-line text-ink-dim hover:border-environment hover:text-environment focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-environment focus-visible:ring-offset-2 focus-visible:ring-offset-panel-surface transition-colors rounded-md py-2.5 text-xs font-semibold tracking-wide disabled:opacity-50 disabled:cursor-not-allowed"
      @click="emit('record')"
    >
      {{ isRecording ? `RECORDING... ${durationS}s` : `${mode === 'required' ? 'ENROLL MY VOICE' : 'CALIBRATE (OPTIONAL)'} (${durationS}s)` }}
    </button>
  </div>

  <!-- Reserved for future passive personalization (section 41) -- only
       renders if the backend ever reports it supported; no fake
       "learning your voice" state exists today because no backend sets
       this true. -->
  <div v-if="supportsPassivePersonalization" class="text-[11px] text-ink-faint font-mono">
    PASSIVE PERSONALIZATION: OFF
  </div>
</template>
